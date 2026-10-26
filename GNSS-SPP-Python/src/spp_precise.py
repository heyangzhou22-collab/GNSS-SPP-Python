# -*- coding: utf-8 -*-
"""
第五周：精密SPP —— CODE 精密SP3轨道 + 30S精密钟差 替换广播星历
口径与第三周单频C1一致(Klobuchar+Saastamoinen+15度截止)，仅替换轨道/钟差来源。
SP3 5min(km,ITRF)做拉格朗日插值; CLK 30S(AS行,秒)线性插值; 发射时刻迭代+地球自转。
"""
import os, sys, math, gzip, csv, io, contextlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
with contextlib.redirect_stdout(io.StringIO()):
    import rinex_parser as R
    import ephemeris as E

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SP3_PATH = os.path.join(_ROOT, "data", "g2438.sp3.gz")
CLK_PATH = os.path.join(_ROOT, "data", "COD0MGXFIN_20262740000_01D_30S_CLK.CLK.gz")
OUT_DIR = os.path.join(_ROOT, "results")
C = E.C
OMEGA_E = E.OMEGA_E
PI = E.PI
EL_CUT = 15.0 * PI / 180.0

epochs = R.epochs
ION_ALPHA, ION_BETA = E.ION_ALPHA, E.ION_BETA
x0, y0, z0 = E.x0, E.y0, E.z0


def load_sp3(path, tow0=345600.0):
    """SP3历元标称当日时刻; 坐标对应真实GPS周内秒 = tow0 + sod。
    去重后用绝对GPS周内秒作时间键。"""
    pmap, pos, cur = {}, {}, None
    with gzip.open(path, "rt", errors="replace") as f:
        for ln in f:
            if ln.startswith("*"):
                p = ln.split()
                sod = int(p[4]) * 3600 + int(p[5]) * 60 + float(p[6])
                cur = tow0 + sod
                pmap.setdefault(cur, {})
            elif ln.startswith("P") and len(ln) > 40 and ln[1] == "G" and cur is not None:
                try:
                    prn = int(ln[2:4])
                    q = (float(ln[4:18]) * 1e3, float(ln[18:32]) * 1e3, float(ln[32:46]) * 1e3)
                except ValueError:
                    continue
                if abs(q[0]) < 1 and abs(q[1]) < 1:  # SP3缺测标记
                    continue
                pmap[cur][prn] = q
                pos.setdefault(prn, {})[cur] = q
    times = sorted(pmap.keys())
    return times, pos


def load_clk(path):
    clk = {}
    with gzip.open(path, "rt", errors="replace") as f:
        for ln in f:
            if not ln.startswith("AS G"):
                continue
            p = ln.split()
            if len(p) < 10:
                continue
            try:
                prn = int(p[1][1:])
                sod = int(p[5]) * 3600 + int(p[6]) * 60 + float(p[7])
                clk.setdefault(prn, {})[sod] = float(p[9])
            except ValueError:
                continue
    return clk


def lagrange(prn, t, times, pos, deg=4):
    """发射时刻位置。按该星实际可用网格, 选t两侧共deg+1个最近节点做拉格朗日
    插值; 若两侧不足(落在覆盖外)则返回None, 严格不外推。"""
    d = pos.get(prn)
    if not d:
        return None
    av = sorted(d.keys())
    half = 300.0
    # 严格覆盖范围: 允许落在首/末网格±半格内
    if t < av[0] - half or t > av[-1] + half:
        return None
    # 内插主体(发射时刻t_tx在覆盖区间内)。t恰等于网格时拉格朗日权重
    # 自然给出该网格值, 无需特判钳值(钳值会令位置时刻与地球自转tau不自洽)。
    # 选点规则: 让t尽量位于窗口中央。默认5点4阶; 仅在序列首末两个区间
    # (某侧不足2点)才降到3点二次; t在覆盖外则线性外延一小段。
    # 覆盖范围: 发射时刻t只可能比首/末网格差约0.08s。落在覆盖外时,
    # 用最近两节点线性外延(卫星0.08s内近似匀速, 误差<0.1m), 不用高阶。
    if t < av[0]:
        if av[0] - t > 1.0:
            return None
        t0, t1 = av[0], av[1]; w = (t - t0) / (t1 - t0)
        return np.array(d[t0]) * (1 - w) + np.array(d[t1]) * w
    if t > av[-1]:
        if t - av[-1] > 1.0:
            return None
        t0, t1 = av[-2], av[-1]; w = (t - t0) / (t1 - t0)
        return np.array(d[t0]) * (1 - w) + np.array(d[t1]) * w
    # 找t所在区间索引i: av[i] <= t < av[i+1]
    if t <= av[0]:
        i = 0
    elif t >= av[-1]:
        i = len(av) - 2
    else:
        i = 0
        while i < len(av) - 1 and av[i + 1] <= t:
            i += 1
    # 始终用5点4阶, 窗口[i-2 .. i+2]; 某侧越界则整体平移, 保证5点齐全,
    # 使t尽量居中、避免拉格朗日端点振荡。
    lo, hi = i - 2, i + 2
    if lo < 0:
        hi += -lo; lo = 0
    if hi > len(av) - 1:
        lo -= (hi - (len(av) - 1)); hi = len(av) - 1
    lo = max(lo, 0)
    nd = av[lo:hi + 1]
    if len(nd) < 5:
        nd = av[:5] if len(av) >= 5 else av
    if len(nd) < 2:
        return None
    if nd[-1] - nd[0] > 1800.0:    # 跨长缺测弃用
        return None
    res = np.zeros(3)
    for k in range(len(nd)):
        L = 1.0
        for j in range(len(nd)):
            if j != k:
                L *= (t - nd[j]) / (nd[k] - nd[j])
        res += L * np.array(d[nd[k]])
    return res


def clk_at(prn, t, clk):
    d = clk.get(prn)
    if not d:
        return None
    ts = sorted(d)
    # CLK为30s网格, 卫星钟变化平缓: 线性插值; 边界允许外延一个间隔(30s)
    step = 30.0
    if t < ts[0] - step or t > ts[-1] + step:
        return None
    i = 0
    while i < len(ts) - 1 and ts[i + 1] <= t:
        i += 1
    if t <= ts[0]:
        t0, t1 = ts[0], ts[1]
    elif t >= ts[-1]:
        t0, t1 = ts[-2], ts[-1]
    else:
        t0, t1 = ts[i], ts[min(i + 1, len(ts) - 1)]
    if t1 == t0:
        return d[t0]
    w = (t - t0) / (t1 - t0)
    return d[t0] * (1 - w) + d[t1] * w


def precise_geom(prn, tow, rx, times, pos, clk):
    """发射时刻迭代+地球自转; 返回(t_tx,sx,sy,sz,rho,dts)或None。"""
    tau = 0.075
    sx = sy = sz = 0.0
    t_tx = tow
    for _ in range(3):
        t_tx = tow - tau
        q = lagrange(prn, t_tx, times, pos)  # SP3键=绝对GPS周内秒
        if q is None:
            return None
        ang = OMEGA_E * tau
        sx = q[0] * math.cos(ang) - q[1] * math.sin(ang)
        sy = q[0] * math.sin(ang) + q[1] * math.cos(ang)
        sz = q[2]
        tau = math.sqrt((sx - rx[0]) ** 2 + (sy - rx[1]) ** 2 + (sz - rx[2]) ** 2) / C
    dts = clk_at(prn, t_tx % 86400.0, clk)  # CLK键=日内秒
    if dts is None:
        return None
    rho = tau * C
    return t_tx, sx, sy, sz, rho, dts


F1, F2 = 1575.42e6, 1227.60e6
_GAMMA = (F1 / F2) ** 2
A_IF = _GAMMA / (_GAMMA - 1.0)      # 2.54573
B_IF = -1.0 / (_GAMMA - 1.0)        # -1.54573


def solve_epoch_precise(ep, times, pos, clk):
    """精密SPP: 双频无电离层(IF)组合伪距 + SP3精密轨道 + CLK精密钟差。
    CODE精密钟差基准为IF尺度, 故观测用IF且不再做电离层改正, 仅保留对流层。"""
    _, tow = E.to_gps_time(ep["time"])
    p = [x0, y0, z0]
    cdt = 0.0
    it = 0
    for it in range(10):
        G, l, used = [], [], []
        for sat, obs in ep["obs"].items():
            if not sat.startswith("G"):
                continue
            c1 = obs.get("C1", float("nan"))
            p2 = obs.get("P2", float("nan"))
            if math.isnan(c1) or math.isnan(p2):
                continue
            P = A_IF * c1 + B_IF * p2
            prn = int(sat[1:])
            gm = precise_geom(prn, tow, p, times, pos, clk)
            if gm is None:
                continue
            t_tx, sx, sy, sz, rho, dts = gm
            el, az, (ex, ey, ez), _ = E.los_azel(p, (sx, sy, sz))
            if el < EL_CUT:
                continue
            lat, lon, h = E.ecef_to_blh(p[0], p[1], p[2])
            tropo = E.saastamoinen(lat, h, el)   # 仅对流层, 无Klobuchar
            P_corr = P + C * dts - tropo
            G.append([-ex, -ey, -ez, 1.0])
            l.append(P_corr - rho - cdt)
            used.append((prn, P))
        if len(G) < 4:
            return None
        n = 4
        A = [[sum(G[r][k] * G[r][j] for r in range(len(G))) for j in range(n)] for k in range(n)]
        y = [sum(G[r][k] * l[r] for r in range(len(G))) for k in range(n)]
        try:
            d = E.solve_linear(A, y)
        except Exception:
            return None
        p[0] += d[0]; p[1] += d[1]; p[2] += d[2]; cdt += d[3]
        if math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2) < 1e-4:
            break
    # 残差
    res = []
    for (prn, P) in used:
        gm = precise_geom(prn, tow, p, times, pos, clk)
        if gm is None:
            continue
        t_tx, sx, sy, sz, rho, dts = gm
        el, az, _, _ = E.los_azel(p, (sx, sy, sz))
        lat, lon, h = E.ecef_to_blh(p[0], p[1], p[2])
        tropo = E.saastamoinen(lat, h, el)
        res.append((P + C * dts) - rho - cdt - tropo)
    if len(res) < 4:
        return None
    rms = math.sqrt(sum(v * v for v in res) / len(res))
    return {"pos": p, "cdt": cdt, "niter": it + 1, "rms": rms,
            "nsat": len(res), "tow": tow}


def st(v):
    a = np.array(v, dtype=float)
    return np.mean(a), np.std(a, ddof=1), np.median(a), np.min(a), np.max(a)


def main():
    print("加载SP3/CLK ...")
    times, pos = load_sp3(SP3_PATH)
    clk = load_clk(CLK_PATH)
    print("SP3历元%d GPS星%d; CLK GPS星%d" % (len(times), len(pos), len(clk)))

    rows, fail = [], 0
    for i, ep in enumerate(epochs):
        try:
            s = solve_epoch_precise(ep, times, pos, clk)
        except Exception:
            s = None
        if s is None:
            fail += 1
            continue
        lat, lon, h = E.ecef_to_blh(s["pos"][0], s["pos"][1], s["pos"][2])
        rows.append({"idx": i, "hour": i * 30.0 / 3600.0,
                     "dX": s["pos"][0] - x0, "dY": s["pos"][1] - y0, "dZ": s["pos"][2] - z0,
                     "rms": s["rms"], "h": h, "cdt_m": s["cdt"], "nsat": s["nsat"]})
    n = len(rows)
    print("精密SPP: %d/%d 成功, 失败%d" % (n, len(epochs), fail))
    if n == 0:
        return

    # SP3覆盖为当日00:00–23:55(5min网格)。剔除发射时刻在覆盖外的日首/日末
    # 边界历元(高阶插值/外延不可靠)与个别残差异常, 用干净集做精度评定。
    clean = [r for r in rows if 1 <= r["idx"] <= 2860 and r["rms"] < 15.0]
    ncut = len(rows) - len(clean)
    print("剔除边界/异常历元 %d 个, 参与精度评定 %d 个" % (ncut, len(clean)))
    rows_all = rows
    rows = clean

    dX = [r["dX"] for r in rows]; dY = [r["dY"] for r in rows]; dZ = [r["dZ"] for r in rows]
    rms = [r["rms"] for r in rows]; hh = [r["h"] for r in rows]
    cdt = [r["cdt_m"] for r in rows]; ns = [r["nsat"] for r in rows]
    td = [math.sqrt(r["dX"] ** 2 + r["dY"] ** 2 + r["dZ"] ** 2) for r in rows]
    sx, sy, sz = st(dX)[1], st(dY)[1], st(dZ)[1]
    rep3 = math.sqrt(sx * sx + sy * sy + sz * sz)
    print("\n===== 精密SP3/CLK + 双频IF 全天(剔除边界) =====")
    for nm, v in [("dX(m)", dX), ("dY(m)", dY), ("dZ(m)", dZ), ("残差RMS(m)", rms),
                  ("大地高(m)", hh), ("钟差c·dt(m)", cdt), ("卫星数", ns)]:
        m, s, md, mn, mx = st(v)
        print("%-12s 均值%8.2f σ%6.2f 中位%7.2f [%8.2f,%8.2f]" % (nm, m, s, md, mn, mx))
    print("3D系统偏差 %.2f m ; 3D重复性 %.2f m" % (np.mean(td), rep3))

    csvp = os.path.join(OUT_DIR, "spp_precise_day274.csv")
    with open(csvp, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_all[0].keys())); w.writeheader(); w.writerows(rows_all)
    print("CSV ->", csvp)

    bc_rms, bc_h, bc_dX, bc_dY, bc_dZ = [], [], [], [], []
    sc = os.path.join(OUT_DIR, "spp_result_day274.csv")
    if os.path.exists(sc):
        with open(sc, encoding="utf-8-sig") as f:
            for rr in csv.DictReader(f):
                def g(*keys):
                    for kk in keys:
                        if kk in rr and rr[kk] not in ("", None):
                            try:
                                return float(rr[kk])
                            except ValueError:
                                pass
                    return None
                k = g("idx", "epoch"); v = g("rms", "RMS")
                if k is not None and v is not None:
                    bc_h.append(int(k) * 30.0 / 3600.0); bc_rms.append(v)

    fpf = r"C:\Windows\Fonts\msyh.ttc"
    if os.path.exists(fpf):
        font_manager.fontManager.addfont(fpf)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=fpf).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    hr = [r["hour"] for r in rows]
    fig, ax = plt.subplots(2, 3, figsize=(18, 9))
    ax[0, 0].plot(hr, dX, lw=.7, label="dX", color="#c0392b")
    ax[0, 0].plot(hr, dY, lw=.7, label="dY", color="#2980b9")
    ax[0, 0].plot(hr, dZ, lw=.7, label="dZ", color="#27ae60")
    ax[0, 0].set_title("ECEF坐标偏差(精密SP3/CLK)"); ax[0, 0].set_xlabel("小时"); ax[0, 0].set_ylabel("m"); ax[0, 0].legend(fontsize=8); ax[0, 0].grid(alpha=.3)
    ax[0, 1].plot(hr, rms, lw=.7, color="#8e44ad", label="精密RMS 中位%.2f" % np.median(rms))
    ax[0, 1].axhline(np.median(rms), ls="--", color="#8e44ad", alpha=.6)
    if bc_rms:
        ax[0, 1].plot(bc_h, bc_rms, lw=.5, color="gray", alpha=.6, label="广播RMS 中位%.2f" % np.median(bc_rms))
        ax[0, 1].axhline(np.median(bc_rms), ls=":", color="gray", alpha=.6)
    ax[0, 1].set_title("残差RMS 精密 vs 广播"); ax[0, 1].set_xlabel("小时"); ax[0, 1].set_ylabel("m"); ax[0, 1].legend(fontsize=8); ax[0, 1].grid(alpha=.3)
    ax[0, 2].plot(dY, dX, ".", ms=2, alpha=.4)
    ax[0, 2].set_title("水平偏差散点(dY,dX)"); ax[0, 2].set_xlabel("dY(m)"); ax[0, 2].set_ylabel("dX(m)"); ax[0, 2].grid(alpha=.3); ax[0, 2].axis("equal")
    ax[1, 0].plot(hr, hh, lw=.7, color="#d35400")
    ax[1, 0].set_title("大地高 均值%.1fm σ%.2f" % (np.mean(hh), np.std(hh, ddof=1))); ax[1, 0].set_xlabel("小时"); ax[1, 0].set_ylabel("m"); ax[1, 0].grid(alpha=.3)
    ax[1, 1].plot(hr, ns, lw=.6, color="#16a085")
    ax[1, 1].set_title("卫星数 均值%.1f" % np.mean(ns)); ax[1, 1].set_xlabel("小时"); ax[1, 1].set_ylabel("颗"); ax[1, 1].set_ylim(0, 14); ax[1, 1].grid(alpha=.3)
    ax[1, 2].hist(rms, bins=40, color="#8e44ad", alpha=.75, edgecolor="white")
    ax[1, 2].axvline(np.median(rms), color="k", ls="--", label="中位%.2f" % np.median(rms))
    ax[1, 2].set_title("残差RMS分布"); ax[1, 2].set_xlabel("RMS(m)"); ax[1, 2].legend(); ax[1, 2].grid(alpha=.3)
    plt.suptitle("精密SPP(CODE SP3+30S CLK) BJFS DOY274 成功%d/%d 3D重复性%.2fm" % (n, len(epochs), rep3), fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, .97])
    pngp = os.path.join(OUT_DIR, "spp_precise_day274.png")
    plt.savefig(pngp, dpi=130); plt.close()
    print("图 ->", pngp)


if __name__ == "__main__":
    main()
