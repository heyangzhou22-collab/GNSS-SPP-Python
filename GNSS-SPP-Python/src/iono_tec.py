# -*- coding: utf-8 -*-
"""第四周补充：双频 C1/P2 反演电离层 VTEC（载波相位 Hatch 平滑版）。

流程：
  1) 逐星连续弧段做载波相位平滑伪距(Hatch)：
       Ps(k)=P(k)/N + (N-1)/N · [Ps(k-1) + λ·(L(k)-L(k-1))]
     用 L1-L2 几何无关组合 G=λ1L1-λ2L2 的历元跳变检测周跳(阈值0.10m)，检出即重置弧段。
  2) 平滑伪距反演 L1 斜向电离层延迟 I1 = gamma/(gamma-1)·(Ps2-Ps1)，
     单层薄壳映射函数转垂直 VTEC，逐星去常数 DCB(全天中位)。
  1 TECU=1e16 el/m^2，对应 L1 延迟 0.1624 m。
"""
import os, glob, sys, io, contextlib, math
from collections import defaultdict

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here); os.chdir(here)
_ROOT = os.path.dirname(here)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    import ephemeris as E
from rinex_parser import epochs

F1, F2 = 1575.42e6, 1227.60e6
GAMMA = (F1 / F2) ** 2
A_IF = GAMMA / (GAMMA - 1.0)
LAM1, LAM2 = E.C / F1, E.C / F2
M_PER_TECU = 40.3e16 / (F1 ** 2)          # ≈0.1624 m/TECU
RH, H_ION = 6371.0e3, 350.0e3
SLIP_M = 0.10                             # 几何无关组合周跳阈值 m
GAP_IDX = 2                               # 历元间隔>60s 视为新弧段

# 先收集每颗GPS星全天观测（含载波相位）与几何高度角
track = defaultdict(list)   # prn -> [(idx,hour,P1,P2,L1cyc,L2cyc,el)]
rx0 = [E.x0, E.y0, E.z0]
for idx, ep in enumerate(epochs):
    _, tow = E.to_gps_time(ep['time'])
    hour = idx * 30 / 3600.0
    for sat, obs in ep['obs'].items():
        if not sat.startswith('G'):
            continue
        P1 = obs.get('C1', float('nan')); P2 = obs.get('P2', float('nan'))
        L1 = obs.get('L1', float('nan')); L2 = obs.get('L2', float('nan'))
        if any(math.isnan(v) for v in (P1, P2, L1, L2)) or L1 == 0 or L2 == 0:
            continue
        prn = int(sat[1:])
        gm = E.geom(prn, tow, rx0)
        if gm is None:
            continue
        eph, t_tx, sx, sy, sz, Ecc, rho = gm
        dte = t_tx - eph['Toe']
        if dte > 302400: dte -= 604800
        if dte < -302400: dte += 604800
        if abs(dte) > 7200:
            continue
        el, _, _, _ = E.los_azel(rx0, (sx, sy, sz))
        if el < 15 * math.pi / 180:
            continue
        track[prn].append((idx, hour, P1, P2, L1, L2, el))


def hatch_smooth(arc):
    """对一段连续弧段分别平滑 P1/P2。返回 [(idx,hour,Ps1,Ps2,el)]。"""
    out = []
    ps1 = ps2 = None
    l1p = l2p = None
    n = 0
    nslip = 0
    for (idx, hour, P1, P2, L1, L2, el) in arc:
        G = LAM1 * L1 - LAM2 * L2
        slip = (ps1 is None) or (l1p is not None and abs(G - Gp) > SLIP_M)
        Gp = G
        if slip:
            ps1, ps2, n = P1, P2, 1
            if l1p is not None:
                nslip += 1
        else:
            n = min(n + 1, 100)
            w = 1.0 / n
            ps1 = w * P1 + (1 - w) * (ps1 + LAM1 * (L1 - l1p))
            ps2 = w * P2 + (1 - w) * (ps2 + LAM2 * (L2 - l2p))
        l1p, l2p = L1, L2
        out.append((idx, hour, ps1, ps2, el))
    return out, nslip


# 弧段划分（时间间断）+ Hatch 平滑 + 反演
raw = defaultdict(list)       # prn -> [(hour, Iv_m, el)]
tot_slip = 0
for prn, obs in track.items():
    obs.sort(key=lambda x: x[0])
    arcs, cur, last_idx = [], [], None
    for rec in obs:
        if last_idx is not None and rec[0] - last_idx > GAP_IDX:
            arcs.append(cur); cur = []
        cur.append(rec); last_idx = rec[0]
    if cur:
        arcs.append(cur)
    for arc in arcs:
        sm, nslip = hatch_smooth(arc)
        tot_slip += nslip
        for (idx, hour, Ps1, Ps2, el) in sm:
            I1 = A_IF * (Ps2 - Ps1)                 # L1 斜向延迟 m
            zp = math.asin(RH / (RH + H_ION) * math.cos(el))
            Iv = I1 * math.cos(zp)                 # 垂直延迟 m
            raw[prn].append((hour, Iv, math.degrees(el), idx))

# 逐星去常数 DCB（全天中位）
med = {}
for prn, pts in raw.items():
    vals = sorted(p[1] for p in pts)
    med[prn] = vals[len(vals) // 2]

series = defaultdict(list)
all_v = []
for prn, pts in raw.items():
    for hour, Iv, eldeg, idx in pts:
        v = (Iv - med[prn]) / M_PER_TECU
        series[prn].append((hour, v, eldeg))
        all_v.append((hour, v))

bins = defaultdict(list)
for hour, v in all_v:
    bins[round(hour * 2) / 2].append(v)
bh = sorted(bins)
bv = [sorted(bins[h])[len(bins[h]) // 2] for h in bh]
print('参与卫星 %d 颗  样本 %d  检出周跳 %d 次' % (len(series), len(all_v), tot_slip))
print('平滑后 VTEC 相对范围 [%.1f, %.1f] TECU  峰谷 %.1f TECU' % (min(bv), max(bv), max(bv) - min(bv)))
pk = bh[bv.index(max(bv))]
print('VTEC 峰值出现在 GPS %.1f h (北京 %.1f h)' % (pk, (pk + 8) % 24))

# ---------- 绘图 ----------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
for fp in [r'C:\Windows\Fonts\msyh.ttc', r'C:\Windows\Fonts\simhei.ttf']:
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp)
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=fp).get_name()
        break
plt.rcParams['axes.unicode_minus'] = False

medcurve = dict(zip(bh, bv))
fig, ax = plt.subplots(1, 2, figsize=(15, 5.2))
cmap = plt.cm.tab20
for i, prn in enumerate(sorted(series)):
    pts = sorted(series[prn])
    ax[0].plot([p[0] for p in pts], [p[1] for p in pts], '.', ms=2, alpha=.5, color=cmap(i % 20))
ax[0].plot(bh, bv, color='k', lw=2.4, label='全天中位曲线')
ax[0].axhline(0, color='r', lw=.8, ls='--')
ax[0].set_title('Hatch平滑后双频反演垂直电离层 VTEC（各星已去DCB）')
ax[0].set_xlabel('GPS小时 (括号为北京时)'); ax[0].set_ylabel('相对 VTEC (TECU)')
ax[0].set_xticks(range(0, 25, 2))
ax[0].set_xticklabels(['%d\n(%d)' % (h, (h + 8) % 24) for h in range(0, 25, 2)])
ax[0].grid(alpha=.3); ax[0].legend()

hv = [(eldeg, v - medcurve[round(hour * 2) / 2])
      for prn in series for (hour, v, eldeg) in series[prn]]
ax[1].scatter([p[0] for p in hv], [p[1] for p in hv], s=3, alpha=.2)
ax[1].axhline(0, color='r', lw=.8, ls='--')
ax[1].set_title('去周日趋势后 VTEC 残差随高度角（平滑后应明显收敛）')
ax[1].set_xlabel('高度角 (°)'); ax[1].set_ylabel('VTEC残差 (TECU)'); ax[1].grid(alpha=.3)

fig.suptitle('BJFS站 2026-10-01(DOY274) 载波相位平滑双频反演电离层TEC  1 TECU≈0.162m(L1)', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = os.path.join(_ROOT, 'results', 'iono_vtec_day274.png')
fig.savefig(out, dpi=130)
print('已导出:', out)
