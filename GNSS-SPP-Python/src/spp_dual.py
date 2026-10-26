# -*- coding: utf-8 -*-
"""第四周：双频无电离层组合(IF)伪距单点定位，全天批处理 + 与单频(Klobuchar)对比。

IF 观测： P_IF = a*C1 + b*P2,  a = gamma/(gamma-1)=2.54573, b = -1/(gamma-1)=-1.54573
  - 消除一阶电离层延迟（模型误差），观测噪声放大约 sqrt(a^2+b^2)=2.95 倍。
  - 广播钟差三参数 af 参考 L1-L2 无电离层时间尺度（已含 -Tgd），IF 组合直接用、不再另减 Tgd。
  - 只保留 Saastamoinen 对流层；不用 Klobuchar。
  - 高度角截止 15°、地球自转改正、发射时刻迭代、过期星历剔除，与单频完全一致。
单频对照：同一天、同样 15° 截止、Klobuchar+Saastamoinen，重跑保证口径一致。
"""
import os, glob, sys, io, contextlib, csv, math

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here); os.chdir(here)
_ROOT = os.path.dirname(here)

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    import ephemeris as E
from rinex_parser import epochs

F1, F2 = 1575.42e6, 1227.60e6
GAMMA = (F1 / F2) ** 2          # 1.64694
A_IF = GAMMA / (GAMMA - 1.0)    # 2.54573
B_IF = -1.0 / (GAMMA - 1.0)     # -1.54573
NOISE_IF = math.sqrt(A_IF ** 2 + B_IF ** 2)
EL_CUT = 15.0


def sat_clock_if(eph, t_tx, Ecc):
    """IF 时间尺度钟差(秒)：三参数 + 相对论，不减 Tgd（af 已对应 L1-L2 IF 尺度）。"""
    dt = t_tx - eph['Toe']
    dtr = -2 * math.sqrt(E.MU) * eph['sqrtA'] * eph['e'] * math.sin(Ecc) / E.C ** 2
    return eph['af0'] + eph['af1'] * dt + eph['af2'] * dt * dt + dtr


def _dte_ok(t_tx, Toe):
    dte = t_tx - Toe
    if dte > 302400:
        dte -= 604800
    if dte < -302400:
        dte += 604800
    return abs(dte) <= 7200


def solve_epoch_if(ep, el_cut_deg=15.0):
    """双频 IF 组合单历元解算。失败返回 None。"""
    _, tow = E.to_gps_time(ep['time'])
    pos = [E.x0, E.y0, E.z0]
    cdt = 0.0
    el_cut = el_cut_deg * E.PI / 180.0
    it = 0
    for it in range(10):
        G, l = [], []
        for sat, obs in ep['obs'].items():
            if not sat.startswith('G'):
                continue
            P1 = obs.get('C1', float('nan'))
            P2 = obs.get('P2', float('nan'))
            if math.isnan(P1) or math.isnan(P2):
                continue                      # 仅双频齐全的星参与 IF
            P = A_IF * P1 + B_IF * P2
            prn = int(sat[1:])
            gm = E.geom(prn, tow, pos)
            if gm is None:
                continue
            eph, t_tx, sx, sy, sz, Ecc, rho = gm
            if not _dte_ok(t_tx, eph['Toe']):
                continue
            el, az, (ex, ey, ez), _ = E.los_azel(pos, (sx, sy, sz))
            if el < el_cut:
                continue
            dts = sat_clock_if(eph, t_tx, Ecc)
            P_corr = P + E.C * dts
            lat0, lon0, h0 = E.ecef_to_blh(pos[0], pos[1], pos[2])
            tropo = E.saastamoinen(lat0, h0, el)   # 只改正对流层，无电离层
            P_corr -= tropo
            G.append([-ex, -ey, -ez, 1.0])
            l.append(P_corr - rho - cdt)
        if len(G) < 4:
            return None
        n = 4
        Nm = [[sum(G[r][k] * G[r][j] for r in range(len(G))) for j in range(n)] for k in range(n)]
        yv = [sum(G[r][k] * l[r] for r in range(len(G))) for k in range(n)]
        d = E.solve_linear(Nm, yv)
        pos[0] += d[0]; pos[1] += d[1]; pos[2] += d[2]
        cdt += d[3]
        if math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2) < 1e-4:
            break

    # 收敛后残差 / PDOP
    res, used, els, GQ = [], [], [], []
    for sat, obs in ep['obs'].items():
        if not sat.startswith('G'):
            continue
        P1 = obs.get('C1', float('nan'))
        P2 = obs.get('P2', float('nan'))
        if math.isnan(P1) or math.isnan(P2):
            continue
        P = A_IF * P1 + B_IF * P2
        prn = int(sat[1:])
        gm = E.geom(prn, tow, pos)
        if gm is None:
            continue
        eph, t_tx, sx, sy, sz, Ecc, rho = gm
        if not _dte_ok(t_tx, eph['Toe']):
            continue
        el, az, (ex, ey, ez), _ = E.los_azel(pos, (sx, sy, sz))
        if el < el_cut:
            continue
        lat0, lon0, h0 = E.ecef_to_blh(pos[0], pos[1], pos[2])
        tropo = E.saastamoinen(lat0, h0, el)
        dts = sat_clock_if(eph, t_tx, Ecc)
        res.append((P + E.C * dts) - rho - cdt - tropo)
        used.append(sat); els.append(math.degrees(el))
        GQ.append([-ex, -ey, -ez, 1.0])
    if len(res) < 4:
        return None
    n = 4
    Nm = [[sum(GQ[r][k] * GQ[r][j] for r in range(len(GQ))) for j in range(n)] for k in range(n)]
    try:
        Q = E.mat_inverse(Nm)
        pdop = math.sqrt(max(0.0, Q[0][0] + Q[1][1] + Q[2][2]))
    except Exception:
        pdop = float('nan')
    rms = math.sqrt(sum(v * v for v in res) / len(res))
    return {'time': ep['time'], 'pos': pos, 'cdt': cdt, 'niter': it + 1,
            'rms': rms, 'used': used, 'res': res, 'pdop': pdop,
            'el_min': min(els), 'nsat': len(used), 'tow': tow}


def run_all():
    print('IF 系数: a=%.5f b=%.5f  理论噪声放大 %.2f 倍' % (A_IF, B_IF, NOISE_IF))
    rows_if, rows_sf = [], []
    nf_if = nf_sf = 0
    for idx, ep in enumerate(epochs):
        # 双频 IF
        try:
            ri = solve_epoch_if(ep)
        except Exception:
            ri = None
        if ri is None:
            nf_if += 1
        else:
            lat, lon, h = E.ecef_to_blh(*ri['pos'])
            rows_if.append(_row(idx, ep, ri, lat, lon, h))
        # 单频对照（同 15° 截止）
        try:
            rs = E.solve_epoch(ep, el_cut_deg=15.0, use_iono=True, use_tropo=True, obs_key='C1')
        except Exception:
            rs = None
        if rs is None:
            nf_sf += 1
        else:
            lat, lon, h = E.ecef_to_blh(*rs['pos'])
            rows_sf.append(_row(idx, ep, rs, lat, lon, h))
    print('双频IF  成功 %d/%d (%.1f%%)  失败 %d' % (len(rows_if), len(epochs), 100.0 * len(rows_if) / len(epochs), nf_if))
    print('单频C1  成功 %d/%d (%.1f%%)  失败 %d' % (len(rows_sf), len(epochs), 100.0 * len(rows_sf) / len(epochs), nf_sf))
    return rows_if, rows_sf


def _row(idx, ep, r, lat, lon, h):
    return {
        'idx': idx, 'time': ep['time'].strftime('%H:%M:%S'), 'hour': idx * 30 / 3600.0,
        'X': r['pos'][0], 'Y': r['pos'][1], 'Z': r['pos'][2],
        'lat': math.degrees(lat), 'lon': math.degrees(lon), 'h': h,
        'dX': r['pos'][0] - E.x0, 'dY': r['pos'][1] - E.y0, 'dZ': r['pos'][2] - E.z0,
        'cdt_m': r['cdt'], 'rms': r['rms'], 'nsat': r['nsat'],
        'pdop': r['pdop'], 'el_min': r['el_min'], 'niter': r['niter'],
    }


def stats(vals):
    n = len(vals); m = sum(vals) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1))
    return m, sd, min(vals), max(vals)


def report(tag, rows):
    print('\n==== %s ====' % tag)
    for k, name in [('dX', 'dX'), ('dY', 'dY'), ('dZ', 'dZ'), ('rms', '残差RMS'),
                    ('h', '大地高'), ('cdt_m', '钟差'), ('pdop', 'PDOP'), ('nsat', '卫星数')]:
        m, sd, mn, mx = stats([r[k] for r in rows])
        print('%-8s 均值 %9.2f   σ %7.2f   范围 [%8.2f, %8.2f]' % (name, m, sd, mn, mx))
    sx = stats([r['dX'] for r in rows])[1]; sy = stats([r['dY'] for r in rows])[1]; sz = stats([r['dZ'] for r in rows])[1]
    rep = math.sqrt(sx * sx + sy * sy + sz * sz)
    rms3d = math.sqrt(sum(r['dX'] ** 2 + r['dY'] ** 2 + r['dZ'] ** 2 for r in rows) / len(rows))
    rms_med = sorted(r['rms'] for r in rows)[len(rows) // 2]
    print('定位重复性 σx/σy/σz = %.2f / %.2f / %.2f m   3D重复性 %.2f m' % (sx, sy, sz, rep))
    print('3D位置RMS(相对概略坐标,含系统偏差) %.2f m   残差RMS中位 %.2f m' % (rms3d, rms_med))
    return dict(sx=sx, sy=sy, sz=sz, rep3d=rep, rms3d=rms3d, rms_med=rms_med,
                sh=stats([r['h'] for r in rows])[1], srms=stats([r['rms'] for r in rows])[1])


def write_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print('已导出:', path)


def plot_compare(rows_sf, rows_if, OUT):
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
    RE = 6378137.0

    def enu(rows):
        mlat = stats([r['lat'] for r in rows])[0]; mlon = stats([r['lon'] for r in rows])[0]
        e = [(r['lon'] - mlon) * math.pi / 180 * RE * math.cos(mlat * math.pi / 180) for r in rows]
        n = [(r['lat'] - mlat) * math.pi / 180 * RE for r in rows]
        u = [r['h'] - stats([x['h'] for x in rows])[0] for r in rows]
        return e, n, u

    e1, n1, u1 = enu(rows_sf)
    e2, n2, u2 = enu(rows_if)
    t = [r['hour'] for r in rows_sf]
    t2 = [r['hour'] for r in rows_if]

    fig, ax = plt.subplots(3, 2, figsize=(15, 12))

    # 1) 水平散点叠加
    ax[0, 0].scatter(e1, n1, s=4, alpha=.35, color='steelblue', label='单频C1+Klobuchar')
    ax[0, 0].scatter(e2, n2, s=4, alpha=.35, color='crimson', label='双频IF组合')
    ax[0, 0].set_title('水平定位散点(相对各自全天均值)'); ax[0, 0].set_xlabel('东 E (m)'); ax[0, 0].set_ylabel('北 N (m)')
    ax[0, 0].axis('equal'); ax[0, 0].grid(alpha=.3); ax[0, 0].legend(markerscale=3)

    # 2) 垂直(大地高去均值)时间序列叠加
    ax[0, 1].plot(t, u1, '.', ms=2, alpha=.5, color='steelblue', label='单频')
    ax[0, 1].plot(t2, u2, '.', ms=2, alpha=.5, color='crimson', label='双频IF')
    ax[0, 1].axhline(0, color='k', lw=.6); ax[0, 1].set_title('大地高偏差时间序列(去均值)——电离层主要影响垂直向')
    ax[0, 1].set_xlabel('GPS小时'); ax[0, 1].set_ylabel('m'); ax[0, 1].grid(alpha=.3); ax[0, 1].legend(markerscale=3)

    # 3) 残差RMS时间序列
    ax[1, 0].plot(t, [r['rms'] for r in rows_sf], '.', ms=2, alpha=.5, color='steelblue', label='单频')
    ax[1, 0].plot(t2, [r['rms'] for r in rows_if], '.', ms=2, alpha=.5, color='crimson', label='双频IF')
    ax[1, 0].set_title('每历元伪距残差RMS（IF噪声放大约2.98倍）')
    ax[1, 0].set_xlabel('GPS小时'); ax[1, 0].set_ylabel('m'); ax[1, 0].grid(alpha=.3); ax[1, 0].legend(markerscale=3)

    # 4) 残差RMS直方图叠加
    ax[1, 1].hist([r['rms'] for r in rows_sf], bins=60, alpha=.55, color='steelblue', label='单频')
    ax[1, 1].hist([r['rms'] for r in rows_if], bins=60, alpha=.55, color='crimson', label='双频IF')
    ax[1, 1].set_title('残差RMS分布对比'); ax[1, 1].set_xlabel('RMS (m)'); ax[1, 1].set_ylabel('历元数')
    ax[1, 1].grid(alpha=.3); ax[1, 1].legend()

    # 5) 重复性σ柱状图（x/y/z/3D）
    met_sf = _metrics(rows_sf); met_if = _metrics(rows_if)
    labels = ['σX', 'σY', 'σZ', '3D重复']
    v_sf = [met_sf['sx'], met_sf['sy'], met_sf['sz'], met_sf['rep3d']]
    v_if = [met_if['sx'], met_if['sy'], met_if['sz'], met_if['rep3d']]
    xb = range(len(labels)); w = .38
    b1 = ax[2, 0].bar([i - w / 2 for i in xb], v_sf, w, color='steelblue', label='单频')
    b2 = ax[2, 0].bar([i + w / 2 for i in xb], v_if, w, color='crimson', label='双频IF')
    ax[2, 0].bar_label(b1, fmt='%.2f', fontsize=8); ax[2, 0].bar_label(b2, fmt='%.2f', fontsize=8)
    ax[2, 0].set_xticks(list(xb)); ax[2, 0].set_xticklabels(labels)
    ax[2, 0].set_title('定位重复性对比（越小越稳）'); ax[2, 0].set_ylabel('m'); ax[2, 0].grid(alpha=.3, axis='y'); ax[2, 0].legend()

    # 6) ECEF三方向去均值离散（单频 vs 双频，取σ文字+垂直时序已给，这里放PDOP/星数对比）
    ax[2, 1].plot(t, [r['nsat'] for r in rows_sf], '.', ms=2, alpha=.4, color='steelblue', label='单频星数')
    ax[2, 1].plot(t2, [r['nsat'] for r in rows_if], '.', ms=2, alpha=.4, color='crimson', label='双频星数')
    ax[2, 1].set_title('参与解算卫星数（双频需C1/P2齐全，略少）')
    ax[2, 1].set_xlabel('GPS小时'); ax[2, 1].set_ylabel('卫星数'); ax[2, 1].grid(alpha=.3); ax[2, 1].legend(markerscale=3)

    fig.suptitle('BJFS站 2026-10-01(DOY274) 单频SPP vs 双频无电离层组合(IF) 全天2880历元对比', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    png = os.path.join(OUT, 'spp_dual_vs_single_day274.png')
    fig.savefig(png, dpi=130)
    print('已导出:', png)


def _metrics(rows):
    sx = stats([r['dX'] for r in rows])[1]; sy = stats([r['dY'] for r in rows])[1]; sz = stats([r['dZ'] for r in rows])[1]
    return dict(sx=sx, sy=sy, sz=sz, rep3d=math.sqrt(sx * sx + sy * sy + sz * sz))


if __name__ == '__main__':
    OUT = os.path.join(_ROOT, 'results')
    rows_if, rows_sf = run_all()
    m_sf = report('单频 C1 + Klobuchar电离层 + Saastamoinen对流层', rows_sf)
    m_if = report('双频 无电离层组合 IF + Saastamoinen对流层', rows_if)

    print('\n############ 关键对比 ############')
    print('指标            单频      双频IF')
    for key, lab in [('sx', 'σX(m)'), ('sy', 'σY(m)'), ('sz', 'σZ(m)'), ('rep3d', '3D重复性'),
                     ('sh', '大地高σ'), ('rms_med', '残差RMS中位'), ('srms', '残差RMSσ')]:
        print('%-12s %8.2f %9.2f' % (lab, m_sf[key], m_if[key]))

    write_csv(os.path.join(OUT, 'spp_single_day274_cmp.csv'), rows_sf)
    write_csv(os.path.join(OUT, 'spp_if_day274.csv'), rows_if)
    plot_compare(rows_sf, rows_if, OUT)
