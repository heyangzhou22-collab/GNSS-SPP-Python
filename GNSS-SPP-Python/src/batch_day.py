# -*- coding: utf-8 -*-
"""第三周收尾：全天2880历元伪距单点定位批处理 + 统计 + 绘图。
单频C1，广播星历，Klobuchar电离层 + Saastamoinen对流层改正，
高度角截止15°，地球自转改正，发射时刻迭代，过期星历剔除。"""
import os, glob, sys, io, contextlib, csv, math

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here); os.chdir(here)
_ROOT = os.path.dirname(here)

# import ephemeris 时屏蔽它的首历元打印
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    import ephemeris as E
from rinex_parser import epochs

OUT = os.path.join(_ROOT, 'results')
rows = []
n_fail = 0
for idx, ep in enumerate(epochs):
    try:
        r = E.solve_epoch(ep)
    except Exception:
        r = None
    if r is None:
        n_fail += 1
        continue
    lat, lon, h = E.ecef_to_blh(*r['pos'])
    rows.append({
        'idx': idx,
        'time': ep['time'].strftime('%H:%M:%S'),
        'hour': idx * 30 / 3600.0,
        'X': r['pos'][0], 'Y': r['pos'][1], 'Z': r['pos'][2],
        'lat': math.degrees(lat), 'lon': math.degrees(lon), 'h': h,
        'dX': r['pos'][0] - E.x0, 'dY': r['pos'][1] - E.y0, 'dZ': r['pos'][2] - E.z0,
        'cdt_m': r['cdt'], 'rms': r['rms'], 'nsat': r['nsat'],
        'pdop': r['pdop'], 'el_min': r['el_min'], 'niter': r['niter'],
    })

n_ok = len(rows)
print('全天历元: %d   成功: %d (%.1f%%)   失败: %d'
      % (len(epochs), n_ok, 100.0 * n_ok / len(epochs), n_fail))

# ---------- 统计 ----------
def stats(vals):
    n = len(vals); m = sum(vals) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1))
    return m, sd, min(vals), max(vals)

for k, name in [('dX', 'dX(m)'), ('dY', 'dY(m)'), ('dZ', 'dZ(m)'), ('rms', 'RMS(m)'),
                ('h', '大地高(m)'), ('cdt_m', '钟差(m)'), ('pdop', 'PDOP'), ('nsat', '卫星数')]:
    m, sd, mn, mx = stats([r[k] for r in rows])
    print('%-8s 均值 %9.2f   σ %7.2f   范围 [%8.2f, %8.2f]' % (name, m, sd, mn, mx))

# 平面/3D RMS（相对O头概略坐标；概略有~50m级近似，主要看离散度）
rms3d = math.sqrt(sum(r['dX']**2 + r['dY']**2 + r['dZ']**2 for r in rows) / n_ok)
print('3D位置RMS(相对概略坐标): %.2f m' % rms3d)
# 重复性：扣除均值后的离散
mx, sx, _, _ = stats([r['dX'] for r in rows]); my, sy, _, _ = stats([r['dY'] for r in rows]); mz, sz, _, _ = stats([r['dZ'] for r in rows])
print('定位重复性 σx/σy/σz = %.2f / %.2f / %.2f m' % (sx, sy, sz))

# ---------- 写CSV ----------
csv_path = os.path.join(OUT, 'spp_result_day274.csv')
with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print('已导出:', csv_path)

# ---------- 绘图 ----------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
# 中文字体
for fp in [r'C:\Windows\Fonts\msyh.ttc', r'C:\Windows\Fonts\simhei.ttf']:
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp)
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=fp).get_name()
        break
plt.rcParams['axes.unicode_minus'] = False

t = [r['hour'] for r in rows]
fig, ax = plt.subplots(3, 2, figsize=(15, 11))

# 1) 三方向误差时间序列
ax[0, 0].plot(t, [r['dX'] for r in rows], s_on := '.', ms=2, label='dX')
ax[0, 0].plot(t, [r['dY'] for r in rows], '.', ms=2, label='dY')
ax[0, 0].plot(t, [r['dZ'] for r in rows], '.', ms=2, label='dZ')
ax[0, 0].axhline(0, color='k', lw=0.6); ax[0, 0].set_title('ECEF坐标偏差时间序列(相对概略坐标)')
ax[0, 0].set_xlabel('GPS小时'); ax[0, 0].set_ylabel('m'); ax[0, 0].legend(markerscale=4); ax[0, 0].grid(alpha=.3)

# 2) 残差RMS时间序列
ax[0, 1].plot(t, [r['rms'] for r in rows], '.', ms=2, color='crimson')
ax[0, 1].set_title('每历元伪距残差RMS'); ax[0, 1].set_xlabel('GPS小时')
ax[0, 1].set_ylabel('m'); ax[0, 1].grid(alpha=.3)

# 3) 水平轨迹散点（相对均值，米）——看漂移
mlat, _, _, _ = stats([r['lat'] for r in rows]); mlon, _, _, _ = stats([r['lon'] for r in rows])
Re = 6378137.0
east = [(r['lon'] - mlon) * math.pi / 180 * Re * math.cos(mlat * math.pi / 180) for r in rows]
north = [(r['lat'] - mlat) * math.pi / 180 * Re for r in rows]
sc = ax[1, 0].scatter(east, north, c=t, cmap='viridis', s=4)
ax[1, 0].set_title('水平定位散点(相对全天均值, 颜色=GPS小时)')
ax[1, 0].set_xlabel('东 E (m)'); ax[1, 0].set_ylabel('北 N (m)'); ax[1, 0].axis('equal'); ax[1, 0].grid(alpha=.3)
plt.colorbar(sc, ax=ax[1, 0], label='小时')

# 4) 大地高时间序列
ax[1, 1].plot(t, [r['h'] for r in rows], '.', ms=2, color='seagreen')
ax[1, 1].set_title('大地高时间序列'); ax[1, 1].set_xlabel('GPS小时'); ax[1, 1].set_ylabel('m'); ax[1, 1].grid(alpha=.3)

# 5) 卫星数 & PDOP
ax[2, 0].plot(t, [r['nsat'] for r in rows], '.', ms=2, color='navy', label='卫星数')
ax2 = ax[2, 0].twinx(); ax2.plot(t, [r['pdop'] for r in rows], '.', ms=2, color='orange', label='PDOP')
ax[2, 0].set_title('参与卫星数 / PDOP'); ax[2, 0].set_xlabel('GPS小时')
ax[2, 0].set_ylabel('卫星数', color='navy'); ax2.set_ylabel('PDOP', color='orange'); ax[2, 0].grid(alpha=.3)

# 6) 残差分布直方图
allrms = [r['rms'] for r in rows]
ax[2, 1].hist(allrms, bins=60, color='purple', alpha=.75)
ax[2, 1].set_title('残差RMS分布 (中位 %.2f m)' % sorted(allrms)[len(allrms)//2])
ax[2, 1].set_xlabel('RMS (m)'); ax[2, 1].set_ylabel('历元数'); ax[2, 1].grid(alpha=.3)

fig.suptitle('BJFS站 2026-10-01(DOY274) 伪距单点定位(SPP)全天结果  单频C1+Klobuchar+Saastamoinen', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.98])
png_path = os.path.join(OUT, 'spp_result_day274.png')
fig.savefig(png_path, dpi=130)
print('已导出:', png_path)
