# -*- coding: utf-8 -*-
"""
GNSS 第三周：伪距单点定位（SPP）—— 完整版
- RINEX 2.11 广播星历解析（GPS 8行/组，GLONASS 4行/组自动跳过）
- 开普勒轨道计算卫星位置（含六摄动、地球自转改正）
- 卫星钟差改正（三参数 + 相对论 - Tgd）
- Klobuchar 电离层改正（N文件头 ION ALPHA/BETA，单频L1=C1）
- Saastamoinen 对流层改正（标准大气模型）
- 高度角截止、泰勒展开 + 最小二乘逐历元解算坐标与钟差、PDOP
关键：RINEX 观测历元时间标签本身就是 GPS 时，与星历 Toe 同时间系统，
      不要额外加 18 秒闰秒（加了会把卫星位置整体推偏，残差散到公里级）。
"""
import os
from datetime import datetime
from collections import defaultdict
import math

from rinex_parser import epochs

# ---------------- 常量 ----------------
C = 299792458.0               # 光速 m/s
MU = 3.986005e14              # 地球引力常数 m^3/s^2
OMEGA_E = 7.2921151467e-5     # 地球自转角速度 rad/s
RE_WGS84 = 6378137.0          # WGS84 长半轴 m
F_WGS84 = 1.0 / 298.257223563 # WGS84 扁率
E2 = F_WGS84 * (2 - F_WGS84)  # 第一偏心率平方
PI = 3.1415926535898
SC = PI / 180.0               # 度 -> 半圆弧（Klobuchar 专用单位）

NAV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'brdc2740.26n')
# 测站相位中心近似坐标（O文件头 APPROX POSITION XYZ + ANTENNA DELTA H 已并入观测，这里用参考真值评估）
x0, y0, z0 = -2148744.8004, 4426641.2527, 4044655.8579


# ==================== 1. 广播星历解析 ====================
def pf(field):
    """N文件 D 格式科学计数法转 float"""
    s = field.strip().replace('D', 'E')
    return float(s) if s else float('nan')


def is_header(line):
    """判断是不是星历首行：月/日/时字段合法"""
    try:
        return (1 <= int(line[6:8]) <= 12 and
                1 <= int(line[9:11]) <= 31 and
                0 <= int(line[12:14]) <= 23 and
                '.' in line[17:22])
    except ValueError:
        return False


def load_nav(path):
    """解析广播星历，返回 (records, ion_alpha, ion_beta)"""
    with open(path, 'r') as f:
        lines = f.readlines()

    ion_alpha = [0.0, 0.0, 0.0, 0.0]
    ion_beta = [0.0, 0.0, 0.0, 0.0]
    i = 0
    while 'END OF HEADER' not in lines[i]:
        ln = lines[i]
        if 'ION ALPHA' in ln:
            ion_alpha = [pf(ln[2 + 12*j: 14 + 12*j]) for j in range(4)]
        elif 'ION BETA' in ln:
            ion_beta = [pf(ln[2 + 12*j: 14 + 12*j]) for j in range(4)]
        i += 1
    i += 1

    records = []
    while i < len(lines):
        l1 = lines[i].rstrip('\n')
        if not l1.strip() or not is_header(l1):
            i += 1
            continue
        syschar = l1[0] if l1[0].isalpha() else 'G'   # 无字母 = GPS/SBAS
        nlines = 4 if syschar == 'R' else 8            # GLONASS 只有4行
        if i + nlines > len(lines):
            break

        if syschar == 'G':
            prn = int(l1[0:3] if l1[0] != ' ' else l1[1:3])
            if 1 <= prn <= 32:
                t = l1[2:22].split()
                yr, mon, day, hr, minute = [int(x) for x in t[:5]]
                sec = float(t[5])
                year = 2000 + yr if yr < 80 else 1900 + yr
                toc = datetime(year, mon, day, hr, minute, int(round(sec)))
                af0, af1, af2 = pf(l1[22:41]), pf(l1[41:60]), pf(l1[60:79])
                v = []
                for k in range(1, 8):
                    l = lines[i + k]
                    v += [pf(l[3 + 19*j: 22 + 19*j]) for j in range(4)]
                records.append({
                    'prn': prn, 'toc': toc,
                    'af0': af0, 'af1': af1, 'af2': af2,
                    'IODE': v[0],  'Crs': v[1],  'dn': v[2],   'M0': v[3],
                    'Cuc': v[4],   'e': v[5],    'Cus': v[6],  'sqrtA': v[7],
                    'Toe': v[8],   'Cic': v[9],  'Omega0': v[10], 'Cis': v[11],
                    'i0': v[12],   'Crc': v[13], 'omega': v[14],  'Omegadot': v[15],
                    'idot': v[16], 'week': int(v[18]),
                    'Tgd': v[22],  'tom': v[26],
                })
        i += nlines
    return records, ion_alpha, ion_beta


records, ION_ALPHA, ION_BETA = load_nav(NAV_PATH)
eph_by_sat = defaultdict(list)
for rec in records:
    eph_by_sat[rec['prn']].append(rec)


# ==================== 2. 时间换算 ====================
GPS0 = datetime(1980, 1, 6)

def to_gps_time(t):
    """民用时间 -> (GPS周, 周内秒)。RINEX历元标签本身即GPS时，不加闰秒。"""
    days = (t - GPS0).days
    sod = t.hour * 3600 + t.minute * 60 + t.second
    week = days // 7
    tow = (days % 7) * 86400 + sod
    return week, tow


def find_eph(prn, tow):
    """选 Toe 周内秒离计算时刻最近的一组星历；无该星历史记录时返回 None"""
    if prn not in eph_by_sat:
        return None
    best, bd = None, 1e18
    for e in eph_by_sat[prn]:
        d = abs(tow - e['Toe'])
        d = min(d, 604800 - d)      # 处理周界
        if d < bd:
            best, bd = e, d
    return best


# ==================== 3. 开普勒轨道计算 ====================
def sat_pos(eph, t):
    """eph: 一组星历；t: 计算时刻GPS周内秒。返回 ECEF x,y,z 与偏近点角 E"""
    A = eph['sqrtA'] ** 2
    n = math.sqrt(MU / A ** 3) + eph['dn']      # 平运动角速度 + 改正
    tk = t - eph['Toe']
    if tk > 302400:  tk -= 604800               # 跨周处理
    if tk < -302400: tk += 604800

    # 平近点角 -> 迭代解偏近点角 E（M = E - e·sinE）
    Mk = eph['M0'] + n * tk
    E = Mk
    for _ in range(20):
        E2 = Mk + eph['e'] * math.sin(E)
        if abs(E2 - E) < 1e-13:
            E = E2
            break
        E = E2

    # 偏近点角 -> 真近点角 ν -> 升交角距 φ
    cos_v = (math.cos(E) - eph['e']) / (1 - eph['e'] * math.cos(E))
    sin_v = math.sqrt(1 - eph['e'] ** 2) * math.sin(E) / (1 - eph['e'] * math.cos(E))
    phi = math.atan2(sin_v, cos_v) + eph['omega']

    # 六个摄动改正项
    du = eph['Cus'] * math.sin(2 * phi) + eph['Cuc'] * math.cos(2 * phi)
    dr = eph['Crs'] * math.sin(2 * phi) + eph['Crc'] * math.cos(2 * phi)
    di = eph['Cis'] * math.sin(2 * phi) + eph['Cic'] * math.cos(2 * phi)
    u = phi + du
    rk = A * (1 - eph['e'] * math.cos(E)) + dr
    inc = eph['i0'] + eph['idot'] * tk + di

    # 轨道平面内坐标
    xp = rk * math.cos(u)
    yp = rk * math.sin(u)

    # 升交点经度（含地球自转）-> ECEF
    Om = eph['Omega0'] + (eph['Omegadot'] - OMEGA_E) * tk - OMEGA_E * eph['Toe']
    x = xp * math.cos(Om) - yp * math.cos(inc) * math.sin(Om)
    y = xp * math.sin(Om) + yp * math.cos(inc) * math.cos(Om)
    z = yp * math.sin(inc)
    return x, y, z, E


def sat_clock(eph, t_tx, E):
    """卫星钟差(秒)：三参数 + 相对论改正 - Tgd。钟差时间变量用 t-Toe（星历参考时刻）"""
    dt = t_tx - eph['Toe']
    dtr = -2 * math.sqrt(MU) * eph['sqrtA'] * eph['e'] * math.sin(E) / C ** 2
    return eph['af0'] + eph['af1'] * dt + eph['af2'] * dt * dt + dtr - eph['Tgd']


# ==================== 4. 坐标转换与改正模型 ====================
def ecef_to_blh(x, y, z):
    """ECEF -> (纬度rad, 经度rad, 大地高m)，Bowring 闭式迭代"""
    lon = math.atan2(y, x)
    p = math.sqrt(x * x + y * y)
    theta = math.atan2(z * 1.0, p * (1 - F_WGS84))
    lat = math.atan2(z + E2 / (1 - F_WGS84) / (1 - E2) * RE_WGS84 * math.sin(theta) ** 3,
                     p - E2 * RE_WGS84 * math.cos(theta) ** 3)
    for _ in range(8):
        sl = math.sin(lat)
        N = RE_WGS84 / math.sqrt(1 - E2 * sl * sl)
        h = p / math.cos(lat) - N
        lat = math.atan2(z, p * (1 - E2 * N / (N + h)))
    N = RE_WGS84 / math.sqrt(1 - E2 * math.sin(lat) ** 2)
    h = p / math.cos(lat) - N
    return lat, lon, h


def los_azel(rx, sat):
    """站心视线 -> (高度角rad, 方位角rad, 单位视线向量ex,ey,ez)，视线由测站指向卫星"""
    lat, lon, _ = ecef_to_blh(rx[0], rx[1], rx[2])
    sl, cl, so, co = math.sin(lat), math.cos(lat), math.sin(lon), math.cos(lon)
    dx, dy, dz = sat[0] - rx[0], sat[1] - rx[1], sat[2] - rx[2]
    rho = math.sqrt(dx * dx + dy * dy + dz * dz)
    # ECEF -> 站心东北天(ENU)
    e = -so * dx + co * dy
    n = -sl * co * dx - sl * so * dy + cl * dz
    u = cl * co * dx + cl * so * dy + sl * dz
    el = math.asin(max(-1.0, min(1.0, u / rho)))
    az = math.atan2(e, n)
    if az < 0:
        az += 2 * PI
    return el, az, (dx / rho, dy / rho, dz / rho), rho


def klobuchar(lat, lon, el, az, t_gps, alpha, beta):
    """Klobuchar 单频(L1)倾斜路径电离层延迟(米)。
    lat/lon测站经纬(rad)，el/az高度/方位角(rad)，t_gps GPS周内秒。
    IS-GPS-200模型：内部角度一律用半周(semicircle=rad/pi)，且半周值直接喂cos/sin。"""
    phi_u, lam_u, els, azs = lat / PI, lon / PI, el / PI, az / PI
    # 1. 地心角(半周)
    psi = 0.0137 / (els + 0.11) - 0.022
    # 2. 穿刺点纬度(半周)，限幅±0.416
    phi_ipp = phi_u + psi * math.cos(azs)
    phi_ipp = max(-0.416, min(0.416, phi_ipp))
    # 3. 穿刺点经度(半周)；cos里cos(纬度弧度)=cos(phi_ipp*pi)
    lam_ipp = lam_u + psi * math.sin(azs) / math.cos(phi_ipp * PI)
    # 4. 穿刺点本地时间(秒)
    t_local = (43200.0 * lam_ipp + t_gps) % 86400.0
    # 5. 倾斜映射因子
    F = 1.0 + 16.0 * (0.53 - els) ** 3
    # 6. 振幅(秒)/周期(秒)，对穿刺点纬度展开
    amp = sum(alpha[k] * phi_ipp ** k for k in range(4))
    amp = max(amp, 0.0)
    per = sum(beta[k] * phi_ipp ** k for k in range(4))
    per = max(per, 72000.0)
    # 7. 相位(用本地时间，非t_gps)
    x = 2.0 * PI * (t_local - 50400.0) / per
    if abs(x) < 1.57:
        T = 5e-9 + amp * (1.0 - x * x / 2.0 + x ** 4 / 24.0)
    else:
        T = 5e-9
    return C * F * T


def saastamoinen(lat, h, el, P=1013.25, Tk=273.15 + 15.0, rh=0.5):
    """Saastamoinen 对流层斜路径延迟(米)，标准大气(15℃,1013.25hPa,相对湿度50%)"""
    # 标准大气：气压/温度随高度，水汽压
    Pst = P * (1 - 2.2557e-5 * h) ** 5.2568
    Tst = Tk - 6.5e-3 * h
    if Tst < 273.15 - 60:
        Tst = 273.15 - 60
    e_mb = rh * 6.108 * math.exp((17.15 * (Tst - 273.15)) / (234.7 + (Tst - 273.15)))
    z = PI / 2 - el
    tz = 1.0 / max(0.05, math.cos(z))
    trop = 0.002277 * tz * (Pst + (1255.0 / Tst + 0.05) * e_mb)
    return trop


# ==================== 5. 最小二乘 ====================
def solve_linear(A, b):
    """选主元高斯消元解 Ax=b（4阶，不依赖numpy）；奇异抛异常由上层兜底"""
    n = len(A)
    M = [A[i][:] + [b[i]] for i in range(n)]
    for col in range(n):
        p = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[p][col]) < 1e-12:
            raise ValueError('法方程奇异(卫星几何过差)')
        M[col], M[p] = M[p], M[col]
        for r in range(col + 1, n):
            f = M[r][col] / M[col][col]
            for k in range(col, n + 1):
                M[r][k] -= f * M[col][k]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))) / M[i][i]
    return x


def geom(prn, tow, rx):
    """迭代求发射时刻、地球自转改正后卫星位置、几何距离、E、星历"""
    sx = sy = sz = E = 0.0
    eph = None
    tau = 0.075
    for _ in range(3):
        t_tx = tow - tau
        eph = find_eph(prn, t_tx)
        if eph is None:
            return None
        sx, sy, sz, E = sat_pos(eph, t_tx)
        ang = OMEGA_E * tau                       # 信号飞行期间地球自转改正
        sx, sy = sx * math.cos(ang) - sy * math.sin(ang), \
                 sx * math.sin(ang) + sy * math.cos(ang)
        tau = math.sqrt((sx - rx[0]) ** 2 + (sy - rx[1]) ** 2 + (sz - rx[2]) ** 2) / C
    return eph, t_tx, sx, sy, sz, E, tau * C


# ==================== 6. 单历元解算 ====================
def solve_epoch(ep, el_cut_deg=10.0, use_iono=True, use_tropo=True, obs_key='C1'):
    """
    解单个历元。el_cut_deg: 高度角截止角(度)。
    返回 dict：pos/cdt/niter/rms/used/res/pdop/el_lim 等；失败返回 None。
    """
    _, tow = to_gps_time(ep['time'])
    pos = [x0, y0, z0]
    cdt = 0.0
    el_cut = el_cut_deg * PI / 180.0
    it = 0

    for it in range(10):
        G, l = [], []
        for sat, obs in ep['obs'].items():
            if not sat.startswith('G'):
                continue
            P = obs.get(obs_key, float('nan'))
            if math.isnan(P):
                continue
            prn = int(sat[1:])
            gm = geom(prn, tow, pos)
            if gm is None:
                continue
            eph, t_tx, sx, sy, sz, E, rho = gm

            # 过期星历剔除
            dte = t_tx - eph['Toe']
            if dte > 302400:  dte -= 604800
            if dte < -302400: dte += 604800
            if abs(dte) > 7200:  # 2h，放宽以覆盖日界尾段外推(有次日星历可改回4000)
                continue

            # 高度角/方位角
            el, az, (ex, ey, ez), _ = los_azel(pos, (sx, sy, sz))
            if el < el_cut:
                continue

            # 卫星钟差
            dts = sat_clock(eph, t_tx, E)
            P_corr = P + C * dts

            # 电离层 + 对流层（信号延迟，使测距变大，从观测伪距中扣除）
            lat0, lon0, h0 = ecef_to_blh(pos[0], pos[1], pos[2])
            iono = klobuchar(lat0, lon0, el, az, tow, ION_ALPHA, ION_BETA) if use_iono else 0.0
            tropo = saastamoinen(lat0, h0, el) if use_tropo else 0.0
            P_corr -= (iono + tropo)

            # 线性化观测方程： -e·dpos + c·dtR = P_corr - rho - cdt
            G.append([-ex, -ey, -ez, 1.0])
            l.append(P_corr - rho - cdt)

        if len(G) < 4:
            return None
        n = 4
        A = [[sum(G[r][k] * G[r][j] for r in range(len(G))) for j in range(n)] for k in range(n)]
        y = [sum(G[r][k] * l[r] for r in range(len(G))) for k in range(n)]
        d = solve_linear(A, y)

        pos[0] += d[0]; pos[1] += d[1]; pos[2] += d[2]
        cdt += d[3]
        if math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2) < 1e-4:
            break

    # 收敛后重算精确残差与PDOP
    res, used_sats, els = [], [], []
    GQ = []
    for sat, obs in ep['obs'].items():
        if not sat.startswith('G'):
            continue
        P = obs.get(obs_key, float('nan'))
        if math.isnan(P):
            continue
        prn = int(sat[1:])
        gm = geom(prn, tow, pos)
        if gm is None:
            continue
        eph, t_tx, sx, sy, sz, E, rho = gm
        dte = t_tx - eph['Toe']
        if dte > 302400:  dte -= 604800
        if dte < -302400: dte += 604800
        if abs(dte) > 7200:  # 2h，放宽以覆盖日界尾段外推(有次日星历可改回4000)
            continue
        el, az, (ex, ey, ez), _ = los_azel(pos, (sx, sy, sz))
        if el < el_cut:
            continue
        lat0, lon0, h0 = ecef_to_blh(pos[0], pos[1], pos[2])
        iono = klobuchar(lat0, lon0, el, az, tow, ION_ALPHA, ION_BETA) if use_iono else 0.0
        tropo = saastamoinen(lat0, h0, el) if use_tropo else 0.0
        dts = sat_clock(eph, t_tx, E)
        res.append((P + C * dts) - rho - cdt - (iono + tropo))
        used_sats.append(sat)
        els.append(math.degrees(el))
        GQ.append([-ex, -ey, -ez, 1.0])

    if len(res) < 4:
        return None
    # PDOP = sqrt(qxx+qyy+qzz), Q=(G'G)^-1
    n = 4
    Nm = [[sum(GQ[r][k] * GQ[r][j] for r in range(len(GQ))) for j in range(n)] for k in range(n)]
    try:
        Q = mat_inverse(Nm)
        pdop = math.sqrt(max(0.0, Q[0][0] + Q[1][1] + Q[2][2]))
    except Exception:
        pdop = float('nan')

    rms = math.sqrt(sum(v * v for v in res) / len(res))
    return {'time': ep['time'], 'pos': pos, 'cdt': cdt, 'niter': it + 1,
            'rms': rms, 'used': used_sats, 'res': res, 'pdop': pdop,
            'el_min': min(els), 'nsat': len(used_sats), 'tow': tow}


def mat_inverse(M):
    """求4阶对称正定矩阵的逆（高斯-约当）"""
    n = len(M)
    A = [M[i][:] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        p = max(range(col, n), key=lambda r: abs(A[r][col]))
        A[col], A[p] = A[p], A[col]
        piv = A[col][col]
        for k in range(2 * n):
            A[col][k] /= piv
        for r in range(n):
            if r == col:
                continue
            f = A[r][col]
            for k in range(2 * n):
                A[r][k] -= f * A[col][k]
    return [[A[i][n + j] for j in range(n)] for i in range(n)]


if __name__ == '__main__':
    ep = epochs[0]
    r = solve_epoch(ep)
    pos = r['pos']
    dx = math.sqrt((pos[0] - x0) ** 2 + (pos[1] - y0) ** 2 + (pos[2] - z0) ** 2)
    lat, lon, h = ecef_to_blh(*pos)
    print('首历元:', ep['time'])
    print('迭代次数:', r['niter'], ' 参与卫星数:', r['nsat'], r['used'])
    print('解算坐标: %.4f %.4f %.4f' % (pos[0], pos[1], pos[2]))
    print('大地坐标: %.8f° %.8f° 高 %.2f m' % (math.degrees(lat), math.degrees(lon), h))
    print('与概略坐标差: %.2f m' % dx)
    print('接收机钟差: %.1f m (%.2f 微秒)' % (r['cdt'], r['cdt'] / C * 1e6))
    print('PDOP: %.2f  最低高度角: %.1f°' % (r['pdop'], r['el_min']))
    print('残差RMS: %.2f m' % r['rms'])
    print('各星残差(m):', [round(v, 1) for v in r['res']])
