# -*- coding: utf-8 -*-
"""
GNSS 第六周：EKF卡尔曼滤波定位 + 高度角随机模型加权
=================================================
在W3单频最小二乘基础上，用扩展卡尔曼滤波递推估计位置，
观测噪声协方差随卫星高度角变化：σ² = σ0² / sin²(elev)

输出：
- results/ekf_positions.csv    逐历元EKF滤波结果
- results/ekf_vs_ls.png        最小二乘 vs EKF 精度对比图
"""
import os
import math
import csv
import numpy as np
import matplotlib.pyplot as plt

from rinex_parser import epochs
from ephemeris import (C, PI, x0, y0, z0,
                       to_gps_time, geom, los_azel,
                       klobuchar, saastamoinen, ecef_to_blh,
                       ION_ALPHA, ION_BETA, solve_epoch)

# ---------------- EKF 参数 ----------------
SIGMA0 = 0.3           # 基础伪距噪声标准差 (m)
EL_CUT_DEG = 10.0      # 高度角截止 (度)
Q_POS = 0.1            # 位置过程噪声 (m²/历元)
Q_CLK = 10.0           # 钟差过程噪声 (m²/历元)
P0_POS = 100.0         # 初始位置协方差 (m²)
P0_CLK = 100.0         # 初始钟差协方差 (m²)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')


def build_obs(ep, pos, cdt):
    """构造观测方程：返回 G(设计矩阵), l(观测残差), els(高度角列表)"""
    _, tow = to_gps_time(ep['time'])
    G, l, els = [], [], []
    el_cut = EL_CUT_DEG * PI / 180.0

    for sat, obs in ep['obs'].items():
        if not sat.startswith('G'):
            continue
        P = obs.get('C1', float('nan'))
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
        if abs(dte) > 7200:
            continue

        # 高度角
        el, az, (ex, ey, ez), _ = los_azel(pos, (sx, sy, sz))
        if el < el_cut:
            continue

        # 各项改正
        dt = t_tx - eph['Toe']
        dtr = -2 * math.sqrt(3.986005e14) * eph['sqrtA'] * eph['e'] * math.sin(E) / C ** 2
        dts = eph['af0'] + eph['af1'] * dt + eph['af2'] * dt * dt + dtr - eph['Tgd']

        lat0, lon0, h0 = ecef_to_blh(pos[0], pos[1], pos[2])
        iono = klobuchar(lat0, lon0, el, az, tow, ION_ALPHA, ION_BETA)
        tropo = saastamoinen(lat0, h0, el)

        P_corr = P + C * dts - iono - tropo

        G.append([-ex, -ey, -ez, 1.0])
        l.append(P_corr - rho - cdt)
        els.append(el)

    return np.array(G), np.array(l), np.array(els)


def ekf_run(epochs_list):
    """EKF主循环，状态向量 X = [x, y, z, cdt]"""
    n = 4  # 状态维度

    # ===== 初始化：用第一个历元最小二乘求初值 =====
    r0 = solve_epoch(epochs_list[0], el_cut_deg=EL_CUT_DEG)
    if r0 is None:
        raise RuntimeError('首历元最小二乘失败，无法初始化EKF')

    X = np.array(r0['pos'] + [r0['cdt']])
    P = np.diag([P0_POS, P0_POS, P0_POS, P0_CLK])
    Q = np.diag([Q_POS, Q_POS, Q_POS, Q_CLK])

    print(f'EKF初始化完成')
    print(f'  初值坐标: {X[0]:.3f}, {X[1]:.3f}, {X[2]:.3f}')
    print(f'  初始钟差: {X[3]:.3f} m')

    results = []

    for idx, ep in enumerate(epochs_list):
        # ===== 预测步（静态模型：F=I） =====
        X_pred = X.copy()
        P_pred = P + Q

        # ===== 构造观测 =====
        G, l, els = build_obs(ep, X_pred[:3], X_pred[3])
        nsat = len(G)

        if nsat < 4:
            X = X_pred
            P = P_pred
            results.append({'time': ep['time'], 'pos': X[:3].copy(), 'cdt': X[3],
                            'rms_w': np.nan, 'rms_eq': np.nan, 'nsat': nsat})
            continue

        # ===== 高度角加权 R 矩阵 =====
        sigma_i = SIGMA0 / np.sin(els)
        R_w = np.diag(sigma_i ** 2)
        R_eq = np.diag(np.full(nsat, SIGMA0 ** 2))

        # ===== 卡尔曼增益（加权） =====
        S = G @ P_pred @ G.T + R_w
        K = P_pred @ G.T @ np.linalg.inv(S)

        # ===== 状态更新 =====
        dx = K @ l
        X = X_pred + dx

        # ===== 协方差更新（约瑟夫形式，更稳定） =====
        I_KG = np.eye(n) - K @ G
        P = I_KG @ P_pred @ I_KG.T + K @ R_w @ K.T

        # ===== 计算残差RMS（加权 vs 等权） =====
        # 残差 = 新息 - 设计矩阵×状态增量
        v_w = l - G @ dx
        rms_w = np.sqrt(np.mean(v_w ** 2))

        # 等权滤波残差（对比用）
        S_eq = G @ P_pred @ G.T + R_eq
        K_eq = P_pred @ G.T @ np.linalg.inv(S_eq)
        dx_eq = K_eq @ l
        v_eq = l - G @ dx_eq
        rms_eq = np.sqrt(np.mean(v_eq ** 2))

        results.append({'time': ep['time'], 'pos': X[:3].copy(), 'cdt': X[3],
                        'rms_w': rms_w, 'rms_eq': rms_eq, 'nsat': nsat})

        if idx % 50 == 0:
            err = np.sqrt(np.sum((X[:3] - np.array([x0, y0, z0])) ** 2))
            print(f'历元 {idx:4d}/{len(epochs_list)} | 卫星: {nsat} | '
                  f'位置误差: {err:7.2f}m | 加权RMS: {rms_w:.3f}m | 等权RMS: {rms_eq:.3f}m')

    return results


def save_csv(results, path):
    """保存逐历元结果CSV"""
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['time', 'x_m', 'y_m', 'z_m', 'cdt_m', 'nsat', 'rms_weighted_m', 'rms_equal_m'])
        for r in results:
            w.writerow([r['time'].strftime('%Y-%m-%d %H:%M:%S'),
                        f'{r["pos"][0]:.4f}', f'{r["pos"][1]:.4f}', f'{r["pos"][2]:.4f}',
                        f'{r["cdt"]:.4f}', r['nsat'],
                        f'{r["rms_w"]:.4f}', f'{r["rms_eq"]:.4f}'])
    print(f'已保存CSV: {path}')


def plot_results(results, path):
    """绘制精度对比图"""
    errs = []
    rms_w_list = []
    rms_eq_list = []

    for r in results:
        e = np.sqrt(np.sum((r['pos'] - np.array([x0, y0, z0])) ** 2))
        errs.append(e)
        rms_w_list.append(r['rms_w'])
        rms_eq_list.append(r['rms_eq'])

    errs = np.array(errs)
    rms_w_list = np.array(rms_w_list)
    rms_eq_list = np.array(rms_eq_list)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # 图1: 3D位置误差时序
    axes[0].plot(errs, 'b-', linewidth=0.8)
    axes[0].set_ylabel('3D Position Error [m]')
    axes[0].set_title('EKF Filter Position Error (Height-Angle Weighted)')
    axes[0].grid(True, alpha=0.3)

    # 图2: 加权 vs 等权 残差对比
    axes[1].plot(rms_w_list, 'b-', label='Height-Angle Weighted (σ²=σ₀²/sin²e)', linewidth=1.2)
    axes[1].plot(rms_eq_list, 'r--', label='Equal Weight', linewidth=1.2, alpha=0.7)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('RMS Residual [m]')
    axes[1].set_title('Residual RMS: Height-Angle Weighted vs Equal Weight')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'已保存图片: {path}')

    # 统计
    valid = ~np.isnan(errs)
    valid_errs = errs[valid]
    print(f'\n===== EKF 滤波精度统计 =====')
    print(f'有效历元数: {len(valid_errs)}')
    print(f'平均3D误差: {np.mean(valid_errs):.3f} m')
    print(f'最小3D误差: {np.min(valid_errs):.3f} m')
    print(f'最大3D误差: {np.max(valid_errs):.3f} m')


if __name__ == '__main__':
    print('=' * 60)
    print('W6: EKF卡尔曼滤波定位 + 高度角随机模型加权')
    print('=' * 60)
    print(f'历元总数: {len(epochs)}')

    # 运行EKF
    results = ekf_run(epochs)

    # 输出
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_csv(results, os.path.join(RESULTS_DIR, 'ekf_positions.csv'))
    plot_results(results, os.path.join(RESULTS_DIR, 'ekf_vs_ls.png'))

    print('\n' + '=' * 60)
    print('EKF滤波完成！')
    print('=' * 60)
