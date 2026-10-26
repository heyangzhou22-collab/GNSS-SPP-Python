# -*- coding: utf-8 -*-
import math
import os
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(_HERE, '..', 'data', 'bjfs2740.26o')

with open(path, encoding='ascii') as f:
    lines = f.readlines()

# ---------- 1. 解析文件头 ----------
i = 0
obs_types = []
approx_xyz = None
while True:
    line = lines[i]; i += 1
    label = line[60:].strip()
    if label == '# / TYPES OF OBSERV':
        obs_types = line[6:60].split()
    elif label == 'APPROX POSITION XYZ':
        approx_xyz = tuple(map(float, line.split()[:3]))
    if label == 'END OF HEADER':
        break

n_type = len(obs_types)
rows_per_sat = math.ceil(n_type / 5)

def parse_obs_line(line):
    vals = []
    for k in range(5):
        v = line[k*16:k*16+14].strip()
        vals.append(float(v) if v else float('nan'))
    return vals

# ---------- 2. 循环解析全部历元 ----------
epochs = []

while i < len(lines):
    line = lines[i]
    # 跳过空行
    if not line.strip():
        i += 1
        continue
    # 历元行第29-31列是卫星数，能转成数字才算历元行
    if not line[29:32].strip().isdigit():
        i += 1
        continue

    i += 1
    t = datetime(2000 + int(line[1:3]), int(line[4:6]), int(line[7:9]),
                 int(line[10:12]), int(line[13:15]), int(float(line[15:26])))
    flag = int(line[28])
    nsat = int(line[29:32])

    # 卫星号（首行+可能的折行，都从第33列开始）
    sats = [line[32+k:35+k] for k in range(0, min(nsat, 12)*3, 3)]
    while len(sats) < nsat:
        c = lines[i][32:68]; i += 1
        sats += [c[k:k+3] for k in range(0, len(c), 3) if c[k:k+3].strip()]

    # 读每颗星的观测行
    obs = {}
    for sat in sats:
        vals = []
        for _ in range(rows_per_sat):
            vals += parse_obs_line(lines[i]); i += 1
        obs[sat] = dict(zip(obs_types, vals[:n_type]))

    # 只收正常历元（事件历元 flag!=0 不进成果）
    if flag == 0:
        epochs.append({'time': t, 'obs': obs})

# ---------- 3. 验收输出 ----------
print(f'历元总数: {len(epochs)}')
print(f'首个历元: {epochs[0]["time"]}')
print(f'末个历元: {epochs[-1]["time"]}')

# 抽查首历元 G05
g05 = epochs[0]['obs']['G05']
print(f'G05 C1={g05["C1"]:.3f}  P2={g05["P2"]:.3f}')