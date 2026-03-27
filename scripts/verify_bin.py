"""
verify_bin_v2.py
================
Correctly handles XKF1 (EKF3, newer ArduPilot) AND NKF1 (EKF2, older).
Run this on the log produced by fly_doublets_v3.py.

Usage:
  python verify_bin_v2.py <log.BIN>
"""

import sys
import os
import math
import numpy as np
import pandas as pd
from pymavlink import mavutil
from scipy.spatial.transform import Rotation

MIN_AIRSPEED_MS = 0.5
FORWARD_FLIGHT_MS = 8.0


def check_message_types(filepath):
    print("\n" + "="*60)
    print("PASS 1: MESSAGE TYPE INVENTORY")
    print("="*60)

    mlog = mavutil.mavlink_connection(filepath)
    counts = {}
    first_ts = {}
    last_ts  = {}

    while True:
        msg = mlog.recv_match()
        if msg is None:
            break
        t = msg.get_type()
        if t == 'BAD_DATA':
            continue
        ts = getattr(msg, 'TimeUS', None) or getattr(msg, '_timestamp', 0) * 1e6
        counts[t] = counts.get(t, 0) + 1
        if t not in first_ts:
            first_ts[t] = ts
        last_ts[t] = ts

    duration = None
    for ref in ['IMU', 'ATT', 'XKF1', 'NKF1']:
        if ref in first_ts:
            duration = (last_ts[ref] - first_ts[ref]) / 1e6
            break

    print(f"\n{'Type':<20} {'Count':>8} {'Hz':>8}")
    print("-" * 40)
    for t, c in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        dur = (last_ts[t] - first_ts[t]) / 1e6 if t in first_ts else 0
        hz  = c / dur if dur > 0 else 0
        print(f"{t:<20} {c:>8} {hz:>8.1f}")

    print(f"\nLog duration: {duration:.1f}s" if duration else "\nCould not determine duration")

    # Detect velocity source
    vel_msg = None
    if 'XKF1' in counts:
        vel_msg = 'XKF1'
        print("\n  Velocity source : XKF1 (EKF3)  ✓")
    elif 'NKF1' in counts:
        vel_msg = 'NKF1'
        print("\n  Velocity source : NKF1 (EKF2)  ✓")
    else:
        print("\n  WARNING: no velocity message (XKF1/NKF1) found")

    print("\n  Critical for SysID:")
    for m in ['IMU', 'ATT', 'RCOU']:
        print(f"    {m:<8}: {'FOUND ✓' if m in counts else 'MISSING ✗'}")

    print("\n  Useful for Phase 2:")
    for m in ['XKF1', 'NKF1', 'AOA', 'ARSP']:
        print(f"    {m:<8}: {'FOUND ✓' if m in counts else 'not present'}")

    return counts, vel_msg


def extract(filepath, counts, vel_msg):
    print("\n" + "="*60)
    print("PASS 2: EXTRACTION")
    print("="*60)

    mlog = mavutil.mavlink_connection(filepath)
    imu = []; att = []; rcou = []; vel = []; aoa = []; arsp = []

    want = ['IMU', 'ATT', 'RCOU']
    if vel_msg:            want.append(vel_msg)
    if 'AOA'  in counts:   want.append('AOA')
    if 'ARSP' in counts:   want.append('ARSP')

    while True:
        msg = mlog.recv_match(type=want)
        if msg is None:
            break
        t = msg.get_type()
        d = msg.to_dict()

        if t == 'IMU':
            imu.append({'TimeUS': d['TimeUS'],
                        'GyrX': d.get('GyrX', np.nan), 'GyrY': d.get('GyrY', np.nan),
                        'GyrZ': d.get('GyrZ', np.nan), 'AccX': d.get('AccX', np.nan),
                        'AccY': d.get('AccY', np.nan), 'AccZ': d.get('AccZ', np.nan)})
        elif t == 'ATT':
            att.append({'TimeUS': d['TimeUS'],
                        'Roll':  np.radians(d.get('Roll',  0)),
                        'Pitch': np.radians(d.get('Pitch', 0)),
                        'Yaw':   np.radians(d.get('Yaw',   0))})
        elif t == 'RCOU':
            rcou.append({'TimeUS': d['TimeUS'],
                         'C1': d.get('C1', 0), 'C2': d.get('C2', 0),
                         'C3': d.get('C3', 0), 'C4': d.get('C4', 0)})
        elif t == vel_msg:
            if d.get('C', 0) == 0:          # XKF1 has core index 'C'; take primary
                vel.append({'TimeUS': d['TimeUS'],
                            'VN': d.get('VN', 0.0),
                            'VE': d.get('VE', 0.0),
                            'VD': d.get('VD', 0.0)})
        elif t == 'AOA':
            aoa.append({'TimeUS': d['TimeUS'], 'AOA': d.get('AOA', np.nan)})
        elif t == 'ARSP':
            arsp.append({'TimeUS': d['TimeUS'], 'airspeed': d.get('Airspeed', 0.0)})

    print(f"  IMU: {len(imu)}, ATT: {len(att)}, RCOU: {len(rcou)}, "
          f"{vel_msg or 'VEL'}: {len(vel)}, AOA: {len(aoa)}, ARSP: {len(arsp)}")

    return (pd.DataFrame(imu), pd.DataFrame(att), pd.DataFrame(rcou),
            pd.DataFrame(vel)  if vel  else None,
            pd.DataFrame(aoa)  if aoa  else None,
            pd.DataFrame(arsp) if arsp else None)


def merge(df_imu, df_att, df_rcou, df_vel, df_aoa, df_arsp):
    print("\n  Merging on IMU time axis...")
    df = df_imu.sort_values('TimeUS').drop_duplicates('TimeUS')
    df = pd.merge_asof(df, df_att.sort_values('TimeUS'),  on='TimeUS', direction='backward')
    df = pd.merge_asof(df, df_rcou.sort_values('TimeUS'), on='TimeUS', direction='backward')

    if df_vel  is not None and len(df_vel):
        df = pd.merge_asof(df, df_vel.sort_values('TimeUS'),  on='TimeUS', direction='backward')
    if df_aoa  is not None and len(df_aoa):
        df = pd.merge_asof(df, df_aoa.sort_values('TimeUS'),  on='TimeUS', direction='backward')
    if df_arsp is not None and len(df_arsp):
        df = pd.merge_asof(df, df_arsp.sort_values('TimeUS'), on='TimeUS', direction='backward')

    df = df.dropna(subset=['GyrX', 'Roll', 'C1'])
    df['TimeSec'] = (df['TimeUS'] - df['TimeUS'].iloc[0]) / 1e6

    for ax in ['X', 'Y', 'Z']:
        col = f'Gyr{ax}'
        dot = f'dot_Gyr{ax}'
        df[dot] = np.gradient(df[col].values, df['TimeSec'].values)
        p1, p99 = df[dot].quantile(0.01), df[dot].quantile(0.99)
        df[dot] = df[dot].clip(p1, p99)

    print(f"  Result: {len(df)} rows, columns: {list(df.columns)}")
    return df


def compute_alpha(df):
    if 'VN' not in df.columns:
        return df
    alphas = []
    for _, row in df.iterrows():
        V_ned = np.array([row['VN'], row['VE'], row['VD']])
        R = Rotation.from_euler('zyx', [row['Yaw'], row['Pitch'], row['Roll']])
        u, v, w = R.apply(V_ned)
        speed = float(np.linalg.norm([u, v, w]))
        alphas.append(np.arctan2(w, u) if speed >= MIN_AIRSPEED_MS else np.nan)
    df['alpha_rad'] = alphas
    df['alpha_deg'] = np.degrees(df['alpha_rad'])
    valid = df['alpha_rad'].notna().sum()
    print(f"\n  Alpha from XKF1/NKF1: {valid}/{len(df)} valid samples")
    return df


def analyse_excitation(df):
    print("\n" + "="*60)
    print("PASS 3: ROTATIONAL EXCITATION QUALITY")
    print("="*60)

    armed = df[(df['C1'] > 1100) | (df['C2'] > 1100) |
               (df['C3'] > 1100) | (df['C4'] > 1100)].copy()
    print(f"  Armed samples: {len(armed)} / {len(df)}")

    results = {}
    axis_map = [('X (roll)',  'GyrX', 'dot_GyrX', 'Ixx'),
                ('Y (pitch)', 'GyrY', 'dot_GyrY', 'Iyy'),
                ('Z (yaw)',   'GyrZ', 'dot_GyrZ', 'Izz')]

    for axis_label, gyr_col, dot_col, inertia in axis_map:
        armed[f'var_{gyr_col}'] = (armed[gyr_col]
                                   .rolling(window=20, min_periods=5).var())
        excited = armed[armed[f'var_{gyr_col}'] > 1e-5]
        pct = 100 * len(excited) / max(len(armed), 1)

        snr = 0.0
        if dot_col in armed.columns:
            snr = armed[dot_col].abs().mean() / (armed[dot_col].std() + 1e-10)

        q = "GOOD" if pct >= 20 else ("MODERATE" if pct >= 5 else "POOR")
        print(f"\n  {axis_label} -> {inertia}:")
        print(f"    Excited : {len(excited)} samples ({pct:.1f}%) — {q}")
        print(f"    SNR     : {snr:.4f}")
        if q == "POOR":
            print(f"    >> Need sharper maneuvers on this axis.")
        results[axis_label] = q

    return results


def verdict(df, counts, vel_msg, excitation):
    print("\n" + "="*60)
    print("FINAL VERDICT")
    print("="*60)

    p1 = all(v != "POOR" for v in excitation.values())
    p2 = ('alpha_rad' in df.columns and df['alpha_rad'].notna().sum() > 50)

    print(f"\n  Phase 1 (mass + inertia)       : {'FEASIBLE ✓' if p1 else 'NOT FEASIBLE ✗'}")
    print(f"  Phase 2 (aero derivatives)     : {'FEASIBLE ✓' if p2 else 'NOT FEASIBLE ✗'}")

    if not p1:
        print("\n  Fix Phase 1: run fly_doublets_v3.py and do sharper maneuvers (±20 deg, 5 reps)")

    if not p2:
        print("\n  Fix Phase 2:")
        if vel_msg is None:
            print("    - XKF1 absent: already present in SITL as XKF1 - check verify_bin_v2.py output")
        print("    - Need forward flight >8 m/s with pitch doublets")

    print("\n  Message coverage:")
    for m, req in [('IMU','required'), ('ATT','required'), ('RCOU','required'),
                   ('XKF1','Phase 2'), ('NKF1','Phase 2 alt'), ('ARSP','optional')]:
        present = m in counts
        print(f"    {m:<8} ({req:<12}): {'PRESENT' if present else 'absent'}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_bin_v2.py <log.BIN>")
        sys.exit(1)

    fp = sys.argv[1]
    if not os.path.exists(fp):
        print(f"Not found: {fp}")
        sys.exit(1)

    print(f"\nFile: {fp}  ({os.path.getsize(fp)/1024:.1f} KB)")
    counts, vel_msg = check_message_types(fp)
    df_imu, df_att, df_rcou, df_vel, df_aoa, df_arsp = extract(fp, counts, vel_msg)
    df = merge(df_imu, df_att, df_rcou, df_vel, df_aoa, df_arsp)
    df = compute_alpha(df)
    exc = analyse_excitation(df)
    verdict(df, counts, vel_msg, exc)

    out = fp.replace('.BIN', '_v2.csv').replace('.bin', '_v2.csv')
    df.to_csv(out, index=False)
    print(f"\n  CSV saved: {out}")