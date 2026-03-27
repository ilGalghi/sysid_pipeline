"""
extractor.py
============
Parse a .BIN DataFlash log and produce a time-aligned CSV
with IMU, ATT, and RCOU on the IMU time axis.

Usage:
  python extractor.py [logfile.BIN]
  Defaults to 00000002.BIN if no argument given.
"""

import os
import sys
import numpy as np
import pandas as pd
from pymavlink import mavutil

DEFAULT_BIN = '00000002.BIN'


def extract_log_data(filepath):
    print(f"Opening {filepath} ...")
    mlog = mavutil.mavlink_connection(filepath)

    imu, att, rcou = [], [], []

    print("Extracting IMU / ATT / RCOU messages ...")
    while True:
        msg = mlog.recv_match(type=['IMU', 'ATT', 'RCOU'])
        if msg is None:
            break

        d = msg.to_dict()
        t = msg.get_type()

        if t == 'IMU':
            imu.append({
                'TimeUS': d['TimeUS'],
                'GyrX': d.get('GyrX', np.nan),
                'GyrY': d.get('GyrY', np.nan),
                'GyrZ': d.get('GyrZ', np.nan),
                'AccX': d.get('AccX', np.nan),
                'AccY': d.get('AccY', np.nan),
                'AccZ': d.get('AccZ', np.nan),
            })
        elif t == 'ATT':
            att.append({
                'TimeUS': d['TimeUS'],
                'Roll':  d.get('Roll',  0),
                'Pitch': d.get('Pitch', 0),
                'Yaw':   d.get('Yaw',   0),
            })
        elif t == 'RCOU':
            rcou.append({
                'TimeUS': d['TimeUS'],
                'C1': d.get('C1', 0),
                'C2': d.get('C2', 0),
                'C3': d.get('C3', 0),
                'C4': d.get('C4', 0),
            })

    print(f"  IMU: {len(imu)}, ATT: {len(att)}, RCOU: {len(rcou)}")
    if not rcou:
        raise ValueError("No RCOU messages found — check LOG_BITMASK")
    return imu, att, rcou


def synchronize(imu, att, rcou):
    """Merge ATT and RCOU onto the IMU time axis via merge_asof."""
    print("Time-aligning to IMU rate ...")

    df_imu  = pd.DataFrame(imu).sort_values('TimeUS').drop_duplicates('TimeUS')
    df_att  = pd.DataFrame(att).sort_values('TimeUS')
    df_rcou = pd.DataFrame(rcou).sort_values('TimeUS')

    df = pd.merge_asof(df_imu, df_att,  on='TimeUS', direction='backward')
    df = pd.merge_asof(df,     df_rcou, on='TimeUS', direction='backward')
    df = df.dropna()

    df['TimeSec'] = (df['TimeUS'] - df['TimeUS'].iloc[0]) / 1e6
    print(f"  Result: {len(df)} aligned rows")
    return df


if __name__ == "__main__":
    binfile = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BIN
    if not os.path.exists(binfile):
        print(f"Error: {binfile} not found"); sys.exit(1)

    imu_list, att_list, rcou_list = extract_log_data(binfile)
    df = synchronize(imu_list, att_list, rcou_list)

    out = 'synchronized_flight_data.csv'
    df.to_csv(out, index=False)
    print(f"Saved {out}  ({len(df)} rows)")
