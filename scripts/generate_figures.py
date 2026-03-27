"""
generate_figures.py
===================
Produce two publication-quality figures for the GSoC proposal.

  1. flight_data_overview.png   — Gyro rates + RCOU (zoomed to doublets)
  2. excitation_segments.png    — Rolling variance with excited segments

Reads:  00000002_verified.csv
Usage:  python generate_figures.py [input.csv]
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEFAULT_CSV = 'data/00000002_verified.csv'


def load(path):
    df = pd.read_csv(path)
    armed = (df['C1'] > 1100) | (df['C2'] > 1100) | \
            (df['C3'] > 1100) | (df['C4'] > 1100)
    df = df[armed].copy().sort_values('TimeSec').reset_index(drop=True)
    return df


def find_active_region(df, margin_before=5.0, margin_after=15.0):
    """Find time window containing the doublet excitation."""
    gyr_total = np.abs(df['GyrX']) + np.abs(df['GyrY']) + np.abs(df['GyrZ'])
    var = gyr_total.rolling(200, center=True, min_periods=20).var()
    baseline = np.nanmedian(var)
    excited = var > baseline * 5
    if not excited.any():
        return df['TimeSec'].min(), df['TimeSec'].max()
    t_exc = df['TimeSec'][excited]
    return max(t_exc.min() - margin_before, df['TimeSec'].min()), \
           min(t_exc.max() + margin_after, df['TimeSec'].max())


def figure_overview(df, outfile='figures/flight_data_overview.png'):
    """Two-panel figure: gyro rates + motor PWMs, zoomed to doublets."""

    t_start, t_end = find_active_region(df)
    crop = (df['TimeSec'] >= t_start) & (df['TimeSec'] <= t_end)
    dc = df[crop]

    plt.rcParams.update({
        'font.size': 10, 'axes.titlesize': 11,
        'axes.labelsize': 10, 'legend.fontsize': 8,
        'figure.dpi': 180,
    })

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 2]})
    fig.suptitle('SITL Doublet Flight — Time-Aligned Sensor Data',
                 fontsize=13, fontweight='bold', y=0.98)

    t = dc['TimeSec'].values

    # gyro rates
    ax1.plot(t, np.degrees(dc['GyrX']), lw=0.8, alpha=0.85,
             color='#2563eb', label='p  (roll rate)')
    ax1.plot(t, np.degrees(dc['GyrY']), lw=0.8, alpha=0.85,
             color='#dc2626', label='q  (pitch rate)')
    ax1.plot(t, np.degrees(dc['GyrZ']), lw=0.8, alpha=0.85,
             color='#16a34a', label='r  (yaw rate)')
    ax1.set_ylabel('Angular rate [deg/s]')
    ax1.legend(loc='upper right', framealpha=0.85, ncol=3)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Gyroscope (IMU)', fontsize=9, loc='left')

    # motor commands
    for ch, col, lbl in [('C1', '#0369a1', 'Motor 1  (FR)'),
                         ('C2', '#b91c1c', 'Motor 2  (RL)'),
                         ('C3', '#15803d', 'Motor 3  (FL)'),
                         ('C4', '#a16207', 'Motor 4  (RR)')]:
        ax2.plot(t, dc[ch], lw=0.7, alpha=0.8, color=col, label=lbl)
    ax2.set_ylabel('PWM [μs]')
    ax2.set_xlabel('Time [s]')
    ax2.legend(loc='upper right', framealpha=0.85, ncol=4, fontsize=7)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Motor outputs (RCOU)', fontsize=9, loc='left')

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outfile, bbox_inches='tight')
    print(f"Saved {outfile}")
    plt.close()


def figure_excitation(df, outfile='figures/excitation_segments.png'):
    """Per-axis rolling variance with excited windows highlighted."""

    t_start, t_end = find_active_region(df)
    crop = (df['TimeSec'] >= t_start) & (df['TimeSec'] <= t_end)
    dc = df[crop].reset_index(drop=True)

    plt.rcParams.update({
        'font.size': 10, 'axes.titlesize': 11,
        'axes.labelsize': 10, 'legend.fontsize': 8,
        'figure.dpi': 180,
    })

    fig, axs = plt.subplots(3, 1, figsize=(9, 6.5), sharex=True)
    fig.suptitle('Variance-Based Excitation Segment Isolation',
                 fontsize=13, fontweight='bold', y=0.98)

    t = dc['TimeSec'].values
    window = 200

    info = [
        ('GyrX', '#2563eb', 'Roll (p)',  'Ixx'),
        ('GyrY', '#dc2626', 'Pitch (q)', 'Iyy'),
        ('GyrZ', '#16a34a', 'Yaw (r)',   'Izz'),
    ]

    for i, (col, color, label, inertia) in enumerate(info):
        ax = axs[i]
        gyr = dc[col].values
        gyr_deg = np.degrees(gyr)
        var = pd.Series(gyr).rolling(window, center=True,
                                     min_periods=20).var()
        baseline = np.nanmedian(var)
        thresh = max(baseline * 5, 1e-6)
        excited = (var > thresh).values

        # full signal, dimmed
        ax.plot(t, gyr_deg, color=color, alpha=0.25, lw=0.5,
                label='Full signal')

        # excited segments, bright
        gyr_exc = gyr_deg.copy()
        gyr_exc[~excited] = np.nan
        ax.plot(t, gyr_exc, color=color, lw=1.0,
                label='Excited segment')

        # shade excited regions
        in_segment = False
        for j in range(len(excited)):
            if excited[j] and not in_segment:
                seg_start = t[j]
                in_segment = True
            elif not excited[j] and in_segment:
                ax.axvspan(seg_start, t[j-1], alpha=0.08, color=color)
                in_segment = False
        if in_segment:
            ax.axvspan(seg_start, t[-1], alpha=0.08, color=color)

        pct = 100 * np.sum(excited) / max(len(excited), 1)
        ax.set_ylabel(f'{label} [deg/s]')
        ax.set_title(
            f'{label} → {inertia}:  {pct:.0f}% of window excited',
            fontsize=9, loc='left')
        ax.legend(loc='upper right', framealpha=0.85)
        ax.grid(True, alpha=0.3)

    axs[-1].set_xlabel('Time [s]')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outfile, bbox_inches='tight')
    print(f"Saved {outfile}")
    plt.close()


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    df = load(csv_path)
    print(f"Loaded {len(df)} armed samples from {csv_path}")
    figure_overview(df)
    figure_excitation(df)
