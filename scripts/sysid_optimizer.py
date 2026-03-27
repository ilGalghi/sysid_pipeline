"""
sysid_optimizer.py
==================
Decoupled hover SysID on roll / pitch / yaw.
Fits only on excited transient segments, not steady hover.

Reads:  00000002_verified.csv
Writes: sysid_validation_3axis.png

Usage:
  python sysid_optimizer.py [input.csv]
"""

import sys
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEFAULT_CSV = 'data/00000002_verified.csv'

# ── Quad-X mixer (ArduPilot AP_MotorsMatrix convention) ──────────────
#
#   Motor 1 (RCOU.C1): front-right, CCW prop  → angle +45°
#   Motor 2 (RCOU.C2): rear-left,   CCW prop  → angle -135°
#   Motor 3 (RCOU.C3): front-left,  CW  prop  → angle -45°
#   Motor 4 (RCOU.C4): rear-right,  CW  prop  → angle +135°
#
# Roll  =  -sin(45°)(C1) + (-sin(-135°))(C2) + (-sin(-45°))(C3) + (-sin(135°))(C4)
#        ∝ (C2 + C3) - (C1 + C4)
#
# Pitch =  cos(45°)(C1) + cos(-135°)(C2) + cos(-45°)(C3) + cos(135°)(C4)
#        ∝ (C1 + C3) - (C2 + C4)
#
# Yaw   = CCW→-1, CW→+1 :  -C1 - C2 + C3 + C4
#        ∝ (C3 + C4) - (C1 + C2)
# ─────────────────────────────────────────────────────────────────────


def load(filepath):
    print(f"Loading {filepath} ...")
    df = pd.read_csv(filepath)
    df = df.drop_duplicates(subset=['TimeSec']).copy()

    # keep armed
    armed = (df['C1'] > 1100) | (df['C2'] > 1100) | \
            (df['C3'] > 1100) | (df['C4'] > 1100)
    df = df[armed].copy().sort_values('TimeSec').reset_index(drop=True)

    # motor inputs (normalised to 0-1000)
    m1 = df['C1'] - 1000
    m2 = df['C2'] - 1000
    m3 = df['C3'] - 1000
    m4 = df['C4'] - 1000

    # Quad-X pseudo-torques  (signs from AP_MotorsMatrix, see above)
    df['Tau_X'] = (m2 + m3) - (m1 + m4)       # roll
    df['Tau_Y'] = (m1 + m3) - (m2 + m4)       # pitch
    df['Tau_Z'] = (m3 + m4) - (m1 + m2)       # yaw

    print(f"  {len(df)} armed samples")
    return df


def isolate_excited(signal, window=200, threshold_factor=5.0):
    """Return boolean mask marking samples inside excited transients."""
    var = pd.Series(signal).rolling(window, center=True, min_periods=20).var()
    baseline = np.nanmedian(var)
    thresh = max(baseline * threshold_factor, 1e-6)
    return var > thresh


def fit_axis(tau, dot_gyr, mask, axis_label):
    """Fit  dot_gyr = (1/I) * tau + bias  on excited samples only."""
    tau_exc   = tau[mask]
    dot_exc   = dot_gyr[mask]

    if len(tau_exc) < 10:
        print(f"  {axis_label}: not enough excited samples ({np.sum(mask)})")
        return np.array([0.0, 0.0]), np.nan, 0

    def residuals(params, x, y):
        return params[0] * x + params[1] - y

    res = least_squares(residuals, [1e-3, 0.0], args=(tau_exc, dot_exc),
                        loss='huber')
    inv_I, bias = res.x
    rmse = np.sqrt(np.mean(res.fun**2))
    print(f"  {axis_label}: 1/I = {inv_I:+.6f}  bias = {bias:+.4f}  "
          f"RMSE = {rmse:.4f}  ({np.sum(mask)} samples)")
    return res.x, rmse, np.sum(mask)


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    df = load(csv_path)

    axes = [
        ('Roll',  'Tau_X', 'dot_GyrX', 'GyrX', 'Ixx'),
        ('Pitch', 'Tau_Y', 'dot_GyrY', 'GyrY', 'Iyy'),
        ('Yaw',   'Tau_Z', 'dot_GyrZ', 'GyrZ', 'Izz'),
    ]

    print("\nDecoupled axis identification (excited segments only):")
    fits = {}
    for label, tau_col, dot_col, gyr_col, inertia in axes:
        mask = isolate_excited(df[gyr_col].values)
        params, rmse, n = fit_axis(
            df[tau_col].values, df[dot_col].values, mask, label)
        pred = params[0] * df[tau_col].values + params[1]
        fits[label] = {
            'params': params, 'rmse': rmse, 'pred': pred,
            'mask': mask, 'tau_col': tau_col, 'dot_col': dot_col,
            'gyr_col': gyr_col, 'inertia': inertia, 'n': n,
        }

    # ── find the active doublet region for zoom ──────────────────────
    all_masks = np.zeros(len(df), dtype=bool)
    for f in fits.values():
        all_masks |= f['mask']

    if np.any(all_masks):
        excited_times = df['TimeSec'].values[all_masks]
        t_start = max(excited_times.min() - 5, df['TimeSec'].min())
        t_end   = min(excited_times.max() + 10, df['TimeSec'].max())
    else:
        t_start, t_end = df['TimeSec'].min(), df['TimeSec'].max()

    # ── produce figure ───────────────────────────────────────────────
    plt.rcParams.update({
        'font.size': 10, 'axes.titlesize': 11,
        'axes.labelsize': 10, 'legend.fontsize': 8,
        'figure.dpi': 180,
    })

    fig, axs = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    fig.suptitle('Hover SysID — Predicted vs Measured Angular Acceleration',
                 fontsize=13, fontweight='bold', y=0.98)

    colors = ['#2563eb', '#dc2626', '#16a34a']
    t = df['TimeSec'].values

    for i, (label, tau_col, dot_col, gyr_col, inertia) in enumerate(axes):
        f = fits[label]
        ax = axs[i]

        # measured
        ax.plot(t, df[dot_col].values, color=colors[i], alpha=0.35,
                linewidth=0.5, label='Measured')
        # predicted
        ax.plot(t, f['pred'], color='black', linewidth=1.2,
                label=f'Predicted (1/{inertia}·τ + b)')

        # shade the excited region
        mask_arr = f['mask']
        for j in range(1, len(mask_arr)):
            if mask_arr[j] and not mask_arr[j-1]:
                ax.axvline(t[j], color=colors[i], alpha=0.15, linewidth=4)

        ax.set_ylabel(f'$\\dot{{{gyr_col[-1]}}}$ [rad/s²]')
        ax.legend(loc='upper right', framealpha=0.85)
        ax.grid(True, alpha=0.3)

        inv_I = f['params'][0]
        note = ''
        if abs(inv_I) < 1e-4:
            note = '  (insufficient excitation)'
        ax.set_title(
            f'{label} — 1/{inertia} = {inv_I:+.5f}, '
            f'RMSE = {f["rmse"]:.4f}' + note,
            fontsize=9, loc='left')

    axs[-1].set_xlabel('Time [s]')
    axs[0].set_xlim(t_start, t_end)    # zoom to doublet region

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = 'figures/sysid_validation_3axis.png'
    fig.savefig(out, bbox_inches='tight')
    print(f"\nFigure saved: {out}")
    plt.close()


if __name__ == "__main__":
    main()