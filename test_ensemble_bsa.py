"""
test_ensemble_bsa.py — Ensemble RAAR density reconstruction for BSA (SASDA32.dat).

Dmax = 14.0 nm, q capped at 4.0 nm⁻¹ (d_min ~ 0.79 nm).
"""

import mrcfile
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from denss_utils import DensityResult, density_to_pr, subsample_q
from denss_raar import solve_density_raar, align_and_average

# ---------------------------------------------------------------------------
# Load and filter data
# ---------------------------------------------------------------------------
data = np.loadtxt("SASDA32.dat", skiprows=4, usecols=(0, 1, 2), max_rows=2168)
q_raw, I_raw, s_raw = data[:, 0], data[:, 1], data[:, 2]

q_max_use = 4.0   # nm⁻¹ — caps resolution at ~0.8 nm
mask = (s_raw > 0) & (I_raw / s_raw > 1.0) & (q_raw <= q_max_use)
q, I_obs, sigma = q_raw[mask], I_raw[mask], s_raw[mask]
print(f"Points after filter : {len(q)}  ({q[0]:.3f} – {q[-1]:.3f} nm⁻¹)")

# Subsample to 128 log-spaced bins
q, I_obs, sigma = subsample_q(q, I_obs, sigma, n_points=128)
print(f"Points after binning: {len(q)}")

# ---------------------------------------------------------------------------
# Ensemble phase retrieval
# ---------------------------------------------------------------------------
Dmax = 14.0   # nm

ensemble = solve_density_raar(
    q, I_obs, sigma, Dmax,
    n_restarts=5,
    n_iter=2000,
    n_er_final=300,
    threshold_sw=0.20,   # raised from 0.10 — less aggressive support pruning
    seed=42,
    verbose=True,
)

print(f"\nRestart χ² values:")
for i, m in enumerate(ensemble.maps):
    marker = " ← best" if m is ensemble.best else ""
    print(f"  restart {i+1}: chi2 = {m.chi2:.4f}{marker}")

# ---------------------------------------------------------------------------
# Align and average
# ---------------------------------------------------------------------------
averaged = align_and_average(ensemble, angle_step=20.0, verbose=True)

# ---------------------------------------------------------------------------
# Save averaged density as MRC
# ---------------------------------------------------------------------------
dr_A = averaged.dr * 10.0                    # nm → Å (MRC convention)
origin_A = float(averaged.x_grid[0] * 10.0) # centre of voxel [0,0,0] in Å

with mrcfile.new("bsa_ensemble_mean.mrc", overwrite=True) as mrc:
    mrc.set_data(averaged.density.astype(np.float32))
    mrc.voxel_size = dr_A
    mrc.header.origin.x = origin_A
    mrc.header.origin.y = origin_A
    mrc.header.origin.z = origin_A

print(f"MRC saved  →  bsa_ensemble_mean.mrc  "
      f"({averaged.density.shape[0]}³ voxels, {dr_A:.3f} Å/voxel)")

# ---------------------------------------------------------------------------
# P(r) — light Gaussian pre-smooth reduces grid-aliasing ripple
# ---------------------------------------------------------------------------
def _pr_from_density(density, x_grid, dr, Dmax):
    return density_to_pr(DensityResult(
        density=gaussian_filter(density, sigma=1.0),
        x_grid=x_grid, dr=dr, Dmax=Dmax,
        chi2=np.nan, scale=np.nan,
        I_pred=np.array([]), loss_history=np.array([]),
    ))

r_best, pr_best = _pr_from_density(
    ensemble.best.density, ensemble.best.x_grid, ensemble.best.dr, Dmax
)
r_mean, pr_mean = _pr_from_density(
    averaged.density, averaged.x_grid, averaged.dr, Dmax
)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# --- 1. Fit quality (best map) ---
ax = axes[0]
covered = ensemble.best.I_pred > 0
ax.errorbar(q, I_obs, yerr=sigma, fmt='k.', ms=2, alpha=0.35, label='data', zorder=1)
ax.plot(q[covered], ensemble.best.I_pred[covered], 'r-', lw=1.5,
        label=f'best fit  χ²={ensemble.best.chi2:.3f}')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('q  (nm⁻¹)'); ax.set_ylabel('I(q)')
ax.set_title('Scattering fit (best restart)')
ax.legend(fontsize=8)

# --- 2. P(r) comparison ---
ax = axes[1]
ax.plot(r_best, pr_best, 'steelblue', lw=2, label='best restart')
ax.plot(r_mean, pr_mean, 'tomato',    lw=2, label='ensemble mean', ls='--')
ax.axhline(0, color='#888', lw=0.7, ls='--')
ax.set_xlabel('r  (nm)'); ax.set_ylabel('P(r)  (normalised)')
ax.set_title('P(r)  —  BSA  Dmax = 14.0 nm')
ax.set_xlim(0, Dmax)
ax.legend(fontsize=8)

# --- 3. Central slice of mean density with std contour ---
ax = axes[2]
N   = averaged.density.shape[0]
sl  = averaged.density[:, :, N // 2]
sl_std = averaged.std[:, :, N // 2]
dr  = averaged.dr
extent = [-N/2 * dr, N/2 * dr, -N/2 * dr, N/2 * dr]
im = ax.imshow(sl.T, origin='lower', extent=extent, cmap='Blues')
ax.contour(sl_std.T, levels=3, extent=extent, colors='tomato', linewidths=0.7, alpha=0.7)
fig.colorbar(im, ax=ax, shrink=0.8, label='ρ (a.u.)')
ax.set_xlabel('x  (nm)'); ax.set_ylabel('y  (nm)')
ax.set_title('Mean density — central z-slice\n(red contours: std)')

plt.tight_layout()
plt.savefig('bsa_ensemble.png', dpi=150)
print("Plot saved  →  bsa_ensemble.png")
