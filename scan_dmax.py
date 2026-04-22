"""Scan Dmax and compare log-evidence to find the optimal particle size."""

import numpy as np
import matplotlib.pyplot as plt
from prsolve import solve_pr

# ---------------------------------------------------------------------------
# Data range — set q_min / q_max to restrict the usable q window.
# q_max drives real-space resolution (Δr ~ π/q_max) and the ICF width;
# trimming it prevents noisy high-q points from pulling the reconstruction.
# ---------------------------------------------------------------------------
q_min = 0.15    # nm⁻¹  (0 = no lower cut)
q_max = 3.5    # nm⁻¹  (practical upper limit; override if data quality allows)

# ---------------------------------------------------------------------------
# Load and filter data
# ---------------------------------------------------------------------------
data = np.loadtxt("SASDA32.dat", skiprows=4, usecols=(0, 1, 2), max_rows=2168)
q, I_obs, sigma = data[:, 0], data[:, 1], data[:, 2]

mask = (sigma > 0) & (I_obs / sigma > 1.0) & (q >= q_min) & (q <= q_max)
q, I_obs, sigma = q[mask], I_obs[mask], sigma[mask]
N = len(q)

# ICF width: half the real-space resolution limit (π / 2 q_max)
icf_width = np.pi / (2.0 * q.max())
print(f"Data points: {N},  q range: {q[0]:.3f} – {q[-1]:.3f} nm⁻¹")
print(f"ICF width  : {icf_width:.3f} nm  (= π / 2q_max)\n")

# ---------------------------------------------------------------------------
# Dmax scan
# ---------------------------------------------------------------------------
dmax_values = np.arange(6.0, 16.5, 0.5)   # nm
results = []

print(f"{'Dmax':>6}  {'conv':>5}  {'iter':>5}  {'alpha':>8}  "
      f"{'c²':>6}  {'chi²/c²+G':>10}  {'log_evid':>10}")
print("-" * 65)

for Dmax in dmax_values:
    res = solve_pr(q, I_obs, sigma, Dmax, n_r=60, tol=0.01, icf_width=icf_width)
    results.append((Dmax, res))
    chi2_sc = res.chi2 / res.c2
    print(f"{Dmax:6.1f}  {str(res.converged):>5}  {res.iterations:5d}  "
          f"{res.alpha:8.3f}  {res.c2:6.3f}  {chi2_sc + res.G:10.1f}  "
          f"{res.log_evidence:10.2f}")

# ---------------------------------------------------------------------------
# Find best Dmax
# ---------------------------------------------------------------------------
log_evids = np.array([r.log_evidence for _, r in results])
best_idx = np.argmax(log_evids)
best_dmax, best_res = results[best_idx]
print(f"\nBest Dmax = {best_dmax:.1f} nm  (log_evidence = {log_evids[best_idx]:.2f})")

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Panel 1: log-evidence vs Dmax
ax = axes[0]
dmaxs = [d for d, _ in results]
ax.plot(dmaxs, log_evids, "o-", color="steelblue", lw=1.5, ms=5)
ax.axvline(best_dmax, color="tomato", lw=1.5, ls="--", label=f"best {best_dmax:.1f} nm")
ax.set_xlabel("Dmax  (nm)")
ax.set_ylabel("evidence")
ax.set_title("Evidence vs Dmax")
ax.legend()

# Panel 2: P(r) curves for a few Dmax values near the peak
ax = axes[1]
# pick best and two neighbours
highlight = sorted(set([
    max(0, best_idx - 2), max(0, best_idx - 1),
    best_idx,
    min(len(results) - 1, best_idx + 1), min(len(results) - 1, best_idx + 2)
]))
cmap = plt.cm.viridis(np.linspace(0.2, 0.85, len(highlight)))
for color, idx in zip(cmap, highlight):
    d, res = results[idx]
    lw = 2.5 if idx == best_idx else 1.2
    label = f"{d:.1f} nm" + (" ★" if idx == best_idx else "")
    ax.plot(res.r, res.pr, color=color, lw=lw, label=label)
ax.set_xlabel("r  (nm)")
ax.set_ylabel("P(r)")
ax.set_title("P(r) near best Dmax")
ax.legend(fontsize=8)

# Panel 3: data fit for the best Dmax
ax = axes[2]
qr = np.outer(q, best_res.r)
safe_qr = np.where(qr > 1e-12, qr, 1.0)
sinc = np.where(qr > 1e-12, np.sin(safe_qr) / safe_qr, 1.0)
F_fit = 4 * np.pi * np.trapezoid(best_res.pr[None, :] * sinc, best_res.r, axis=1)
ax.errorbar(q, I_obs, yerr=sigma, fmt="k.", ms=2, alpha=0.3, label="data", zorder=1)
ax.plot(q, F_fit, "r-", lw=1.5, label=f"fit  Dmax={best_dmax:.1f} nm", zorder=2)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("q  (nm⁻¹)")
ax.set_ylabel("I(q)")
ax.set_title("Best-evidence fit")
ax.legend()

plt.tight_layout()
plt.savefig("dmax_scan.png", dpi=150)
print("Plot saved to dmax_scan.png")
