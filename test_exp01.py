"""Quick test of P(r) reconstruction on BSA SAXS data (SASDA32.dat)."""

import matplotlib.pyplot as plt
import numpy as np

from prsolve import sample_pr, solve_pr

# ---------------------------------------------------------------------------
# Data range — set q_min / q_max to restrict the usable q window.
# q_max drives real-space resolution (Δr ~ π/q_max) and the ICF width;
# trimming it prevents noisy high-q points from pulling the reconstruction.
# ---------------------------------------------------------------------------
q_min = 0.01  # nm⁻¹  (0 = no lower cut)
q_max = 0.45  # nm⁻¹  (practical upper limit; override if data quality allows)

# Load data (4-line header, then q / I / sigma columns)
data = np.loadtxt("CHCHD4-WT.dat", usecols=(0, 1, 2))
q, I_obs, sigma = data[:, 0], data[:, 1], data[:, 2]

# Apply S/N and q-range filters
mask = (sigma > 0) & (I_obs / sigma > 1.0) & (q >= q_min) & (q <= q_max)
q, I_obs, sigma = q[mask], I_obs[mask], sigma[mask]
print(f"Data points used: {len(q)}, q range: {q[0]:.3f} – {q[-1]:.3f} nm⁻¹")

# BSA: Dmax ~ 9 nm is typical; adjust as needed
Dmax = 120.0  # nm

# ICF width: half the real-space resolution limit (π / 2 q_max)
icf_width = np.pi / (2.0 * q.max())
print(
    f"ICF width  : {icf_width:.3f} nm  (= π / 2q_max, q_max = {q.max():.3f} nm⁻¹)"
)

result = solve_pr(q, I_obs, sigma, Dmax, n_r=60, tol=0.01, icf_width=icf_width)

N = len(q)
c2 = result.c2

print(f"\nConverged : {result.converged}  ({result.iterations} iterations)")
print(f"alpha     : {result.alpha:.4g}")
print(f"c²        : {c2:.4f}  (noise-scale factor; σ_eff = {c2**0.5:.3f}·σ)")
print(f"chi²      : {result.chi2:.2f}  (raw, vs supplied σ)")
print(
    f"chi²/c²   : {result.chi2 / c2:.2f}  (scaled, should ≈ N−G = {N - result.G:.1f})"
)
print(f"chi²/c²+G : {result.chi2 / c2 + result.G:.1f}  (should ≈ N = {N})")
print(f"G         : {result.G:.1f}  (good measurements)")
print(f"log evid  : {result.log_evidence:.2f}")

# Posterior samples for confidence band
Nsamples = 500
print(f"\nDrawing {Nsamples:d} posterior samples...")
samples = sample_pr(q, I_obs, sigma, result, n_samples=Nsamples)
lo, hi = np.percentile(samples, [5, 95], axis=0)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

ax = axes[0]
ax.fill_between(
    result.r, lo, hi, color="steelblue", alpha=0.25, label="90% band"
)
ax.plot(result.r, result.pr, color="steelblue", lw=2, label="MAP")
ax.axhline(y=0, linestyle="--", c="#888888", lw=0.8)
ax.set_xlabel("r  (nm)")
ax.set_ylabel("P(r)")
ax.set_title(f"BSA  P(r)  —  Dmax = {Dmax} nm")
ax.set_xlim(0, Dmax)
ax.legend(fontsize=9)

qr = np.outer(q, result.r)
safe_qr = np.where(qr > 1e-12, qr, 1.0)
sinc = np.where(qr > 1e-12, np.sin(safe_qr) / safe_qr, 1.0)
F_fit = 4 * np.pi * np.trapezoid(result.pr[None, :] * sinc, result.r, axis=1)
axes[1].errorbar(
    q, I_obs, yerr=sigma, fmt="k.", ms=2, alpha=0.4, label="data", zorder=1
)
axes[1].plot(q, F_fit, "r-", lw=1.5, label="MaxEnt fit", zorder=2)
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_xlabel("q  (nm⁻¹)")
axes[1].set_ylabel("I(q)")
axes[1].legend()
axes[1].set_title("Data fit")

plt.tight_layout()
plt.savefig("bsa_pr.png", dpi=150)
print("\nPlot saved to bsa_pr.png")
