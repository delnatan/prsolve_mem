"""
Quantified Maximum Entropy reconstruction of P(r) from SAXS data.

Implements the hidden-space iterative algorithm of Gull & Skilling (MEMSYS5
manual, Chapter 4.1) with classic automatic noise scaling (Option 2) and a
Gaussian intrinsic correlation function (ICF) with method-of-images boundary
correction.

Spaces
------
Hidden space : h on L grid points - uncorrelated, metric μ_i = h_i
Visible space: f = C h, same L grid - the actual P(r) estimate
Data space   : I(q) at N scattering vectors

Key equations (Gaussian ICF, visible-space entropy, Gaussian likelihood)
-------------------------------------------------------------------------
ICF           : C[j,i] = G(r_j−r_i) − G(r_j+r_i),  G(x) = exp(−x²/2w²)
               column-normalised; antisymmetric reflection at r=0 gives
               row 0 = 0, so f(0) = 0 exactly  (method of images)
Forward model : F = R f = R C h,  effective kernel R_eff = R C  (N×L)
               R[k,j] = 4π sin(q_k r_j)/(q_k r_j) Δr
Prior         : m_f in visible space (sphere P(r)); entropy drives f → m_f
Entropy       : S(f) = Σ (f − m_f − f log(f/m_f))  ≤ 0, max at f = m_f
               convention: 0·log(0/m) = 0 at the r=0 boundary
Metric tensor : μ_i = h_i  (diagonal approx; exact if C = I)
A matrix      : A = [√h] R_eff^T[σ⁻²]R_eff[√h]   (L×L)
B matrix      : B = I + A/α                        (L×L)
Gradient      : g = −α Cᵀ log(f/m_f) + R_eff^T[σ⁻²](D − R_eff h)
G (good data) : G = Σ λ_i/(α + λ_i)   via eigh(A)
Step          : δh = √h ⊙ U (βI+Λ)⁻¹ Uᵀ(√h⊙g),  β set by trust region
Alpha schedule: secant interpolation on (log α, Ω) history targeting Ω = 1
               Option 2 target: α = G L / (S(G−N))  at convergence
Alpha steering: Ω = G c²/(−2αS) guides α toward α_stop via secant + direct update
Termination   : cloud mismatch H = ½ Σ v_i²/(α+λ_i), Test = 2H/G (MEMSYS5 §2.3)
               H is re-evaluated at α_new (no Newton steps; only the entropy
               weight in g changes).  Test_new < tol ⟹ h is already at MAP
               for α_new ⟹ α has converged ⟹ algorithm stops.
Log evidence  : -N/2 log(L_val - αS) - ½ Σ log(1 + λ_i/α)  [Option 2, c² marginalised out]
Output        : P(r) = f = C h  (smooth visible-space reconstruction)

ICF width default: w = π/(2 q_max)  - half the data resolution limit in real space.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq


@dataclass
class PRResult:
    r: np.ndarray  # r-grid [same units as 1/q], shape (L,)
    pr: np.ndarray  # P(r) reconstruction f = C h, shape (L,)
    h: np.ndarray  # hidden variable at MAP, shape (L,)
    icf_width: float  # Gaussian ICF width used
    alpha: float  # regularisation constant at convergence
    c2: float  # noise-scale factor squared; effective σ_eff = √c² · σ
    chi2: float  # χ² (unscaled, relative to supplied σ)
    G: float  # number of 'good' measurements
    log_evidence: float  # log Pr(D|α) [Option 2, c² marginalised], relative
    iterations: int
    converged: bool


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _saxs_kernel(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Build N×L SAXS forward matrix.

    R[k,j] = 4π sin(q_k r_j)/(q_k r_j) * Δr
    """
    dr = r[1] - r[0]
    qr = np.outer(q, r)
    safe_qr = np.where(qr > 1e-12, qr, 1.0)  # avoid 0/0
    sinc = np.where(qr > 1e-12, np.sin(safe_qr) / safe_qr, 1.0)
    return 4.0 * np.pi * sinc * dr


def _gaussian_icf(r: np.ndarray, width: float) -> np.ndarray:
    """Build L×L Gaussian ICF with zero boundary at r=0 (method of images).

    C[j,i] = G(r_j − r_i) − G(r_j + r_i),   G(x) = exp(−x²/2w²)

    The antisymmetric reflection about r=0 gives row 0 identically zero,
    so f(0) = (C h)[0] = 0 for any h.  All entries are non-negative because
    (r_j + r_i)² ≥ (r_j − r_i)² for r_j, r_i ≥ 0.

    Columns are normalised so Σ_j f_j = Σ_i h_i (total signal preserved).
    The column at r_i = 0 is identically zero (h[0] is decoupled); it is
    left as-is after normalisation.
    """
    diff = r[:, None] - r[None, :]  # r_j - r_i
    rsum = r[:, None] + r[None, :]  # r_j + r_i  (image)
    C = np.exp(-0.5 * (diff / width) ** 2) - np.exp(-0.5 * (rsum / width) ** 2)
    C = np.maximum(C, 0.0)  # numerical safety
    C /= np.maximum(C.sum(axis=0, keepdims=True), 1e-300)  # column-normalise
    return C


def _entropy(f: np.ndarray, m: np.ndarray) -> float:
    """S(f) = Σ(f − m − f log(f/m)).  Global maximum 0 at f = m.

    Convention: 0·log(0/m) = 0, handling f = 0 at the r = 0 boundary.
    """
    log_ratio = np.where(f > 0, np.log(f / m), 0.0)
    return float(np.sum(f - m - f * log_ratio))


def _sphere_prior(r: np.ndarray, Dmax: float, I0: float) -> np.ndarray:
    """Default model: homogeneous-sphere P(r), scaled to match I(0).

    For a uniform sphere of diameter Dmax (radius R = Dmax/2):
        P_sphere(r) = r² (1 - 3r/2Dmax + r³/2Dmax³)   0 ≤ r ≤ Dmax

    Derived from the intersection-volume correlation function γ(r):
        γ(r) = 3(1 - 3r/4R + r³/16R³)  with R = Dmax/2.

    Scaling uses  I(0) = 4π ∫ P(r) dr  so the prior predicts the correct
    zero-angle intensity and gives a sensible starting chi-squared.
    """
    shape = r**2 * (1.0 - 3.0 * r / (2.0 * Dmax) + r**3 / (2.0 * Dmax**3))
    shape = np.maximum(shape, 0.0)
    integral = np.trapezoid(shape, r)
    scale = I0 / (4.0 * np.pi * integral) if integral > 0 else 1.0
    return shape * scale


def _guinier_I0(q: np.ndarray, I_obs: np.ndarray, n_pts: int = 10) -> float:
    """Estimate I(0) by linear Guinier fit: ln I(q) = ln I₀ − Rg²q²/3."""
    n = min(n_pts, len(q))
    coeffs = np.polyfit(q[:n] ** 2, np.log(I_obs[:n]), 1)
    return float(np.exp(coeffs[1]))


def _find_beta(
    lam: np.ndarray, v: np.ndarray, alpha: float, r0_sq: float
) -> float:
    """Find β ≥ α such that Σ v_i²/(β+λ_i)² ≤ r0_sq (trust constraint).

    The trust norm |δr|² = Σ v_i²/(β+λ_i)² is strictly decreasing in β,
    so a simple bisection on a scalar function suffices.
    """

    def trust_norm_sq(beta: float) -> float:
        return float(np.sum(v**2 / (beta + lam) ** 2))

    if trust_norm_sq(alpha) <= r0_sq:
        return alpha

    beta_hi = alpha
    while trust_norm_sq(beta_hi) > r0_sq:
        beta_hi *= 2.0

    return brentq(
        lambda b: trust_norm_sq(b) - r0_sq, alpha, beta_hi, xtol=1e-8
    )


def _next_log_alpha(omega_table: list[tuple[float, float]]) -> float:
    """Estimate next log(α) to drive Ω → 1.

    Strategy:
    - If history brackets Ω = 1 (one point above, one below), use linear
      interpolation in log(α) to land at Ω = 1 (secant).
    - Otherwise all points are on the same side:
        * Ω < 1  → over-regularised, decrease α.  Step size is proportional
                   to distance from Ω = 1, capped at one decade.
        * Ω > 1  → under-regularised, increase α by factor 2.
    """
    la_curr, omega_curr = omega_table[-1]

    # Search history for a bracket
    if len(omega_table) >= 2:
        for i in range(len(omega_table) - 1, 0, -1):
            la1, O1 = omega_table[i - 1]
            la2, O2 = omega_table[i]
            if (O1 - 1.0) * (O2 - 1.0) < 0.0:  # bracket around Ω = 1
                dO = O2 - O1
                dl = la2 - la1
                if abs(dO) > 1e-6:
                    la_target = la1 + (1.0 - O1) * dl / dO
                    # Stay inside the bracket (plus a small margin)
                    lo, hi = min(la1, la2), max(la1, la2)
                    return float(np.clip(la_target, lo - 0.5, hi + 0.5))

    # No bracket: move toward Ω = 1 monotonically
    if omega_curr > 1.0:
        return la_curr + np.log(2.0)  # over-shot: nudge α up
    else:
        # step ∝ log(1/Ω), clipped to [log 2, log 10]
        step = np.clip(
            np.log(max(1.0 / (omega_curr + 1e-30), 2.0)),
            np.log(2.0),
            np.log(10.0),
        )
        return la_curr - float(step)


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------


def solve_pr(
    q: np.ndarray,
    I_obs: np.ndarray,
    sigma: np.ndarray,
    Dmax: float,
    n_r: int = 50,
    max_iter: int = 500,
    tol: float = 0.01,
    rate: float = 1.0,
    alpha_init: float | None = None,
    n_guinier: int = 10,
    icf_width: float | None = None,
) -> PRResult:
    """Reconstruct P(r) from SAXS data via Quantified Maximum Entropy with ICF.

    Parameters
    ----------
    q        : scattering vectors (N,), units must match Dmax (e.g. both nm)
    I_obs    : observed intensities (N,)
    sigma    : measurement uncertainties / standard deviations (N,)
    Dmax     : maximum particle dimension (same unit as 1/q)
    n_r      : number of r-grid points (hidden and visible space, L)
    max_iter : maximum outer iterations
    tol      : convergence tolerance (UTOL in MEMSYS5 parlance, default 0.01)
    rate     : trust-region size multiplier (RATE, default 1.0)
    alpha_init : initial regularisation constant; auto-set if None
    n_guinier  : number of low-q points used for Guinier I(0) estimate
    icf_width  : Gaussian ICF width w (same units as r).  Default: π/(2 q_max),
                 which is half the data resolution limit in real space.

    Returns
    -------
    PRResult with fields: r, pr, alpha, c2, chi2, G, log_evidence,
    iterations, converged.
    pr is the visible-space P(r) = C h (smooth, ICF-blurred reconstruction).

    Notes
    -----
    Uses Option 2 (automatic noise scaling).  The supplied σ values are
    treated as relative; the algorithm infers an overall scale factor c so
    that the effective noise is c·σ.  At convergence:
        c² = 2(L − αS)/N      (MAP noise-scale factor squared)
        χ²/c² + G = N         (scaled misfit + good data = total data)
    The returned chi2 is unscaled (relative to the supplied σ); c2 is the
    inferred scale factor squared.  log_evidence uses the Option 2 formula
    with c² marginalised out: −N/2·log(L−αS) − ½Σlog(1+λᵢ/α).
    """
    N = len(q)
    r = np.linspace(0.0, Dmax, n_r)

    # ICF: Gaussian with width w = π/(2 q_max) by default
    if icf_width is None:
        icf_width = np.pi / (2.0 * float(q.max()))
    C = _gaussian_icf(r, icf_width)

    R = _saxs_kernel(q, r)  # N×L raw SAXS kernel
    R_eff = R @ C  # N×L effective kernel: R_eff = R C
    W = R_eff / sigma[:, None]  # noise-scaled effective kernel

    # Prior in visible space: sphere P(r) scaled to Guinier I(0).
    # Entropy drives f = C h toward m_f; h_floor uses same values as baseline.
    I0 = _guinier_I0(q, I_obs, n_guinier)
    m_f = np.maximum(_sphere_prior(r, Dmax, I0), 1e-30)

    h = m_f.copy()

    # Initial alpha: large enough so entropy dominates and h stays near m_f
    if alpha_init is None:
        F0 = R_eff @ m_f
        chi2_0 = float(np.sum(((I_obs - F0) / sigma) ** 2))
        alpha = max(chi2_0 / (2.0 * n_r), 1.0)
    else:
        alpha = float(alpha_init)

    omega_table: list[tuple[float, float]] = []  # (log α, Ω) history
    Omega = 0.0
    Test = np.inf
    converged = False

    h_floor = m_f * 1e-6  # relative floor on hidden variable

    for it in range(max_iter):
        h = np.maximum(h, h_floor)

        # ---- Scalars ----
        S = _entropy(h, m_f)
        F = R_eff @ h
        resid = (I_obs - F) / sigma
        chi2 = float(np.dot(resid, resid))
        L_val = 0.5 * chi2

        # ---- Hidden-space matrices  A = [√h] R_eff^T[σ⁻²]R_eff[√h]  (L×L) ----
        sqrth = np.sqrt(h)
        Wh = W * sqrth[None, :]
        A = Wh.T @ Wh
        lam, U = np.linalg.eigh(A)
        lam = np.maximum(lam, 0.0)

        G = float(np.sum(lam / (alpha + lam)))

        # ---- Gradient of Q = αS − L (entropy in hidden space) ----
        g = -alpha * np.log(h / m_f) + W.T @ resid

        # ---- Trust-region Newton step  δh = √h ⊙ U(βI+Λ)⁻¹Uᵀ(√h⊙g) ----
        v = U.T @ (sqrth * g)
        r0_sq = (rate * np.sqrt(np.sum(h))) ** 2
        beta = _find_beta(lam, v, alpha, r0_sq)
        dh = sqrth * (U @ (v / (beta + lam)))

        h = h + dh
        h = np.maximum(h, h_floor)

        # ---- Alpha update ----
        if S < -1e-20 and (N - G) > 1e-6:
            # Option 2 condition:  Ω = G c²/(-2αS) = 1
            #   with c² = 2(L_val - αS)/N
            c2 = 2.0 * (L_val - alpha * S) / N
            Omega = G * c2 / (-2.0 * alpha * S)
            H = 0.5 * float(np.sum(v**2 / (alpha + lam)))
            Test = 2.0 * H / G if G > 0 else np.inf

            if Test < tol:
                # h is at the MaxEnt solution for this alpha.
                # Use the closed-form Option 2 target directly:
                #   Ω = 1  ⟺  α = G L / (S (G − N))
                alpha_new = float(np.clip(
                    G * L_val / (S * (G - N)), alpha * 0.1, alpha * 10.0
                ))
                # Cloud mismatch termination (MEMSYS5 §2.3, p.27-28):
                # Re-evaluate H at alpha_new without Newton steps.
                # Only the entropy weight α changes in the gradient; A, U, lam
                # and the data residual are unchanged because h has not moved.
                g_new = -alpha_new * np.log(h / m_f) + W.T @ resid
                v_new = U.T @ (sqrth * g_new)
                G_new = float(np.sum(lam / (alpha_new + lam)))
                H_new = 0.5 * float(np.sum(v_new**2 / (alpha_new + lam)))
                Test_new = 2.0 * H_new / G_new if G_new > 0 else np.inf
                # Test_new < tol: h is already at MAP for alpha_new, so alpha
                # has converged to the stopping value → terminate.
                if Test_new < tol:
                    converged = True
                    alpha = alpha_new
                    break
                alpha = alpha_new
            else:
                # h still moving - use table/secant to steer α toward Ω = 1
                omega_table.append((np.log(alpha), Omega))
                alpha = float(np.exp(_next_log_alpha(omega_table)))
        else:
            # h still very close to m (S ≈ 0): force alpha down by a decade
            alpha *= 0.1

    # ---- Polish h for the final alpha ----
    # The main loop may exit just after an alpha change (one Newton step with
    # new α), leaving g non-zero. Continue iterating with fixed alpha until
    # Test is tight so h is truly at the MaxEnt solution.
    for _ in range(200):
        h = np.maximum(h, h_floor)
        sqrth = np.sqrt(h)
        Wh = W * sqrth[None, :]
        A = Wh.T @ Wh
        lam_p, U_p = np.linalg.eigh(A)
        lam_p = np.maximum(lam_p, 0.0)
        F_p = R_eff @ h
        resid_p = (I_obs - F_p) / sigma
        g_p = -alpha * np.log(h / m_f) + W.T @ resid_p
        v_p = U_p.T @ (sqrth * g_p)
        H_p = 0.5 * float(np.sum(v_p**2 / (alpha + lam_p)))
        G_p = float(np.sum(lam_p / (alpha + lam_p)))
        Test_p = 2.0 * H_p / G_p if G_p > 0 else np.inf
        if Test_p < tol * 0.01:
            break
        r0_sq_p = (rate * np.sqrt(np.sum(h))) ** 2
        beta_p = _find_beta(lam_p, v_p, alpha, r0_sq_p)
        dh_p = sqrth * (U_p @ (v_p / (beta_p + lam_p)))
        h = h + dh_p
        h = np.maximum(h, h_floor)

    # ---- Final output quantities ----
    f = C @ h  # visible P(r): smooth ICF-blurred reconstruction
    F = R @ f  # predicted data from visible P(r)
    chi2 = float(np.sum(((I_obs - F) / sigma) ** 2))
    S = _entropy(h, m_f)
    L_val = 0.5 * chi2
    sqrth = np.sqrt(h)
    Wh = W * sqrth[None, :]
    lam = np.maximum(np.linalg.eigvalsh(Wh.T @ Wh), 0.0)
    G = float(np.sum(lam / (alpha + lam)))
    # Option 2 noise-scale factor: c² = 2(L − αS)/N
    c2 = 2.0 * (L_val - alpha * S) / N
    # Log-evidence with c² marginalised out (Option 2, up to additive constants):
    #   log Pr(D|α) ∝ −N/2·log(L − αS) − ½ Σ log(1 + λᵢ/α)
    log_evidence = -0.5 * N * float(np.log(L_val - alpha * S)) - 0.5 * float(
        np.sum(np.log1p(lam / alpha))
    )

    return PRResult(
        r=r,
        pr=f,
        h=h,
        icf_width=icf_width,
        alpha=alpha,
        c2=c2,
        chi2=chi2,
        G=G,
        log_evidence=log_evidence,
        iterations=it + 1,
        converged=converged,
    )


# ---------------------------------------------------------------------------
# Posterior sampling
# ---------------------------------------------------------------------------


def sample_pr(
    q: np.ndarray,
    I_obs: np.ndarray,
    sigma: np.ndarray,
    result: PRResult,
    n_samples: int = 100,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw posterior samples of P(r) via Gaussian approximation at the MAP.

    At the MAP ĥ the posterior is approximately Gaussian.  In the rescaled
    coordinates t = δh / √ĥ the covariance is

        Cov(t) = U diag(1/(α + λ_i)) Uᵀ

    where (λ, U) come from the hidden-space A matrix A = [√ĥ] W^T W [√ĥ].
    Each sample is therefore

        h_sample = ĥ + √ĥ ⊙ U @ (ε / √(α + λ)),   ε ~ N(0, I)
        f_sample = C @ max(h_sample, 0)

    Parameters
    ----------
    q, I_obs, sigma : same arrays passed to solve_pr
    result          : MAP solution returned by solve_pr
    n_samples       : number of posterior samples to draw
    rng             : numpy Generator; created fresh if None

    Returns
    -------
    samples : ndarray, shape (n_samples, L)
        Each row is a sampled visible P(r) curve f = C h_sample.
    """
    if rng is None:
        rng = np.random.default_rng()

    r = result.r
    h_hat = result.h
    alpha = result.alpha
    L = len(r)

    # Rebuild ICF and effective kernel at the MAP r-grid
    C = _gaussian_icf(r, result.icf_width)
    R_mat = _saxs_kernel(q, r)
    W = (R_mat @ C) / sigma[:, None]

    # Eigendecomposition of A at ĥ
    sqrth = np.sqrt(h_hat)
    Wh = W * sqrth[None, :]
    lam, U = np.linalg.eigh(Wh.T @ Wh)
    lam = np.maximum(lam, 0.0)

    # Posterior width in each eigendirection: 1/sqrt(α + λ_i)
    width = 1.0 / np.sqrt(alpha + lam)

    samples = np.empty((n_samples, L))
    for i in range(n_samples):
        eps = rng.standard_normal(L)
        # t ~ N(0, U diag(1/(α+λ)) Uᵀ)
        t = U @ (width * (U.T @ eps))
        # Do NOT clamp h_s here: clamping would bias E[h_s] above h_hat
        # (negative-h tail gets folded up), shifting the whole band above MAP.
        # Instead we let h_s go negative so E[f_sample] = C @ h_hat = MAP,
        # and clip only at the visible f level after applying the ICF.
        h_s = h_hat + sqrth * t
        f_s = C @ h_s
        np.maximum(f_s, 0.0, out=f_s)  # P(r) ≥ 0 in visible space
        samples[i] = f_s

    return samples
