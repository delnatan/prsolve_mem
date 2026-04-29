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
B matrix      : B = βI + A,  β ≥ α                 (L×L)
Gradient      : g = −α log(h/m_f) + R_eff^T[σ⁻²](D − R_eff h)
               (entropy on h in hidden space; gradient is exact ∂(αS−L)/∂h)
G (good data) : G = Σ λ_i/(α + λ_i)   via eigh(A)
Step          : δh = √h ⊙ U (βI+Λ)⁻¹ Uᵀ(√h⊙g),  β set by trust region
Trust region  : Σ v_i²/(β+λ_i)² ≤ rate²·Σh   (hidden-space dist² = |δh/√h|²/Σh)
               MEMSYS MemLb uses Σ λ_i v_i²/(β+λ_i)² (data-space dist), which
               applies after a full CG run.  The unweighted form is appropriate
               for the Python's single-Newton-step-per-iteration architecture.
Alpha schedule: secant interpolation on (log α, log Ω) history targeting Ω = 1
               MEMSYS MemLaw/MemLar works in log-log space; linear Ω fails when
               Ω spans orders of magnitude in early iterations.
               Option 2 target: α = G L / (S(G−N))  at convergence
Alpha steering: Ω = G c²/(−2αS) guides α toward α_stop via secant + direct update
Termination   : Test = 1 − cos(angle(log(h/m_f), W^T resid)) < tol  [MEMSYS MemTest]
               At the MAP: α log(h/m_f) = W^T resid, so vectors are parallel,
               cos = 1, Test = 0.  Neither vector depends on α, so Test is
               alpha-independent: it measures whether h lies on the trajectory.
               Convergence requires Test < tol AND |log(α_new/α)| < tol, i.e.
               both h on the trajectory and α at the stopping value.
Log evidence  : -N/2 log(L_val - αS) - ½ Σ log(1 + λ_i/α)  [Option 2, c² marginalised out]
Output        : P(r) = f = C h  (smooth visible-space reconstruction)

ICF width default: w = π/(2 q_max)  - half the data resolution limit in real space.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
    trace: list | None = field(default=None)  # per-iter debug rows, or None


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

    C[j,i] = G(r_j−r_i) − G(r_j+r_i) − G(r_j+r_i−2·Dmax),  G(x)=exp(−x²/2w²)

    Two antisymmetric reflections enforce zero boundaries:
      r=0   : G(r_j−r_i) − G(r_j+r_i)        → row 0 = 0 for any h
      r=Dmax: G(Dmax−r_i) − G(Dmax−r_i) = 0  → last row ≈ 0 (exact after clip)
    so f(0) = f(Dmax) = 0 for any h.

    Columns are normalised so Σ_j f_j = Σ_i h_i (total signal preserved).
    The column at r_i = 0 is identically zero (h[0] is decoupled); it is
    left as-is after normalisation.
    """
    Dmax = r[-1]
    diff = r[:, None] - r[None, :]  # r_j - r_i
    rsum = r[:, None] + r[None, :]  # r_j + r_i  (image at r=0)
    C = np.exp(-0.5 * (diff / width) ** 2) - np.exp(-0.5 * (rsum / width) ** 2)
    # Method-of-images at r=Dmax: image at 2*Dmax - r_i.
    # At r_j=Dmax the first and third terms cancel exactly (G symmetric),
    # leaving only -G(Dmax+r_i) ≈ 0, so f(Dmax) = 0 after clipping.
    C -= np.exp(-0.5 * ((rsum - 2.0 * Dmax) / width) ** 2)
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


def _guinier_I0(
    q: np.ndarray,
    I_obs: np.ndarray,
    n_pts: int = 10,
    qRg_max: float = 1.3,
) -> float:
    """Estimate I(0) by linear Guinier fit: ln I(q) = ln I₀ − Rg²q²/3.

    Uses a rough Rg to restrict the fit to the Guinier region (q·Rg < qRg_max)
    so the result is independent of q units and q-vector spacing.
    """
    mask = I_obs > 0
    q_pos, I_pos = q[mask], I_obs[mask]
    n_rough = min(n_pts, len(q_pos))
    if n_rough < 2:
        return float(I_pos[0]) if len(I_pos) > 0 else 1.0
    # rough fit to estimate Rg, then restrict to Guinier region
    slope_rough, intercept_rough = np.polyfit(
        q_pos[:n_rough] ** 2, np.log(I_pos[:n_rough]), 1
    )
    Rg_rough = np.sqrt(max(-3.0 * slope_rough, 1e-10))
    guinier_mask = q_pos <= qRg_max / Rg_rough
    n_g = guinier_mask.sum()
    if n_g < 2:
        return float(np.exp(intercept_rough))
    coeffs = np.polyfit(q_pos[guinier_mask] ** 2, np.log(I_pos[guinier_mask]), 1)
    return float(np.exp(coeffs[1]))


def _find_beta(
    lam: np.ndarray, g_eig: np.ndarray, alpha: float, r0_sq: float
) -> float:
    """Find β ≥ α such that Σ g_eig_i²/(β+λ_i)² ≤ r0_sq (trust-region step limit).

    g_eig is the gradient g = ∂(αS−L)/∂h projected into the eigenbasis of A:
    g_eig = Uᵀ(√h ⊙ g).  The hidden-space step distance is then
    d² = Σ g_eig_i²/(β+λ_i)² / summet, and the trust-region constraint
    d ≤ rate is equivalent to Σ g_eig_i²/(β+λ_i)² ≤ rate²·summet = r0_sq.

    Note: MEMSYS MemLb uses a data-space variant Σ λ_i g_eig_i²/(β+λ_i)² ≤ r0_sq
    that applies after a full CG run.  The unweighted form here is appropriate
    for the Python's single-Newton-step-per-iteration architecture, where the
    gradient can be large in all directions at once (not just data-constrained).
    """

    def hidden_dist_sq(beta: float) -> float:
        return float(np.sum(g_eig**2 / (beta + lam) ** 2))

    if hidden_dist_sq(alpha) <= r0_sq:
        return alpha

    beta_hi = alpha
    while hidden_dist_sq(beta_hi) > r0_sq:
        beta_hi *= 2.0

    return brentq(
        lambda b: hidden_dist_sq(b) - r0_sq, alpha, beta_hi, xtol=1e-8
    )


def _next_log_alpha(omega_table: list[tuple[float, float]]) -> float:
    """Estimate next log(α) to drive Ω → 1.

    Table stores (log α, log Ω) with at most one entry per distinct log-α
    (caller deduplicates).  Matches MEMSYS MemLaw/MemLar log-log space.

    Strategy:
    - Sort table by log α, find the tightest adjacent bracket where log Ω
      changes sign, interpolate to log Ω = 0.
    - No bracket yet: monotone step (factor 2 up if Ω>1, |log Ω| down if Ω<1).
    """
    la_curr, lo_curr = omega_table[-1]  # current (most-recently-added) alpha

    if len(omega_table) >= 2:
        sorted_t = sorted(omega_table)  # sort by log α
        best = None
        best_width = np.inf
        for i in range(1, len(sorted_t)):
            la1, lo1 = sorted_t[i - 1]
            la2, lo2 = sorted_t[i]
            if lo1 * lo2 < 0.0:  # adjacent bracket
                w = abs(la2 - la1)
                if w < best_width:
                    best_width = w
                    best = (la1, lo1, la2, lo2)
        if best is not None:
            la1, lo1, la2, lo2 = best
            dlo = lo2 - lo1
            la_target = la1 + (0.0 - lo1) * (la2 - la1) / dlo
            return float(np.clip(la_target, min(la1, la2) - 0.5, max(la1, la2) + 0.5))

    # No bracket yet: conservative monotone step
    if lo_curr > 0.0:  # Ω > 1: under-regularised → increase α
        return la_curr + np.log(2.0)
    else:  # Ω < 1: over-regularised → decrease α
        step = float(np.clip(abs(lo_curr), np.log(2.0), np.log(10.0)))
        return la_curr - step


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------


def solve_pr(
    q: np.ndarray,
    I_obs: np.ndarray,
    sigma: np.ndarray,
    Dmax: float,
    n_r: int = 50,
    max_iter: int = 50,
    tol: float = 0.01,
    rate: float = 1.0,
    alpha_init: float | None = None,
    n_guinier: int = 10,
    icf_width: float | None = None,
    debug: bool = False,
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

    omega_table: list[tuple[float, float]] = []  # (log α, log Ω) history
    Omega = 0.0
    Test = np.inf
    converged = False
    _trace: list[dict] = [] if debug else None  # type: ignore[assignment]

    h_floor = m_f * 1e-6  # relative floor on hidden variable

    for it in range(max_iter):
        h = np.maximum(h, h_floor)
        _alpha_start = alpha  # snapshot before any update this iteration

        # ---- Scalars ----
        S = _entropy(h, m_f)
        F = R_eff @ h
        resid = (I_obs - F) / sigma
        chi2 = float(np.dot(resid, resid))
        L_val = 0.5 * chi2

        # Pre-compute log(h/m_f) once; reused in gradient and alpha clamping.
        log_h_over_m = np.log(h / m_f)
        # Trust-region radius r0 = rate * sqrt(Σh) — shared by h step and
        # the alpha step constraint (MEMSYS5 §2.3, p.27).
        sum_h = float(np.sum(h))
        r0 = rate * np.sqrt(sum_h)
        # Metric-weighted entropy gradient ||√h ⊙ log(h/m_f)||: changing α by
        # Δα shifts ∂Q/∂h by Δα·log(h/m_f), inducing a hidden-space step whose
        # magnitude is bounded by |Δα|·entropy_grad_norm / α.  The trust-region
        # constraint then limits how large Δα can be.
        entropy_grad_norm = float(np.sqrt(np.sum(h * log_h_over_m ** 2)))

        # ---- Hidden-space matrices  A = [√h] R_eff^T[σ⁻²]R_eff[√h]  (L×L) ----
        sqrt_h = np.sqrt(h)
        Wh = W * sqrt_h[None, :]
        A = Wh.T @ Wh
        lam, U = np.linalg.eigh(A)
        lam = np.maximum(lam, 0.0)

        G = float(np.sum(lam / (alpha + lam)))

        # ---- Gradient of Q = αS − L (entropy in hidden space) ----
        likelihood_gradient = W.T @ resid  # hidden-space direction of -∂L/∂h
        g = -alpha * log_h_over_m + likelihood_gradient

        # ---- Convergence test: 1 − cos(angle(log(h/m_f), W^T resid)) ----
        # MEMSYS MemTest §2.3: test = 1 − cos(angle(<23>, <24>)), where <23>
        # is the data-space entropy gradient (Lagrange multipliers) and <24>
        # is the normalised residuals.  In our hidden-space formulation the
        # equivalent vectors are log(h/m_f) and W^T resid: at the MAP,
        # α log(h/m_f) = W^T resid, so they are parallel → cos = 1 → Test = 0.
        # Neither vector depends on α, so Test measures whether h lies on the
        # maximum-entropy trajectory independently of the current α value.
        norm_log = float(np.linalg.norm(log_h_over_m))
        norm_lkl = float(np.linalg.norm(likelihood_gradient))
        if norm_log > 1e-30 and norm_lkl > 1e-30:
            Test = 1.0 - float(
                np.dot(log_h_over_m, likelihood_gradient) / (norm_log * norm_lkl)
            )
        else:
            Test = np.inf

        # ---- Trust-region Newton step  δh = √h ⊙ U(βI+Λ)⁻¹Uᵀ(√h⊙g) ----
        # g_eig: gradient projected into eigenbasis of A; β found by MemLb criterion
        g_eig = U.T @ (sqrt_h * g)
        r0_sq = r0 ** 2
        beta = _find_beta(lam, g_eig, alpha, r0_sq)
        dh = sqrt_h * (U @ (g_eig / (beta + lam)))

        h = h + dh
        h = np.maximum(h, h_floor)

        # ---- Alpha update ----
        # Trust-region constraint on the alpha step (MEMSYS5 §2.3, p.27):
        # changing α by Δα shifts ∂Q/∂h by Δα·log(h/m_f), inducing a step
        # δh whose hidden-space distance is ≈ |Δα|·entropy_grad_norm / α.
        # Requiring that distance ≤ r0 gives the maximum multiplicative factor:
        #   alpha_step_factor = 1 + r0 / entropy_grad_norm
        # This mirrors MEMSYS MemLa:  r = 1 + alpha*r0/agrads  with
        # agrads = alpha*grads and r0 = rate*sqrt(summet).
        if entropy_grad_norm > 1e-30:
            alpha_step_factor = 1.0 + r0 / entropy_grad_norm
        else:
            alpha_step_factor = np.inf  # h at prior: log(h/m)≈0, no constraint
        alpha_lo = _alpha_start / alpha_step_factor
        alpha_hi = _alpha_start * alpha_step_factor

        if S < -1e-20 and (N - G) > 1e-6:
            # Option 2 condition:  Ω = G c²/(-2αS) = 1
            #   with c² = 2(L_val - αS)/N
            c2 = 2.0 * (L_val - alpha * S) / N
            Omega = G * c2 / (-2.0 * alpha * S)

            if Test < tol:
                # h is on the maximum-entropy trajectory (entropy and likelihood
                # gradients aligned).  Use the closed-form Option 2 target:
                #   Ω = 1  ⟺  α = G L / (S (G − N))
                # Clamp to the trust-region alpha step limit.
                alpha_new = float(
                    np.clip(G * L_val / (S * (G - N)), alpha_lo, alpha_hi)
                )
                # Termination: Test is alpha-independent (it compares gradient
                # directions, not magnitudes), so "Test_new < tol" would always
                # equal "Test < tol".  Instead, check directly whether alpha has
                # converged: if alpha_new ≈ alpha, the stopping value is found.
                if abs(np.log(alpha_new / _alpha_start)) < tol:
                    converged = True
                    alpha = alpha_new
                    break
                alpha = alpha_new
            else:
                # h still moving - steer α toward Ω = 1 via table in log-log
                # space (matching MEMSYS MemLaw: table stores log Ω, not Ω).
                # Note: MEMSYS MemLa skips alpha updates when the CG step was
                # distance-penalised (bcodeb=FALSE).  That gate doesn't apply
                # here because the Python takes one Newton step per outer
                # iteration rather than running CG to full convergence per
                # call; beta > alpha is routine early on and cannot be used as
                # a proxy for "h has not converged for the current alpha".
                la_new = float(np.log(alpha))
                lo_new = float(np.log(max(Omega, 1e-30)))
                # Dedup: replace any existing entry within 5% in log α space
                replaced = False
                for _i, (_la, _lo) in enumerate(omega_table):
                    if abs(_la - la_new) < 0.05:
                        omega_table[_i] = (la_new, lo_new)
                        replaced = True
                        break
                if not replaced:
                    omega_table.append((la_new, lo_new))
                    if len(omega_table) > 8:
                        omega_table.pop(0)  # drop oldest when over NSIZE
                alpha_next = float(
                    np.clip(np.exp(_next_log_alpha(omega_table)), alpha_lo, alpha_hi)
                )
                alpha_stable = abs(np.log(alpha_next / _alpha_start)) < tol
                # Primary secondary path: Omega ≈ 1 AND alpha stable.
                # h may have picked up null-space components (in the L-G
                # directions where W^T W ≈ 0) that make Test > tol even
                # though the Option 2 criterion (Omega = 1, alpha stable)
                # is genuinely satisfied.  Declare convergence here — the
                # MaxEnt stopping value has been found.
                omega_ok = abs(Omega - 1.0) < 0.05
                if omega_ok and alpha_stable:
                    converged = True
                    alpha = alpha_next
                    break
                # Fallback secondary path: alpha stable but h is far off the
                # MaxEnt trajectory (Test > rate/(1+rate)).  The model/ICF
                # may be genuinely inconsistent; exit without converged=True.
                bcodet_fail = Test > rate / (1.0 + rate)
                if bcodet_fail and alpha_stable:
                    alpha = alpha_next
                    break
                alpha = alpha_next
        else:
            # h still very close to m (S ≈ 0): step alpha down; entropy_grad_norm
            # is near zero here so alpha_step_factor ≈ inf and clamp is a no-op.
            alpha = float(np.clip(alpha * 0.1, alpha_lo, alpha_hi))

        if debug:
            _trace.append(
                dict(
                    it=it,
                    alpha=_alpha_start,
                    alpha_next=alpha,
                    alpha_lo=alpha_lo,
                    alpha_hi=alpha_hi,
                    alpha_step_factor=alpha_step_factor,
                    Omega=Omega,
                    Test=Test,
                    G=G,
                    S=S,
                    beta=beta,
                    chi2=chi2,
                    c2=(2.0 * (L_val - _alpha_start * S) / N if S < -1e-20 else float("nan")),
                    entropy_grad_norm=entropy_grad_norm,
                )
            )

    # ---- Final output quantities ----
    f = C @ h  # visible P(r): smooth ICF-blurred reconstruction
    F = R @ f  # predicted data from visible P(r)
    chi2 = float(np.sum(((I_obs - F) / sigma) ** 2))
    S = _entropy(h, m_f)
    L_val = 0.5 * chi2
    sqrt_h = np.sqrt(h)
    Wh = W * sqrt_h[None, :]
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
        trace=_trace,
    )


# ---------------------------------------------------------------------------
# ICF width scan
# ---------------------------------------------------------------------------


def scan_icf_width(
    q: np.ndarray,
    I_obs: np.ndarray,
    sigma: np.ndarray,
    Dmax: float,
    w_grid: np.ndarray,
    n_r: int = 50,
    max_iter: int = 50,
    tol: float = 0.01,
    rate: float = 1.0,
    alpha_init: float | None = None,
    n_guinier: int = 10,
) -> np.ndarray:
    """Scan ICF width and return array with columns:
    [w, log_evidence, converged, G, alpha, iterations].

    Only converged rows carry a meaningful log_evidence; the caller should
    restrict evidence comparisons to rows where converged == 1.
    """
    rows = []
    for w in w_grid:
        res = solve_pr(
            q, I_obs, sigma, Dmax,
            n_r=n_r, max_iter=max_iter, tol=tol, rate=rate,
            alpha_init=alpha_init, n_guinier=n_guinier, icf_width=w,
        )
        rows.append([w, res.log_evidence, float(res.converged),
                     res.G, res.alpha, float(res.iterations)])
    return np.array(rows)


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
    sqrt_h = np.sqrt(h_hat)
    Wh = W * sqrt_h[None, :]
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
        h_s = h_hat + sqrt_h * t
        f_s = C @ h_s
        np.maximum(f_s, 0.0, out=f_s)  # P(r) ≥ 0 in visible space
        samples[i] = f_s

    return samples
