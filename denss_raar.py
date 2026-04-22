"""
denss_raar.py — RAAR iterative phase retrieval for SAXS 3D density (MLX).

Implements the Relaxed Averaged Alternating Reflections (RAAR) algorithm
(Luke 2004) for recovering 3D electron density from 1D SAXS data.

No automatic differentiation — two FFTs per iteration, all other operations
are O(N³) element-wise numpy/MLX.

RAAR update
-----------
    ρ_M      = P_M(ρ_n)
    ρ_S      = P_S(ρ_n)
    ρ_{n+1}  = β(ρ_n + P_S(2ρ_M − ρ_n) − ρ_M) + (1−β) ρ_S

Projections
-----------
    P_M : hard Fourier magnitude projection (shell-average with interpolated
          targets).  FFT shells are defined by integer k² = kx²+ky²+kz²;
          the target mean|F|² for each shell is obtained by log-log
          interpolation of I_obs at that shell's |q⃗|.  Every shell that
          falls inside [q_min, q_max] is constrained — no empty-bin problem.

    P_S : hard real-space projection.
          ρ → clip(ρ, min=0) × support_mask

Shrinkwrap
----------
    Every `shrinkwrap_interval` RAAR iterations, the support is refined:
      1. Gaussian-blur max(ρ, 0) with σ decreasing from σ_max → σ_min.
      2. Threshold at `threshold_sw` × max of the blurred density.
      3. Intersect with the initial sphere (support can only shrink).

After RAAR, a final ER polishing pass (ρ_{n+1} = P_S(P_M(ρ_n))) is applied.

MLX is used for 3D FFT/IFFT (Metal GPU); shell averaging and scaling are done
in numpy, which on Apple Silicon is a zero-copy view of unified memory.
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from typing import Optional

import numpy as np
import mlx.core as mx
from scipy.ndimage import affine_transform, gaussian_filter
from scipy.spatial.transform import Rotation as Rot

from denss_utils import (
    AveragedDensityResult,
    DensityResult,
    EnsembleDensityResult,
    subsample_q,
    density_to_pr,
    _nyquist_grid_size,
    _center_by_com,
)

__all__ = ["solve_density_raar", "align_and_average", "EnsembleDensityResult"]


# ---------------------------------------------------------------------------
# Reciprocal-space grid helpers
# ---------------------------------------------------------------------------

def _build_fft_shells(N: int, dr: float) -> tuple[np.ndarray, np.ndarray]:
    """Build integer-k² based FFT shells.

    Returns
    -------
    shell_inverse : (N³,) int64 — index into unique shells for each voxel
    q_shells      : (n_shells,) float64 — physical |q| for each unique shell
    """
    k = np.fft.fftfreq(N) * N                        # integer wavenumbers
    kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
    k2_flat = (kx**2 + ky**2 + kz**2).ravel().astype(np.int32)
    unique_k2, shell_inverse = np.unique(k2_flat, return_inverse=True)
    dq = 2.0 * np.pi / (N * dr)
    q_shells = np.sqrt(unique_k2.astype(np.float64)) * dq
    return shell_inverse, q_shells


def _build_proj_args(
    shell_inverse: np.ndarray,
    q_shells: np.ndarray,
    q_data: np.ndarray,
    I_obs: np.ndarray,
    q_min: float,
    q_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute per-shell interpolated target intensities.

    Returns
    -------
    shell_inverse   : (N³,) int64
    in_range        : (n_shells,) bool — shells inside data q range
    I_target_shells : (n_shells,) float64 — interpolated I_obs per shell
    """
    n_shells = len(q_shells)
    in_range = (q_shells >= q_min) & (q_shells <= q_max) & (q_shells > 0.0)

    log_q_data = np.log(q_data.astype(np.float64))
    log_I_obs  = np.log(I_obs.astype(np.float64))

    I_target = np.zeros(n_shells, dtype=np.float64)
    I_target[in_range] = np.exp(
        np.interp(np.log(q_shells[in_range]), log_q_data, log_I_obs)
    )
    return shell_inverse, in_range, I_target


def _q_mag_grid(N: int, dr: float) -> np.ndarray:
    """(N³,) float32 array of physical |q⃗| for χ² monitoring."""
    k = np.fft.fftfreq(N) * N
    kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
    dq = 2.0 * np.pi / (N * dr)
    return (np.sqrt(kx**2 + ky**2 + kz**2) * dq).ravel().astype(np.float32)


def _build_shell_arrays(
    q_mag_flat: np.ndarray, q_data: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shell index, validity mask, and per-shell counts for χ² monitoring."""
    edges = np.empty(len(q_data) + 1)
    edges[0] = 0.0
    edges[1:-1] = 0.5 * (q_data[:-1] + q_data[1:])
    edges[-1] = np.inf

    shell_idx = np.searchsorted(edges, q_mag_flat, side='right') - 1
    shell_idx = shell_idx.clip(0, len(q_data) - 1).astype(np.int32)

    valid = (q_mag_flat >= float(q_data.min())) & (q_mag_flat <= float(q_data.max()))
    shell_counts = np.bincount(shell_idx[valid], minlength=len(q_data)).astype(np.float32)
    return shell_idx, valid, shell_counts


# ---------------------------------------------------------------------------
# Hard projections
# ---------------------------------------------------------------------------

def _fourier_project(
    rho: np.ndarray,
    shell_inverse: np.ndarray,
    in_range: np.ndarray,
    I_target_shells: np.ndarray,
) -> np.ndarray:
    """Hard Fourier magnitude projection.

    For each FFT shell (unique integer k²), compute mean|F|² over all voxels
    in the shell, then rescale the entire shell so the mean matches the
    interpolated I_obs at that shell's |q⃗|.  Shells outside the data q range
    are left unchanged (scale = 1).

    This is physically correct — SAXS constrains orientational averages, not
    individual voxel amplitudes — and achieves 100 % q-range coverage because
    shells are defined on the FFT grid and targets are interpolated from data.
    """
    N = rho.shape[0]
    n_shells = len(I_target_shells)

    # --- FFT on Metal -------------------------------------------------------
    F_np = np.array(mx.fft.fftn(mx.array(rho)))           # complex64

    # --- Shell-average power ------------------------------------------------
    power = (F_np.real**2 + F_np.imag**2).ravel().astype(np.float64)
    shell_power  = np.bincount(shell_inverse, weights=power, minlength=n_shells)
    shell_counts = np.bincount(shell_inverse,               minlength=n_shells).astype(np.float64)
    mean_power   = shell_power / np.maximum(shell_counts, 1.0)

    # --- Per-shell scale: sqrt(I_target / mean|F|²) -------------------------
    shell_scale = np.where(
        in_range & (mean_power > 1e-30),
        np.sqrt(I_target_shells / np.maximum(mean_power, 1e-30)),
        1.0,
    ).astype(np.float32)

    # --- Broadcast to per-voxel, IFFT on Metal ------------------------------
    scale_3d = shell_scale[shell_inverse].reshape(N, N, N)
    F_scaled  = F_np * scale_3d                            # complex64 × float32
    rho_M     = np.array(mx.fft.ifftn(mx.array(F_scaled)).real)
    return rho_M


def _support_project(rho: np.ndarray, support: np.ndarray) -> np.ndarray:
    """Hard real-space projection: positivity + finite support."""
    return np.clip(rho, 0.0, None) * support


# ---------------------------------------------------------------------------
# Shrinkwrap
# ---------------------------------------------------------------------------

def _shrinkwrap(
    rho: np.ndarray,
    sigma: float,
    threshold_frac: float,
    initial_support: np.ndarray,
) -> np.ndarray:
    """Refine the support by Gaussian-blur + threshold of current density.

    The new support is intersected with the initial sphere so it can only
    shrink, never expand beyond the Dmax constraint.

    Parameters
    ----------
    rho             : current density (negative values treated as zero).
    sigma           : Gaussian blur σ in voxels.
    threshold_frac  : threshold = frac × max(blurred).
    initial_support : the original sphere mask (hard upper bound).

    Returns a float32 binary mask of the same shape as rho.
    """
    blurred   = gaussian_filter(np.maximum(rho, 0.0).astype(np.float64), sigma=sigma)
    threshold = threshold_frac * blurred.max()
    if threshold <= 0.0:
        return initial_support.copy()
    new_support = (blurred > threshold).astype(np.float32) * initial_support
    # Fall back to sphere if shrinkwrap collapses the support entirely
    if new_support.sum() == 0:
        return initial_support.copy()
    return new_support


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

_ProjArgs = tuple[np.ndarray, np.ndarray, np.ndarray]   # shell_inverse, in_range, I_target


def _random_phase_init(
    N: int,
    proj_args: _ProjArgs,
    support: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Amplitude-seeded random-phase initialisation.

    Assigns per-voxel amplitudes sqrt(I_target) from the interpolated data,
    attaches uniform-random phases, then IFFTs into real space.  The result
    is support-projected (positivity + mask).

    This is preferable to pure random noise because the Fourier magnitudes are
    consistent with the measured data from iteration 0, so each subsequent
    projection step is a small correction rather than a wholesale rescaling.
    Out-of-range shells (no data constraint) are initialised to zero amplitude.
    """
    shell_inverse, in_range, I_target_shells = proj_args

    amp_shells = np.where(in_range, np.sqrt(np.maximum(I_target_shells, 0.0)), 0.0)
    amp_3d = amp_shells[shell_inverse].reshape(N, N, N).astype(np.float32)

    phases = rng.uniform(-np.pi, np.pi, (N, N, N)).astype(np.float32)
    F_init = amp_3d * (np.cos(phases) + 1j * np.sin(phases))

    rho_init = np.fft.ifftn(F_init).real.astype(np.float32)
    return _support_project(rho_init, support)


# ---------------------------------------------------------------------------
# RAAR and ER update steps
# ---------------------------------------------------------------------------


def _raar_step(
    rho: np.ndarray,
    support: np.ndarray,
    proj_args: _ProjArgs,
    beta: float,
) -> np.ndarray:
    """One RAAR iteration."""
    rho_M     = _fourier_project(rho, *proj_args)
    rho_S     = _support_project(rho, support)
    rho_S_ref = _support_project(2.0 * rho_M - rho, support)
    return beta * (rho + rho_S_ref - rho_M) + (1.0 - beta) * rho_S


def _er_step(
    rho: np.ndarray,
    support: np.ndarray,
    proj_args: _ProjArgs,
) -> np.ndarray:
    """One Error Reduction step: ρ_{n+1} = P_S(P_M(ρ_n))."""
    return _support_project(_fourier_project(rho, *proj_args), support)


# ---------------------------------------------------------------------------
# χ² monitoring (shell-binned OLS, reported against the data q grid)
# ---------------------------------------------------------------------------

def _chi2_and_pred(
    rho: np.ndarray,
    shell_idx: np.ndarray,
    valid: np.ndarray,
    shell_counts: np.ndarray,
    I_obs: np.ndarray,
    inv_var: np.ndarray,
    n_eff: float,
) -> tuple[float, np.ndarray, float]:
    n_q  = len(I_obs)
    F_np = np.array(mx.fft.fftn(mx.array(rho)))
    power = (F_np.real**2 + F_np.imag**2).ravel()
    shell_sum = np.bincount(shell_idx[valid], weights=power[valid], minlength=n_q)
    I_raw     = shell_sum / np.maximum(shell_counts, 1.0)

    num   = (inv_var * I_raw * I_obs).sum()
    den   = (inv_var * I_raw * I_raw).sum()
    scale = num / max(den, 1e-30)

    I_pred = scale * I_raw
    resid  = (I_pred - I_obs) * np.sqrt(inv_var)
    chi2   = (resid**2).sum() / max(n_eff, 1.0)
    return chi2, I_pred, scale


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def solve_density_raar(
    q: np.ndarray,
    I_obs: np.ndarray,
    sigma: np.ndarray,
    Dmax: float,
    *,
    grid_size: Optional[int] = None,
    oversampling: float = 3.0,
    n_iter: int = 3000,
    n_er_final: int = 200,
    beta: float = 0.87,
    n_restarts: int = 5,
    seed: Optional[int] = None,
    shrinkwrap_interval: int = 50,
    sigma_sw_max: float = 3.0,
    sigma_sw_min: float = 1.5,
    threshold_sw: float = 0.10,
    verbose: bool = True,
) -> DensityResult:
    """RAAR iterative phase retrieval for SAXS 3D electron density.

    Parameters
    ----------
    q, I_obs, sigma : SAXS data (q ascending, sigma > 0).
    Dmax            : maximum particle dimension (same units as 1/q).
    grid_size       : voxels per side; auto-sized to next power-of-2 Nyquist
                      when None.
    oversampling    : real-space box = Dmax × oversampling (≥ 2).
    n_iter          : RAAR iterations per restart.
    n_er_final      : ER polishing steps after RAAR.
    beta            : RAAR feedback ∈ (0, 1].
    n_restarts      : independent random restarts; best χ² result returned.
    seed            : RNG seed.
    shrinkwrap_interval : update support every this many RAAR iterations.
                      Set 0 to disable shrinkwrap.
    sigma_sw_max    : initial Gaussian blur σ in voxels for shrinkwrap.
    sigma_sw_min    : final Gaussian blur σ in voxels for shrinkwrap.
    threshold_sw    : threshold fraction for shrinkwrap (default 0.10).
    verbose         : print progress.

    Returns
    -------
    DensityResult with the best density map and fit diagnostics.
    """
    # --- Input validation ---------------------------------------------------
    q     = np.asarray(q,     dtype=np.float64)
    I_obs = np.asarray(I_obs, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    if q.ndim != 1 or not np.all(np.diff(q) > 0):
        raise ValueError("q must be a 1-D strictly ascending array.")
    if np.any(sigma <= 0):
        raise ValueError("All sigma values must be positive.")

    # --- Grid ---------------------------------------------------------------
    q_max_data = float(q.max())
    n_nyq = _nyquist_grid_size(Dmax, q_max_data, oversampling)

    if grid_size is None:
        grid_size = max(32, int(2 ** math.ceil(math.log2(n_nyq))))
    elif grid_size < n_nyq:
        warnings.warn(
            f"grid_size={grid_size} < Nyquist minimum {n_nyq}.", stacklevel=2
        )

    N   = grid_size
    box = Dmax * oversampling
    dr  = box / N

    if verbose:
        print(
            f"[denss_raar]  grid={N}³  dr={dr:.3f}  box={box:.1f}  "
            f"q_max_FFT={math.pi/dr:.4f}  beta={beta}"
        )

    # --- Real-space grid and initial support (sphere of radius Dmax/2) ------
    x = (np.arange(N) - (N - 1) / 2.0) * dr
    xx, yy, zz = np.meshgrid(x, x, x, indexing='ij')
    initial_support = (np.sqrt(xx**2 + yy**2 + zz**2) <= Dmax / 2.0).astype(np.float32)

    # --- Reciprocal-space shells (FFT integer-k² based) ---------------------
    shell_inverse, q_shells = _build_fft_shells(N, dr)
    q_min_data = float(q.min())
    proj_args  = _build_proj_args(
        shell_inverse, q_shells, q, I_obs, q_min_data, q_max_data
    )

    n_shells_in_range = int(proj_args[1].sum())   # in_range mask
    if verbose:
        dq_fft = 2.0 * np.pi / (N * dr)
        print(
            f"  Fourier projection: shell-average + log-log interp  "
            f"(Δq_FFT={dq_fft:.4f},  {n_shells_in_range} shells in range)"
        )

    # --- Shell arrays for χ² monitoring (data-bin based) -------------------
    q_mag_flat  = _q_mag_grid(N, dr)
    shell_idx, valid, shell_counts = _build_shell_arrays(q_mag_flat, q)
    n_q         = len(q)
    n_covered   = int((shell_counts > 0).sum())
    n_uncovered = n_q - n_covered
    if verbose:
        print(
            f"  χ² monitoring: {n_covered}/{n_q} data q bins"
            + (f"  ({n_uncovered} empty)" if n_uncovered else " — full coverage")
        )

    covered = shell_counts > 0
    inv_var = np.where(covered, 1.0 / sigma**2, 0.0)
    n_eff   = float(covered.sum())

    # --- Multi-restart loop -------------------------------------------------
    rng      = np.random.default_rng(seed)
    log_every = max(1, (n_iter + n_er_final) // 20)

    all_maps: list[DensityResult] = []

    for restart in range(n_restarts):
        support = initial_support.copy()
        rho = _random_phase_init(N, proj_args, support, rng)
        history: list[float] = []

        # --- RAAR phase with shrinkwrap -------------------------------------
        for it in range(n_iter):
            rho = _raar_step(rho, support, proj_args, beta)

            # Shrinkwrap: refine support periodically
            if shrinkwrap_interval > 0 and (it + 1) % shrinkwrap_interval == 0:
                frac = it / max(n_iter - 1, 1)
                sigma_now = sigma_sw_max - (sigma_sw_max - sigma_sw_min) * frac
                support = _shrinkwrap(rho, sigma_now, threshold_sw, initial_support)

            if it % log_every == 0:
                chi2, _, _ = _chi2_and_pred(
                    rho, shell_idx, valid, shell_counts, I_obs, inv_var, n_eff
                )
                history.append(chi2)
                if verbose:
                    n_sup = int(support.sum())
                    print(
                        f"  restart {restart+1}/{n_restarts}  "
                        f"[RAAR] it {it:5d}/{n_iter}  "
                        f"chi2={chi2:.4f}  support={n_sup}"
                    )

        # --- ER polishing ---------------------------------------------------
        for it in range(n_er_final):
            rho = _er_step(rho, support, proj_args)
            if it % log_every == 0:
                chi2, _, _ = _chi2_and_pred(
                    rho, shell_idx, valid, shell_counts, I_obs, inv_var, n_eff
                )
                history.append(chi2)
                if verbose:
                    print(
                        f"  restart {restart+1}/{n_restarts}  "
                        f"[ ER ] it {it:5d}/{n_er_final}  chi2={chi2:.4f}"
                    )

        # --- Final evaluation -----------------------------------------------
        chi2_f, I_pred_f, scale_f = _chi2_and_pred(
            rho, shell_idx, valid, shell_counts, I_obs, inv_var, n_eff
        )
        if verbose:
            print(f"  → restart {restart+1} final chi2={chi2_f:.4f}")

        all_maps.append(DensityResult(
            density=_center_by_com(rho, x),
            x_grid=x,
            dr=dr,
            Dmax=Dmax,
            chi2=chi2_f,
            scale=scale_f,
            I_pred=I_pred_f,
            loss_history=np.asarray(history),
        ))

    return EnsembleDensityResult(maps=all_maps)


# ---------------------------------------------------------------------------
# Alignment helpers
# ---------------------------------------------------------------------------

def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised cross-correlation between two arrays (scalar)."""
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / max(denom, 1e-30))


def _rotate_density(rho: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Rotate density by a 3×3 rotation matrix using trilinear interpolation.

    The rotation is applied about the grid centre so the particle stays
    centred.  affine_transform computes out[o] = in[R^T @ o + offset],
    where offset = (I − R^T) @ centre maps the coordinate origin to the
    box centre before and after rotation.
    """
    N = rho.shape[0]
    c = (N - 1) / 2.0
    centre = np.array([c, c, c])
    offset = centre - R.T @ centre
    return affine_transform(
        rho, R.T, offset=offset, order=1, mode='constant', cval=0.0
    ).astype(rho.dtype)


def _align_to_reference(
    rho: np.ndarray,
    reference: np.ndarray,
    angle_step: float,
) -> np.ndarray:
    """Find the rigid-body alignment of rho that maximises NCC with reference.

    Searches jointly over:
      * both enantiomorphs (rho and its inversion through the origin),
      * ZYZ Euler angles on a regular grid with spacing ``angle_step`` (°).

    The ZYZ parametrisation has gimbal lock at β = 0° and 180°, where many
    (α, γ) pairs describe the same rotation.  The redundant evaluations at
    the poles are harmless for a coarse search and avoided trivially in a
    later simplex refinement pass.

    Returns the rotated/reflected density with the highest NCC.
    """
    alphas = np.arange(0.0, 360.0, angle_step)
    betas  = np.arange(0.0, 180.0 + angle_step / 2.0, angle_step)
    gammas = np.arange(0.0, 360.0, angle_step)

    best_ncc = -np.inf
    best_rho = rho.copy()

    for candidate in (rho, np.flip(rho).copy()):
        for alpha in alphas:
            for beta in betas:
                for gamma in gammas:
                    R = Rot.from_euler(
                        'ZYZ', [alpha, beta, gamma], degrees=True
                    ).as_matrix()
                    rotated = _rotate_density(candidate, R)
                    ncc = _ncc(rotated, reference)
                    if ncc > best_ncc:
                        best_ncc = ncc
                        best_rho = rotated

    return best_rho


# ---------------------------------------------------------------------------
# Public API — ensemble averaging
# ---------------------------------------------------------------------------

def align_and_average(
    ensemble: EnsembleDensityResult,
    angle_step: float = 15.0,
    verbose: bool = True,
) -> AveragedDensityResult:
    """Align all maps in an ensemble to the best-χ² reference, then average.

    Each map (excluding the reference) is aligned by exhaustive search over
    ZYZ Euler angles and both enantiomorphs, maximising the normalised
    cross-correlation (NCC) with the reference density.

    Parameters
    ----------
    ensemble   : output of :func:`solve_density_raar`.
    angle_step : Euler-angle grid spacing in degrees.  15° gives ~15 000
                 trial rotations per map.  Use 20–30° for a quick first pass
                 on large grids; refine later with a simplex optimiser.
    verbose    : print per-map progress and final NCC.

    Returns
    -------
    AveragedDensityResult with voxel-wise mean and std of the aligned maps.
    """
    best_idx = min(range(len(ensemble.maps)), key=lambda i: ensemble.maps[i].chi2)
    reference = ensemble.maps[best_idx].density

    if verbose:
        n = len(ensemble.maps)
        print(
            f"[align_and_average]  {n} maps  angle_step={angle_step}°  "
            f"reference = restart {best_idx + 1}  "
            f"(chi2={ensemble.maps[best_idx].chi2:.4f})"
        )

    aligned_maps: list[DensityResult] = []
    for i, result in enumerate(ensemble.maps):
        if i == best_idx:
            aligned_density = reference.copy()
            if verbose:
                print(f"  map {i+1}/{len(ensemble.maps)}  [reference]")
        else:
            if verbose:
                print(f"  map {i+1}/{len(ensemble.maps)}  searching ...", end=" ", flush=True)
            aligned_density = _align_to_reference(result.density, reference, angle_step)
            if verbose:
                print(f"NCC = {_ncc(aligned_density, reference):.4f}")

        aligned_maps.append(dataclasses.replace(result, density=aligned_density))

    stack = np.stack([r.density for r in aligned_maps], axis=0)   # (n_maps, N, N, N)
    ref = ensemble.maps[best_idx]
    return AveragedDensityResult(
        density=stack.mean(axis=0),
        std=stack.std(axis=0),
        aligned_maps=aligned_maps,
        x_grid=ref.x_grid,
        dr=ref.dr,
        Dmax=ref.Dmax,
    )
