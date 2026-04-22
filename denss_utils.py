"""
denss_utils.py — Shared utilities for SAXS 3D density reconstruction.

Provides the result dataclass, data subsampling, and grid/P(r) helpers used
by the projection-based phase retrieval solvers (e.g. denss_raar.py).

Units
-----
q and Dmax must be in consistent units (e.g. both nm⁻¹ / nm).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "DensityResult",
    "EnsembleDensityResult",
    "AveragedDensityResult",
    "density_to_pr",
    "subsample_q",
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class DensityResult:
    """Output of a density reconstruction solver.

    Attributes
    ----------
    density      : (N, N, N) electron density on a Cartesian voxel grid.
    x_grid       : (N,) voxel-centre coordinates (same units as 1/q).
    dr           : voxel edge length.
    Dmax         : maximum particle dimension (caller-supplied).
    chi2         : reduced χ² at convergence (ideally ≈ 1).
    scale        : fitted overall intensity scale factor.
    I_pred       : (n_q,) predicted profile (scale applied), for plotting.
    loss_history : (n_checkpoints,) χ² sampled during this restart.
    """
    density: np.ndarray
    x_grid: np.ndarray
    dr: float
    Dmax: float
    chi2: float
    scale: float
    I_pred: np.ndarray
    loss_history: np.ndarray


@dataclass
class EnsembleDensityResult:
    """All density maps produced by a multi-restart phase retrieval run.

    Attributes
    ----------
    maps : list of DensityResult, one per restart, each COM-centred.
           Use :attr:`best` to access the lowest-χ² map directly.
    """
    maps: list[DensityResult]

    @property
    def best(self) -> DensityResult:
        """DensityResult with the lowest χ²."""
        return min(self.maps, key=lambda r: r.chi2)


@dataclass
class AveragedDensityResult:
    """Output of :func:`align_and_average`.

    Attributes
    ----------
    density      : (N, N, N) voxel-wise mean over all aligned maps.
    std          : (N, N, N) voxel-wise standard deviation.
    aligned_maps : DensityResult list; densities have been rotated/reflected
                   into the reference frame of the best-χ² map.
    x_grid       : (N,) voxel-centre coordinates (same units as 1/q).
    dr           : voxel edge length.
    Dmax         : maximum particle dimension.
    """
    density: np.ndarray
    std: np.ndarray
    aligned_maps: list[DensityResult]
    x_grid: np.ndarray
    dr: float
    Dmax: float


# ---------------------------------------------------------------------------
# Data subsampling
# ---------------------------------------------------------------------------

def subsample_q(
    q: np.ndarray,
    I_obs: np.ndarray,
    sigma: np.ndarray,
    n_points: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Logarithmically subsample SAXS data for 3D density reconstruction.

    Dense experimental datasets (often 1000+ points) oversaturate the
    shell-averaging matrix relative to what the FFT grid can resolve.
    This function bins the data onto ``n_points`` log-spaced q values,
    propagating errors as 1/sqrt(N) within each bin.

    Parameters
    ----------
    q, I_obs, sigma : input data arrays (same length, q sorted ascending).
    n_points        : target number of output q points.

    Returns
    -------
    q_s, I_s, sigma_s : subsampled arrays of length ≤ n_points.
    """
    q_edges = np.logspace(np.log10(q.min()), np.log10(q.max()), n_points + 1)
    q_out, I_out, sig_out = [], [], []

    for lo, hi in zip(q_edges[:-1], q_edges[1:]):
        mask = (q >= lo) & (q < hi)
        if not mask.any():
            continue
        w = 1.0 / sigma[mask]**2
        I_w = (w * I_obs[mask]).sum() / w.sum()     # inverse-variance weighted mean
        sig_w = 1.0 / np.sqrt(w.sum())              # propagated uncertainty
        q_w = (w * q[mask]).sum() / w.sum()
        q_out.append(q_w)
        I_out.append(I_w)
        sig_out.append(sig_w)

    return np.array(q_out), np.array(I_out), np.array(sig_out)


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def _nyquist_grid_size(Dmax: float, q_max: float, oversampling: float) -> int:
    """Minimum N such that the FFT Nyquist frequency covers q_max.

    Real-space box = Dmax × oversampling  →  dr = box / N
    q_max_FFT = π / dr = π N / box  →  N ≥ box · q_max / π
    """
    return int(math.ceil(Dmax * oversampling * q_max / math.pi))


# ---------------------------------------------------------------------------
# Centre-of-mass shift (applied once on the final result)
# ---------------------------------------------------------------------------

def _center_by_com(density: np.ndarray, x: np.ndarray) -> np.ndarray:
    total = density.sum()
    if total == 0.0:
        return density
    dr = x[1] - x[0]
    cx = (density * x[:, None, None]).sum() / total
    cy = (density * x[None, :, None]).sum() / total
    cz = (density * x[None, None, :]).sum() / total
    shift = (-int(round(cx / dr)), -int(round(cy / dr)), -int(round(cz / dr)))
    return np.roll(density, shift, axis=(0, 1, 2))


# ---------------------------------------------------------------------------
# P(r) from 3D density (Wiener–Khinchin)
# ---------------------------------------------------------------------------

def density_to_pr(result: DensityResult, n_r: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Compute P(r) from a 3D density map via the autocorrelation theorem.

    Uses the Wiener–Khinchin identity:

        P(r) ∝ r² × radial_average[ IFFT(|FFT(ρ)|²) ]

    which avoids the O(N⁶) pairwise-distance loop.

    Parameters
    ----------
    result : DensityResult from a density solver.
    n_r    : number of radial bins.

    Returns
    -------
    r  : (n_r,) distance values in the same units as result.x_grid.
    pr : (n_r,) P(r), normalised to peak = 1 and clipped at 0.
    """
    rho = result.density
    N   = rho.shape[0]
    dr  = result.dr

    F_q      = np.fft.fftn(rho)
    autocorr = np.real(np.fft.ifftn(np.abs(F_q)**2))
    autocorr = np.fft.fftshift(autocorr)                  # centre at origin

    x = (np.arange(N) - (N - 1) / 2.0) * dr
    xx, yy, zz = np.meshgrid(x, x, x, indexing='ij')
    r_mag = np.sqrt(xx**2 + yy**2 + zz**2).ravel()

    r_edges = np.linspace(0.0, result.Dmax, n_r + 1)
    r_mids  = 0.5 * (r_edges[:-1] + r_edges[1:])

    acorr_flat  = autocorr.ravel()
    sum_ac, _   = np.histogram(r_mag, bins=r_edges, weights=acorr_flat)
    count_ac, _ = np.histogram(r_mag, bins=r_edges)
    count_ac    = np.maximum(count_ac, 1)

    pr = (sum_ac / count_ac) * r_mids**2
    pr = pr.clip(0)
    if pr.max() > 0:
        pr /= pr.max()

    return r_mids, pr
