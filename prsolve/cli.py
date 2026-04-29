"""Command-line interface for prsolve P(r) reconstruction."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .core import _gaussian_icf, _saxs_kernel, sample_pr, solve_pr


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_saxs_data(
    path: str, skip_rows: int | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load SAXS data file (q, I, sigma columns).

    Parses line-by-line: skips non-numeric header lines before the data block
    and stops at the first non-numeric line after data has started (footer).
    If skip_rows is given, that many lines are forcibly skipped first.
    """
    rows: list[tuple[float, float, float]] = []
    with open(path) as fh:
        for _ in range(skip_rows or 0):
            fh.readline()
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
                except ValueError:
                    if rows:
                        break  # footer reached; stop
                    # else: still in header; keep scanning
            else:
                if rows:
                    break  # footer
    if not rows:
        raise ValueError(f"No numeric data found in {path}")
    data = np.array(rows)
    return data[:, 0], data[:, 1], data[:, 2]


# ---------------------------------------------------------------------------
# Guinier fitting
# ---------------------------------------------------------------------------


_GUINIER_N_ABS_MIN = 4  # floor for WLS numerical stability


def _guinier_fit(
    q: np.ndarray,
    I_obs: np.ndarray,
    sigma: np.ndarray,
    qRg_max: float = 1.3,
    qRg_span_min: float = 0.5,
    snr_min: float = 3.0,
    r2_min: float = 0.98,
) -> dict:
    """Robustly extract the Guinier region and fit I₀ and Rg.

    Expands the window from the lowest-q points outward, stopping when
    q·Rg > qRg_max or weighted R² drops below r2_min.  A window is only
    accepted when it spans ≥ qRg_span_min in q·Rg units, making the
    criterion sampling-independent.  Returns the largest accepted window.

    Parameters
    ----------
    q, I_obs, sigma : filtered data arrays (nm⁻¹ internally)
    qRg_max       : upper q·Rg limit for the Guinier approximation (default 1.3)
    qRg_span_min  : minimum (q_max - q_min)·Rg for a window to be accepted (default 0.5)
    snr_min       : minimum I/sigma for a point to enter the fit
    r2_min        : minimum weighted R² to accept a window

    Returns
    -------
    dict with keys: I0, Rg, q_min, q_max, n_pts, r2, a, b,
    q2_full, lnI_full, dlnI_full (arrays over the S/N-filtered points)
    """
    snr_mask = (I_obs > 0) & (sigma > 0) & (I_obs / sigma >= snr_min) & (q > 0)
    q_f, I_f, s_f = q[snr_mask], I_obs[snr_mask], sigma[snr_mask]

    if len(q_f) < _GUINIER_N_ABS_MIN:
        raise ValueError(
            f"Only {len(q_f)} points with S/N ≥ {snr_min}; need at least {_GUINIER_N_ABS_MIN}."
        )

    ln_I = np.log(I_f)
    q2 = q_f**2
    w = (I_f / s_f) ** 2  # weights ∝ (S/N)²

    def _wls(n):
        """Weighted least-squares: ln_I = a + b·q² over first n points."""
        q2n, ln_In, wn = q2[:n], ln_I[:n], w[:n]
        sw = wn.sum()
        swq = (wn * q2n).sum()
        swqq = (wn * q2n**2).sum()
        swl = (wn * ln_In).sum()
        swql = (wn * q2n * ln_In).sum()
        det = sw * swqq - swq**2
        if abs(det) < 1e-60:
            return None, None, 0.0
        a = (swqq * swl - swq * swql) / det
        b = (sw * swql - swq * swl) / det
        ln_I_mean = swl / sw
        ss_tot = (wn * (ln_In - ln_I_mean) ** 2).sum()
        ss_res = (wn * (ln_In - (a + b * q2n)) ** 2).sum()
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 0.0
        return a, b, r2

    best: dict = {}       # largest window satisfying qRg_span_min and r2_min
    any_valid: dict = {}  # best window seen regardless of r2/span (for fallback)
    was_above_r2_min = False

    for n in range(_GUINIER_N_ABS_MIN, len(q_f) + 1):
        a, b, r2 = _wls(n)
        if a is None or b is None or b >= 0.0:
            # Transient unphysical slope (common in noisy low-q data); skip, don't break.
            continue
        Rg = float(np.sqrt(-3.0 * b))
        if q_f[n - 1] * Rg > qRg_max:
            break  # exceeded Guinier limit; no point expanding further
        qRg_span_now = (q_f[n - 1] - q_f[0]) * Rg
        candidate = {"n": n, "Rg": Rg, "I0": float(np.exp(a)), "r2": r2, "a": a, "b": b}
        if r2 >= r2_min:
            was_above_r2_min = True
            any_valid = candidate
            if qRg_span_now >= qRg_span_min:
                best = candidate
        elif was_above_r2_min:
            # R² has fallen back below threshold after being good; stop expanding.
            break
        else:
            # Pre-convergence: track highest-R² window seen so far for fallback.
            if r2 > any_valid.get("r2", -1.0):
                any_valid = candidate

    if not best:
        # Span/R² criterion never met; fall back to best physically valid window.
        if any_valid:
            best = any_valid
        else:
            raise ValueError("Guinier fit failed: no valid linear region found.")

    n_used = best["n"]
    return {
        "I0": best["I0"],
        "Rg": best["Rg"],
        "q_min": float(q_f[0]),
        "q_max": float(q_f[n_used - 1]),
        "n_pts": n_used,
        "r2": best["r2"],
        "a": best["a"],
        "b": best["b"],
        # Arrays over all S/N-filtered points (for output file)
        "q2_full": q2,
        "lnI_full": ln_I,
        "dlnI_full": s_f / I_f,  # propagated δ(ln I) = σ/I
        "n_snr": len(q_f),
    }


# ---------------------------------------------------------------------------
# File writers (shared between full and guinier-only modes)
# ---------------------------------------------------------------------------


def _write_guinier_file(path: str, gfit: dict, q_to_out: float, q_unit: str) -> None:
    """Write the Guinier plot data file."""
    q2_out = gfit["q2_full"] * (q_to_out**2)
    b_out = gfit["b"] / (q_to_out**2)
    lnI_fit = gfit["a"] + b_out * q2_out
    n_used = gfit["n_pts"]
    in_range = np.zeros(len(q2_out), dtype=int)
    in_range[:n_used] = 1
    np.savetxt(
        path,
        np.column_stack([q2_out, gfit["lnI_full"], gfit["dlnI_full"], lnI_fit, in_range]),
        header=(
            f"q2 [{q_unit}^2]  ln(I)  delta_ln(I)  ln(I)_guinier_fit"
            f"  in_guinier_range(1=yes)"
        ),
        fmt=["%.6e", "%.6e", "%.6e", "%.6e", "%d"],
    )


# ---------------------------------------------------------------------------
# Quantities derived from P(r)
# ---------------------------------------------------------------------------


def _rg_from_pr(r: np.ndarray, pr: np.ndarray) -> float:
    """Rg² = ∫r² P(r) dr / (2 ∫P(r) dr)."""
    norm = np.trapezoid(pr, r)
    if norm <= 0:
        return float("nan")
    return float(np.sqrt(np.trapezoid(r**2 * pr, r) / (2.0 * norm)))


def _I0_from_pr(r: np.ndarray, pr: np.ndarray) -> float:
    """I(0) = 4π ∫P(r) dr."""
    return float(4.0 * np.pi * np.trapezoid(pr, r))


def _iq_from_pr(q_fine: np.ndarray, r: np.ndarray, pr: np.ndarray) -> np.ndarray:
    """I(q) = 4π ∫P(r) sin(qr)/(qr) dr on an arbitrary q grid."""
    qr = np.outer(q_fine, r)
    safe_qr = np.where(qr > 1e-12, qr, 1.0)
    sinc = np.where(qr > 1e-12, np.sin(safe_qr) / safe_qr, 1.0)
    return 4.0 * np.pi * np.trapezoid(pr[None, :] * sinc, r, axis=1)


def _vc_mw_from_iq(
    q_nm: np.ndarray, Iq: np.ndarray, Rg_nm: float
) -> dict:
    """Rambo & Tainer (2013) volume of correlation and MW estimate (proteins).

    Vc = I(0) / ∫ q·I(q) dq     [nm²  when q in nm⁻¹]
    Qr = Vc² / Rg                [nm³]
    MW = Qr × 1000 / 0.1231      [kDa; c_p = 0.1231 kDa/Å³]

    q_nm must start at or very near 0 so that Iq[0] ≈ I(0).
    MW estimate is valid for proteins only (not RNA/nucleic acids).
    """
    I0 = float(Iq[0])
    denom = float(np.trapezoid(q_nm * Iq, q_nm))
    if denom <= 0 or Rg_nm <= 0 or I0 <= 0:
        nan = float("nan")
        return {"Vc_nm2": nan, "Qr_nm3": nan, "MW_kDa": nan}
    Vc_nm2 = I0 / denom
    Qr_nm3 = Vc_nm2 ** 2 / Rg_nm
    MW_kDa = Qr_nm3 / 0.1231  # c_p = 0.1231 Å³/Da = 0.1231 nm³/kDa
    return {"Vc_nm2": Vc_nm2, "Qr_nm3": Qr_nm3, "MW_kDa": MW_kDa}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="prsolve",
        description="MaxEnt P(r) reconstruction from SAXS data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("datafile", help="SAXS data file (q, I, sigma columns)")
    parser.add_argument(
        "--units",
        required=True,
        choices=["nm", "ang"],
        help="q-vector units: 'nm' (nm⁻¹) or 'ang' (Å⁻¹)",
    )
    parser.add_argument(
        "--dmax",
        type=float,
        default=None,
        help="Maximum particle dimension (same units as 1/q); required for P(r) solve",
    )
    parser.add_argument(
        "--guinier-only",
        action="store_true",
        help="Run Guinier analysis only and exit (no P(r) reconstruction)",
    )
    parser.add_argument("--q-min", type=float, default=0.0, help="Lower q cutoff")
    parser.add_argument("--q-max", type=float, default=None, help="Upper q cutoff")
    parser.add_argument("--n-r", type=int, default=60, help="r-grid points")
    parser.add_argument(
        "--skip-rows",
        type=int,
        default=None,
        help="Header rows to skip (auto-detect if omitted)",
    )
    parser.add_argument(
        "--snr", type=float, default=1.0, help="Minimum S/N for data inclusion"
    )
    parser.add_argument(
        "--guinier-snr",
        type=float,
        default=3.0,
        help="Minimum S/N for Guinier fit points",
    )
    parser.add_argument(
        "--guinier-qrg",
        type=float,
        default=1.3,
        help="Maximum q·Rg for Guinier region",
    )
    parser.add_argument(
        "--scan-dmax",
        nargs=2,
        type=float,
        metavar=("DMIN", "DMAX"),
        help="Scan log-evidence over Dmax from DMIN to DMAX (same units as --dmax)",
    )
    parser.add_argument(
        "--dmax-step",
        type=float,
        default=0.5,
        help="Step size for --scan-dmax (same units as --dmax, default 0.5)",
    )
    parser.add_argument("--tol", type=float, default=0.01, help="Convergence tolerance")
    parser.add_argument("--max-iter", type=int, default=100, help="Maximum iterations")
    parser.add_argument(
        "--n-samples", type=int, default=300, help="Posterior samples for uncertainty band"
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Output file prefix (default: stem of input filename)",
    )
    parser.add_argument(
        "--icf-width",
        type=float,
        default=None,
        help="Override Gaussian ICF width (in the same units as 1/q → r)",
    )
    args = parser.parse_args()

    if not args.guinier_only and args.scan_dmax is None and args.dmax is None:
        parser.error(
            "--dmax is required for P(r) reconstruction "
            "(or use --guinier-only / --scan-dmax DMIN DMAX)"
        )

    stem = Path(args.datafile).stem
    prefix = args.prefix or stem

    # ---- Unit conversion factors (everything internal is nm) ----
    if args.units == "ang":
        q_to_nm = 10.0      # Å⁻¹ → nm⁻¹
        r_to_out = 10.0     # nm  → Å  (for output)
        q_to_out = 0.1      # nm⁻¹ → Å⁻¹ (for output)
        q_unit = "Å⁻¹"
        r_unit = "Å"
        dmax_nm = args.dmax / 10.0 if args.dmax is not None else None
    else:
        q_to_nm = 1.0
        r_to_out = 1.0
        q_to_out = 1.0
        q_unit = "nm⁻¹"
        r_unit = "nm"
        dmax_nm = args.dmax  # may be None in --guinier-only mode

    # ---- Load and filter data ----
    print(f"Loading {args.datafile} ...", end=" ", flush=True)
    q_raw, I_raw, s_raw = _load_saxs_data(args.datafile, args.skip_rows)
    print(f"{len(q_raw)} rows")

    q_nm = q_raw * q_to_nm
    q_min_nm = args.q_min * q_to_nm
    q_max_nm = args.q_max * q_to_nm if args.q_max is not None else np.inf

    filt = (
        (s_raw > 0)
        & (I_raw > 0)
        & (I_raw / s_raw >= args.snr)
        & (q_nm >= q_min_nm)
        & (q_nm <= q_max_nm)
    )
    q, I_obs, sigma = q_nm[filt], I_raw[filt], s_raw[filt]
    N = len(q)
    print(
        f"Data points used: {N},  q range: "
        f"{q[0]*q_to_out:.4f}–{q[-1]*q_to_out:.4f} {q_unit}"
    )

    # ---- Guinier fit ----
    print("\nFitting Guinier region...", end=" ", flush=True)
    gfit: dict | None = None
    try:
        gfit = _guinier_fit(q, I_obs, sigma, qRg_max=args.guinier_qrg, snr_min=args.guinier_snr)
        Rg_g = gfit["Rg"] * r_to_out
        print("OK")
        print(f"  Rg       = {Rg_g:.4f} {r_unit}")
        print(f"  I(0)     = {gfit['I0']:.5g}")
        print(
            f"  q range  : {gfit['q_min']*q_to_out:.4f}–{gfit['q_max']*q_to_out:.4f}"
            f" {q_unit}  ({gfit['n_pts']} pts of {gfit['n_snr']} with S/N ≥ {args.guinier_snr:.1f})"
        )
        print(f"  q·Rg max : {gfit['q_max'] * gfit['Rg']:.4f}  (limit {args.guinier_qrg})")
        print(f"  R²       = {gfit['r2']:.6f}")
    except ValueError as exc:
        print(f"WARNING: {exc}")

    # ---- Guinier-only mode: write data file and exit ----
    if args.guinier_only:
        if gfit is not None:
            guinier_file = f"{prefix}_guinier.dat"
            _write_guinier_file(guinier_file, gfit, q_to_out, q_unit)
            print(f"Wrote {guinier_file}")
        return

    # ---- Dmax scan mode ----
    if args.scan_dmax is not None:
        dmin_out, dlim_out = args.scan_dmax
        step_out = args.dmax_step
        # Build grid in user units, convert to nm for the solver
        dmax_grid_out = np.arange(dmin_out, dlim_out + step_out * 0.5, step_out)
        dmax_grid_nm = dmax_grid_out / r_to_out
        icf_width_nm = args.icf_width / r_to_out if args.icf_width is not None else None

        print(
            f"\nScanning Dmax {dmin_out:.2f}–{dlim_out:.2f} {r_unit}"
            f"  step {step_out:.2f}  ({len(dmax_grid_out)} points)\n"
        )
        hdr = f"{'Dmax':>8}  {'conv':>5}  {'iter':>5}  {'alpha':>10}  {'c²':>7}  {'chi²/c²+G':>10}  {'log_evid':>12}"
        print(hdr)
        print("-" * len(hdr))

        scan_rows = []
        for dm_out, dm_nm in zip(dmax_grid_out, dmax_grid_nm):
            icf = icf_width_nm if icf_width_nm is not None else np.pi / (2.0 * q[-1])
            res = solve_pr(
                q, I_obs, sigma, dm_nm,
                n_r=args.n_r,
                max_iter=args.max_iter,
                tol=args.tol,
                icf_width=icf,
            )
            chi2_sc = res.chi2 / res.c2
            print(
                f"{dm_out:8.3f}  {str(res.converged):>5}  {res.iterations:5d}  "
                f"{res.alpha:10.4g}  {res.c2:7.4f}  {chi2_sc + res.G:10.2f}  "
                f"{res.log_evidence:12.4f}"
            )
            scan_rows.append((dm_out, res.log_evidence, res.alpha, res.c2,
                              chi2_sc, res.G, int(res.converged)))

        scan_arr = np.array(scan_rows)
        best_idx = int(np.argmax(scan_arr[:, 1]))
        best_dm = scan_arr[best_idx, 0]
        best_ev = scan_arr[best_idx, 1]
        print(f"\nBest Dmax = {best_dm:.3f} {r_unit}  (log_evidence = {best_ev:.4f})")

        scan_file = f"{prefix}_dmax_scan.dat"
        np.savetxt(
            scan_file,
            scan_arr,
            header=(
                f"Dmax [{r_unit}]  log_evidence  alpha  c2  chi2_per_c2  G  converged"
            ),
            fmt=["%.4f", "%.6f", "%.6g", "%.6f", "%.4f", "%.4f", "%d"],
        )
        print(f"Wrote {scan_file}")
        return

    # ---- Solve P(r) ----
    icf_width_nm = args.icf_width / r_to_out if args.icf_width is not None else None
    print("\nSolving P(r)...", flush=True)
    result = solve_pr(
        q,
        I_obs,
        sigma,
        dmax_nm,
        n_r=args.n_r,
        max_iter=args.max_iter,
        tol=args.tol,
        icf_width=icf_width_nm,
    )
    status = "converged" if result.converged else "NOT converged"
    print(f"  {status} in {result.iterations} iterations")

    # ---- Posterior samples ----
    print(f"Drawing {args.n_samples} posterior samples...", end=" ", flush=True)
    samples = sample_pr(q, I_obs, sigma, result, n_samples=args.n_samples)
    lo, hi = np.percentile(samples, [5, 95], axis=0)
    print("done")

    # ---- Derived quantities ----
    r_out = result.r * r_to_out
    Rg_nm = _rg_from_pr(result.r, result.pr)
    Rg_pr = Rg_nm * r_to_out
    I0_pr = _I0_from_pr(result.r, result.pr)

    # Theoretical I(q) on a fine grid from q=0 to 1.05·q_max
    q_fine_nm = np.linspace(0.0, q[-1], 500)
    Iq_fine = _iq_from_pr(q_fine_nm, result.r, result.pr)
    q_fine_out = q_fine_nm * q_to_out

    # I(q) at the observed q points (for chi² check and output)
    Iq_obs = _iq_from_pr(q, result.r, result.pr)

    c2 = result.c2
    chi2_scaled = result.chi2 / c2

    # Volume of correlation / MW estimate (Rambo & Tainer 2013)
    vc = _vc_mw_from_iq(q_fine_nm, Iq_fine, Rg_nm)
    Vc_out = vc["Vc_nm2"] * (r_to_out ** 2)
    r2_unit = f"{r_unit}²"

    # Quick summary to stdout
    print(f"  Rg  [P(r)] = {Rg_pr:.4f} {r_unit}")
    print(f"  I(0)[P(r)] = {I0_pr:.5g}")
    print(f"  Vc         = {Vc_out:.4g} {r2_unit}  |  MW (protein) ≈ {vc['MW_kDa']:.1f} kDa")

    # ---- Write output files ----

    # 1. P(r) with posterior band
    pr_file = f"{prefix}_pr.dat"
    np.savetxt(
        pr_file,
        np.column_stack([r_out, result.pr, lo, hi]),
        header=f"r [{r_unit}]  P(r)  P(r)_5pct  P(r)_95pct",
        fmt="%.6e",
    )
    print(f"Wrote {pr_file}")

    # 2. Observed data
    obs_file = f"{prefix}_obs.dat"
    np.savetxt(
        obs_file,
        np.column_stack([q * q_to_out, I_obs, sigma, Iq_obs]),
        header=f"q [{q_unit}]  I_obs  sigma  I_fit",
        fmt="%.6e",
    )
    print(f"Wrote {obs_file}")

    # 3. Theoretical I(q) from P(r), q→0 extrapolation included
    iq_file = f"{prefix}_iq.dat"
    np.savetxt(
        iq_file,
        np.column_stack([q_fine_out, Iq_fine]),
        header=f"q [{q_unit}]  I_theory  (4pi * integral P(r) sinc(qr) dr, q->0 included)",
        fmt="%.6e",
    )
    print(f"Wrote {iq_file}")

    # 4. Guinier plot data (all S/N-filtered points + fit line)
    if gfit is not None:
        guinier_file = f"{prefix}_guinier.dat"
        _write_guinier_file(guinier_file, gfit, q_to_out, q_unit)
        print(f"Wrote {guinier_file}")

    # 5. Report
    report_file = f"{prefix}_report.txt"
    sep = "=" * 62
    with open(report_file, "w") as f:
        f.write(f"{sep}\n")
        f.write(f"  prsolve – P(r) reconstruction report\n")
        f.write(f"{sep}\n\n")

        f.write(f"Input file   : {args.datafile}\n")
        f.write(f"Units        : {args.units}  ({q_unit}  /  {r_unit})\n")
        f.write(
            f"Data points  : {N}   q = {q[0]*q_to_out:.4f}–{q[-1]*q_to_out:.4f} {q_unit}\n"
        )
        f.write(f"Dmax         : {args.dmax:.4f} {r_unit}\n")
        f.write(f"n_r          : {args.n_r}\n")
        f.write(
            f"ICF width    : {result.icf_width*r_to_out:.4f} {r_unit}"
            f"  (= π / 2q_max)\n\n"
        )

        f.write("--- Guinier fit ---\n")
        if gfit is not None:
            f.write(f"  Rg         : {Rg_g:.4f} {r_unit}\n")
            f.write(f"  I(0)       : {gfit['I0']:.6g}\n")
            f.write(
                f"  q range    : {gfit['q_min']*q_to_out:.4f}–"
                f"{gfit['q_max']*q_to_out:.4f} {q_unit}"
                f"  ({gfit['n_pts']} pts)\n"
            )
            f.write(
                f"  q·Rg max   : {gfit['q_max'] * gfit['Rg']:.4f}"
                f"  (limit {args.guinier_qrg})\n"
            )
            f.write(f"  R²         : {gfit['r2']:.6f}\n\n")
        else:
            f.write("  Guinier fit failed (see console warning).\n\n")

        f.write("--- MaxEnt P(r) solution ---\n")
        f.write(f"  Converged  : {result.converged}  ({result.iterations} iters)\n")
        f.write(f"  alpha      : {result.alpha:.5g}\n")
        f.write(f"  c²         : {c2:.5f}   (σ_eff = {c2**0.5:.4f}·σ)\n")
        f.write(f"  χ²  (raw)  : {result.chi2:.3f}\n")
        f.write(f"  χ²/c²      : {chi2_scaled:.3f}   (≈ N−G = {N - result.G:.1f})\n")
        f.write(f"  χ²/c² + G  : {chi2_scaled + result.G:.1f}   (≈ N = {N})\n")
        f.write(f"  G          : {result.G:.1f}  (good measurements)\n")
        f.write(f"  log evid   : {result.log_evidence:.4f}\n\n")

        f.write("--- Derived from P(r) ---\n")
        f.write(f"  Rg  [P(r)] : {Rg_pr:.4f} {r_unit}\n")
        f.write(f"  I(0)[P(r)] : {I0_pr:.6g}   (= 4π ∫P(r) dr)\n")
        f.write(f"  Vc         : {Vc_out:.4g} {r2_unit}  (volume of correlation)\n")
        f.write(f"  MW [prot]  : {vc['MW_kDa']:.1f} kDa  (Rambo & Tainer 2013; relative scale)\n\n")

        f.write("--- Output files ---\n")
        w = max(len(pr_file), len(obs_file), len(iq_file)) + 2
        f.write(f"  {pr_file:<{w}} r, P(r), 5%, 95% posterior band\n")
        f.write(f"  {obs_file:<{w}} q, I_obs, sigma, I_fit (at observed q)\n")
        f.write(f"  {iq_file:<{w}} q, I_theory from P(r) incl. q→0\n")
        if gfit is not None:
            f.write(f"  {guinier_file:<{w}} q², ln I, Guinier fit\n")
        f.write(f"  {report_file:<{w}} this report\n")

    print(f"Wrote {report_file}")
    print()
    # Echo the report to stdout
    with open(report_file) as f:
        print(f.read(), end="")


if __name__ == "__main__":
    main()
