"""PyQt/PySide GUI for prsolve – SAXS P(r) reconstruction."""
from __future__ import annotations

import traceback

import numpy as np
from qtpy.QtCore import QObject, QThread, Qt, Signal
from qtpy.QtGui import QFont
from qtpy.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QMessageBox, QPushButton, QScrollArea,
    QSpinBox, QSplitter, QTabWidget, QTextEdit,
    QVBoxLayout, QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from .cli import (
    _load_saxs_data, _guinier_fit, _rg_from_pr, _I0_from_pr, _iq_from_pr,
    _vc_mw_from_iq, _write_guinier_file,
)
from .core import solve_pr, sample_pr


# ---------------------------------------------------------------------------
# Matplotlib canvas helper
# ---------------------------------------------------------------------------

class MplCanvas(QWidget):
    """Matplotlib Figure + NavigationToolbar wrapped in a QWidget."""

    def __init__(self, nrows: int = 1, ncols: int = 1, figsize=(6, 4), parent=None):
        super().__init__(parent)
        self.fig = Figure(figsize=figsize, layout="constrained")
        if nrows == 1 and ncols == 1:
            self._ax = self.fig.add_subplot(111)
            self._axes_arr = None
        else:
            self._axes_arr = self.fig.subplots(nrows, ncols)
            self._ax = None
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas)

    @property
    def ax(self):
        return self._ax

    @property
    def axes(self):
        return self._axes_arr

    def draw(self):
        self.canvas.draw_idle()


# ---------------------------------------------------------------------------
# Worker threads
# ---------------------------------------------------------------------------

class _BaseWorker(QObject):
    done = Signal(dict)
    error = Signal(str)


class LoadWorker(_BaseWorker):
    def __init__(self, params: dict):
        super().__init__()
        self.p = params

    def run(self):
        try:
            p = self.p
            q_raw, I_raw, s_raw = _load_saxs_data(p["path"], p["skip_rows"])
            q_to_nm = 10.0 if p["units"] == "ang" else 1.0
            q_to_out = 0.1 if p["units"] == "ang" else 1.0
            r_to_out = 10.0 if p["units"] == "ang" else 1.0
            q_unit = "Å⁻¹" if p["units"] == "ang" else "nm⁻¹"
            r_unit = "Å" if p["units"] == "ang" else "nm"
            self.done.emit({
                "q_raw": q_raw * q_to_nm,  # always stored in nm⁻¹
                "I_raw": I_raw,
                "s_raw": s_raw,
                "q_to_out": q_to_out, "r_to_out": r_to_out,
                "q_unit": q_unit, "r_unit": r_unit,
                "units": p["units"],
            })
        except Exception:
            self.error.emit(traceback.format_exc())


class ScanWorker(_BaseWorker):
    progress = Signal(str)

    def __init__(self, params: dict):
        super().__init__()
        self.p = params

    def run(self):
        try:
            p = self.p
            q, I_obs, sigma = p["q"], p["I_obs"], p["sigma"]
            rows = []
            for dm_nm in p["dmax_grid_nm"]:
                res = solve_pr(
                    q, I_obs, sigma, dm_nm,
                    n_r=p["n_r"], max_iter=p["max_iter"], tol=p["tol"],
                    icf_width=p["icf_nm"],
                )
                dm_out = dm_nm * p["r_to_out"]
                chi2_sc = res.chi2 / res.c2
                rows.append((dm_out, res.log_evidence, res.alpha, res.c2,
                             chi2_sc, res.G, int(res.converged)))
                self.progress.emit(
                    f"Dmax {dm_out:.2f} {p['r_unit']}  →  log_evid = {res.log_evidence:.4f}"
                )
            arr = np.array(rows)
            best_idx = int(np.argmax(arr[:, 1]))
            self.done.emit({
                "scan": arr, "best_idx": best_idx,
                "r_unit": p["r_unit"], "r_to_out": p["r_to_out"],
            })
        except Exception:
            self.error.emit(traceback.format_exc())


class IcfScanWorker(_BaseWorker):
    progress = Signal(str)

    def __init__(self, params: dict):
        super().__init__()
        self.p = params

    def run(self):
        try:
            p = self.p
            q, I_obs, sigma = p["q"], p["I_obs"], p["sigma"]
            rows = []
            for w_nm in p["w_grid_nm"]:
                res = solve_pr(
                    q, I_obs, sigma, p["dmax_nm"],
                    n_r=p["n_r"], max_iter=p["max_iter"], tol=p["tol"],
                    icf_width=w_nm,
                )
                w_out = w_nm * p["r_to_out"]
                rows.append([w_out, res.log_evidence, float(res.converged),
                              res.G, res.alpha, float(res.iterations)])
                self.progress.emit(
                    f"ICF w = {w_out:.3f} {p['r_unit']}  →  "
                    f"log_evid = {res.log_evidence:.4f}  "
                    f"({'ok' if res.converged else 'NOT converged'})"
                )
            arr = np.array(rows)
            conv = arr[:, 2] > 0.5
            if conv.any():
                best_idx = int(np.where(conv)[0][np.argmax(arr[conv, 1])])
            else:
                best_idx = int(np.argmax(arr[:, 1]))
            self.done.emit({
                "scan": arr, "best_idx": best_idx,
                "r_unit": p["r_unit"], "r_to_out": p["r_to_out"],
            })
        except Exception:
            self.error.emit(traceback.format_exc())


class SolveWorker(_BaseWorker):
    progress = Signal(str)

    def __init__(self, params: dict):
        super().__init__()
        self.p = params

    def run(self):
        try:
            p = self.p
            self.progress.emit("Solving P(r) via MaxEnt…")
            result = solve_pr(
                p["q"], p["I_obs"], p["sigma"], p["dmax_nm"],
                n_r=p["n_r"], max_iter=p["max_iter"], tol=p["tol"],
                icf_width=p["icf_nm"], debug=True,
            )
            self.progress.emit(f"Drawing {p['n_samples']} posterior samples…")
            samples = sample_pr(
                p["q"], p["I_obs"], p["sigma"], result, n_samples=p["n_samples"]
            )
            lo, hi = np.percentile(samples, [5, 95], axis=0)

            q = p["q"]
            q_fine_nm = np.linspace(0.0, q[-1], 500)
            Iq_fine = _iq_from_pr(q_fine_nm, result.r, result.pr)
            Iq_obs_fit = _iq_from_pr(q, result.r, result.pr)

            r_to_out, q_to_out = p["r_to_out"], p["q_to_out"]
            Rg_nm = _rg_from_pr(result.r, result.pr)
            vc = _vc_mw_from_iq(q_fine_nm, Iq_fine, Rg_nm)
            self.done.emit({
                "result": result,
                "trace": result.trace,
                "lo": lo, "hi": hi,
                "q_fine_nm": q_fine_nm,
                "q_fine": q_fine_nm * q_to_out,
                "Iq_fine": Iq_fine,
                "Iq_obs_fit": Iq_obs_fit,
                "Rg_pr": Rg_nm * r_to_out,
                "I0_pr": _I0_from_pr(result.r, result.pr),
                "vc": vc,
                "r_to_out": r_to_out, "q_to_out": q_to_out,
                "r_unit": p["r_unit"], "q_unit": p["q_unit"],
            })
        except Exception:
            self.error.emit(traceback.format_exc())


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("prsolve – SAXS P(r) Reconstruction")
        self.resize(1260, 820)

        self._data: dict | None = None
        self._pr_out: dict | None = None
        self._scan_out: dict | None = None
        self._icfscan_out: dict | None = None
        self._thread: QThread | None = None
        self._worker: QObject | None = None

        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)
        splitter.addWidget(self._build_controls())
        splitter.addWidget(self._build_tabs())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([300, 960])

        self.statusBar().showMessage("Ready – open a SAXS data file to begin.")

    # ------------------------------------------------------------------
    # Control panel
    # ------------------------------------------------------------------

    def _build_controls(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFixedWidth(300)

        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setSpacing(6)
        lay.setContentsMargins(8, 8, 8, 8)

        # ---- Data file ----
        grp = QGroupBox("Data File")
        vl = QVBoxLayout(grp)

        row = QHBoxLayout()
        self.le_path = QLineEdit()
        self.le_path.setPlaceholderText("path/to/data.dat")
        btn_br = QPushButton("Browse…")
        btn_br.setFixedWidth(72)
        btn_br.clicked.connect(self._browse)
        row.addWidget(self.le_path)
        row.addWidget(btn_br)
        vl.addLayout(row)

        fm = QFormLayout()
        fm.setSpacing(4)

        self.cb_units = QComboBox()
        self.cb_units.addItems(["Å⁻¹  (ang)", "nm⁻¹  (nm)"])
        fm.addRow("q units:", self.cb_units)

        self.sp_skip = QSpinBox()
        self.sp_skip.setRange(0, 9999)
        self.sp_skip.setSpecialValueText("auto")
        fm.addRow("Skip rows:", self.sp_skip)

        self.sp_qmin = QDoubleSpinBox()
        self.sp_qmin.setRange(0.0, 999.0)
        self.sp_qmin.setDecimals(4)
        self.sp_qmin.setSingleStep(0.001)
        fm.addRow("q min:", self.sp_qmin)

        self.sp_qmax = QDoubleSpinBox()
        self.sp_qmax.setRange(0.0, 999.0)
        self.sp_qmax.setDecimals(4)
        self.sp_qmax.setSingleStep(0.001)
        self.sp_qmax.setSpecialValueText("auto")
        fm.addRow("q max:", self.sp_qmax)

        self.sp_snr = QDoubleSpinBox()
        self.sp_snr.setRange(0.0, 99.0)
        self.sp_snr.setDecimals(1)
        self.sp_snr.setValue(1.0)
        fm.addRow("S/N min:", self.sp_snr)

        vl.addLayout(fm)
        lay.addWidget(grp)

        self.btn_load = QPushButton("Load Data")
        self.btn_load.setFixedHeight(34)
        self.btn_load.clicked.connect(self._do_load)
        lay.addWidget(self.btn_load)

        lay.addSpacing(4)

        # ---- Guinier ----
        grp2 = QGroupBox("Guinier Analysis")
        gm = QFormLayout(grp2)
        gm.setSpacing(4)

        self.sp_g_snr = QDoubleSpinBox()
        self.sp_g_snr.setRange(0.0, 99.0)
        self.sp_g_snr.setDecimals(1)
        self.sp_g_snr.setValue(3.0)
        gm.addRow("S/N min:", self.sp_g_snr)

        self.sp_g_qrg = QDoubleSpinBox()
        self.sp_g_qrg.setRange(0.1, 5.0)
        self.sp_g_qrg.setDecimals(2)
        self.sp_g_qrg.setValue(1.3)
        gm.addRow("q·Rg max:", self.sp_g_qrg)

        lay.addWidget(grp2)

        self.btn_guinier = QPushButton("Run Guinier")
        self.btn_guinier.setFixedHeight(34)
        self.btn_guinier.setEnabled(False)
        self.btn_guinier.clicked.connect(self._do_guinier)
        lay.addWidget(self.btn_guinier)

        lay.addSpacing(4)

        # ---- Dmax scan ----
        grp3 = QGroupBox("Dmax Scan")
        sm = QFormLayout(grp3)
        sm.setSpacing(4)

        self.sp_dmin = QDoubleSpinBox()
        self.sp_dmin.setRange(0.1, 9999.0)
        self.sp_dmin.setDecimals(1)
        self.sp_dmin.setValue(30.0)
        sm.addRow("D min:", self.sp_dmin)

        self.sp_dlim = QDoubleSpinBox()
        self.sp_dlim.setRange(0.1, 9999.0)
        self.sp_dlim.setDecimals(1)
        self.sp_dlim.setValue(150.0)
        sm.addRow("D max:", self.sp_dlim)

        self.sp_dstep = QDoubleSpinBox()
        self.sp_dstep.setRange(0.01, 100.0)
        self.sp_dstep.setDecimals(2)
        self.sp_dstep.setValue(1.0)
        sm.addRow("Step:", self.sp_dstep)

        lay.addWidget(grp3)

        self.btn_scan = QPushButton("Scan Dmax")
        self.btn_scan.setFixedHeight(34)
        self.btn_scan.setEnabled(False)
        self.btn_scan.clicked.connect(self._do_scan)
        lay.addWidget(self.btn_scan)

        lay.addSpacing(4)

        # ---- ICF width scan ----
        grp_icf = QGroupBox("ICF Width Scan")
        im = QFormLayout(grp_icf)
        im.setSpacing(4)

        self.sp_wmin = QDoubleSpinBox()
        self.sp_wmin.setRange(0.001, 9999.0)
        self.sp_wmin.setDecimals(3)
        self.sp_wmin.setValue(0.1)
        im.addRow("w min:", self.sp_wmin)

        self.sp_wmax = QDoubleSpinBox()
        self.sp_wmax.setRange(0.001, 9999.0)
        self.sp_wmax.setDecimals(3)
        self.sp_wmax.setValue(2.0)
        im.addRow("w max:", self.sp_wmax)

        self.sp_nw = QSpinBox()
        self.sp_nw.setRange(3, 100)
        self.sp_nw.setValue(20)
        im.addRow("N pts:", self.sp_nw)

        lay.addWidget(grp_icf)

        self.btn_icfscan = QPushButton("Scan ICF Width")
        self.btn_icfscan.setFixedHeight(34)
        self.btn_icfscan.setEnabled(False)
        self.btn_icfscan.clicked.connect(self._do_icfscan)
        lay.addWidget(self.btn_icfscan)

        lay.addSpacing(4)

        # ---- P(r) solution ----
        grp4 = QGroupBox("P(r) Solution")
        pm = QFormLayout(grp4)
        pm.setSpacing(4)

        self.sp_dmax = QDoubleSpinBox()
        self.sp_dmax.setRange(0.1, 9999.0)
        self.sp_dmax.setDecimals(1)
        self.sp_dmax.setValue(100.0)
        pm.addRow("Dmax:", self.sp_dmax)

        self.sp_nr = QSpinBox()
        self.sp_nr.setRange(10, 500)
        self.sp_nr.setValue(60)
        pm.addRow("Grid pts:", self.sp_nr)

        self.sp_maxiter = QSpinBox()
        self.sp_maxiter.setRange(10, 5000)
        self.sp_maxiter.setValue(100)
        pm.addRow("Max iter:", self.sp_maxiter)

        self.sp_tol = QDoubleSpinBox()
        self.sp_tol.setRange(1e-6, 1.0)
        self.sp_tol.setDecimals(4)
        self.sp_tol.setSingleStep(0.005)
        self.sp_tol.setValue(0.01)
        pm.addRow("Tolerance:", self.sp_tol)

        self.sp_samples = QSpinBox()
        self.sp_samples.setRange(0, 5000)
        self.sp_samples.setValue(300)
        pm.addRow("Samples:", self.sp_samples)

        self.sp_icf = QDoubleSpinBox()
        self.sp_icf.setRange(0.0, 9999.0)
        self.sp_icf.setDecimals(3)
        self.sp_icf.setSpecialValueText("auto  (π/2q_max)")
        self.lbl_icf = QLabel("ICF width:")
        pm.addRow(self.lbl_icf, self.sp_icf)

        lay.addWidget(grp4)

        self.btn_solve = QPushButton("Solve P(r)")
        self.btn_solve.setFixedHeight(34)
        self.btn_solve.setEnabled(False)
        self.btn_solve.clicked.connect(self._do_solve)
        lay.addWidget(self.btn_solve)

        self.btn_save = QPushButton("Save Results…")
        self.btn_save.setFixedHeight(34)
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self._do_save)
        lay.addWidget(self.btn_save)

        lay.addStretch()
        scroll.setWidget(inner)
        return scroll

    # ------------------------------------------------------------------
    # Tab widget
    # ------------------------------------------------------------------

    def _build_tabs(self) -> QTabWidget:
        tabs = QTabWidget()

        # --- Data tab: canvas + live q-range / scale controls ---
        data_wrap = QWidget()
        dw_lay = QVBoxLayout(data_wrap)
        dw_lay.setContentsMargins(0, 0, 0, 0)
        dw_lay.setSpacing(2)

        ctrl = QHBoxLayout()
        ctrl.setContentsMargins(6, 4, 6, 0)
        ctrl.addWidget(QLabel("q display:"))

        self.sp_view_qmin = QDoubleSpinBox()
        self.sp_view_qmin.setRange(0.0, 999.0)
        self.sp_view_qmin.setDecimals(4)
        self.sp_view_qmin.setSingleStep(0.001)
        self.sp_view_qmin.setFixedWidth(90)
        ctrl.addWidget(self.sp_view_qmin)

        ctrl.addWidget(QLabel("–"))

        self.sp_view_qmax = QDoubleSpinBox()
        self.sp_view_qmax.setRange(0.0, 999.0)
        self.sp_view_qmax.setDecimals(4)
        self.sp_view_qmax.setSingleStep(0.001)
        self.sp_view_qmax.setFixedWidth(90)
        ctrl.addWidget(self.sp_view_qmax)

        ctrl.addSpacing(16)

        self.cb_xscale = QCheckBox("Linear X")
        ctrl.addWidget(self.cb_xscale)
        ctrl.addStretch()

        dw_lay.addLayout(ctrl)

        self.cvs_data = MplCanvas(figsize=(7, 5))
        dw_lay.addWidget(self.cvs_data, stretch=1)

        self.sp_view_qmin.valueChanged.connect(self._apply_data_view)
        self.sp_view_qmax.valueChanged.connect(self._apply_data_view)
        self.cb_xscale.toggled.connect(self._apply_data_view)

        tabs.addTab(data_wrap, "Data")

        self.cvs_mf = MplCanvas(nrows=1, ncols=2, figsize=(12, 5))
        tabs.addTab(self.cvs_mf, "Model-free")

        self.cvs_scan = MplCanvas(figsize=(7, 5))
        tabs.addTab(self.cvs_scan, "Dmax Scan")

        self.cvs_icfscan = MplCanvas(nrows=1, ncols=2, figsize=(12, 5))
        tabs.addTab(self.cvs_icfscan, "ICF Scan")

        pr_wrap = QWidget()
        pw = QVBoxLayout(pr_wrap)
        pw.setContentsMargins(0, 0, 0, 0)
        pw.setSpacing(0)
        self.cvs_pr = MplCanvas(figsize=(7, 4))
        self.te_summary = QTextEdit()
        self.te_summary.setReadOnly(True)
        self.te_summary.setMaximumHeight(140)
        self.te_summary.setFont(QFont("Courier", 9))
        self.te_summary.setPlaceholderText("P(r) solution summary will appear here.")
        pw.addWidget(self.cvs_pr, stretch=1)
        pw.addWidget(self.te_summary)
        tabs.addTab(pr_wrap, "P(r)")

        self.cvs_diag = MplCanvas(nrows=2, ncols=2, figsize=(12, 8))
        tabs.addTab(self.cvs_diag, "Diagnostics")

        self._tabs = tabs
        return tabs

    # ------------------------------------------------------------------
    # Button slots
    # ------------------------------------------------------------------

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open SAXS data file", "",
            "Data files (*.dat *.txt *.csv *.int);;All (*.*)",
        )
        if path:
            self.le_path.setText(path)

    def _do_load(self):
        path = self.le_path.text().strip()
        if not path:
            QMessageBox.warning(self, "No file", "Please select a data file.")
            return
        units = "ang" if "ang" in self.cb_units.currentText() else "nm"
        skip = self.sp_skip.value() or None
        params = {"path": path, "units": units, "skip_rows": skip}
        self._run_worker(LoadWorker(params), self._on_load_done)

    def _do_guinier(self):
        if self._data is None:
            return
        d = self._data
        units = d["units"]
        q_to_nm = 10.0 if units == "ang" else 1.0
        q_to_out = d["q_to_out"]
        r_to_out = d["r_to_out"]

        q_raw, I_raw, s_raw = d["q_raw"], d["I_raw"], d["s_raw"]
        q_nm = q_raw  # already in nm⁻¹ (converted on load)
        q_min_nm = self.sp_qmin.value() * q_to_nm
        qmax_val = self.sp_qmax.value()
        q_max_nm = (qmax_val * q_to_nm) if qmax_val > 0 else np.inf
        snr = self.sp_snr.value()

        filt = (
            (s_raw > 0) & (I_raw > 0)
            & (I_raw / s_raw >= snr)
            & (q_nm >= q_min_nm) & (q_nm <= q_max_nm)
        )
        q = q_nm[filt]
        I_obs = I_raw[filt]
        sigma = s_raw[filt]

        if len(q) < 2:
            QMessageBox.warning(self, "No data",
                                "Filter leaves fewer than 2 points. "
                                "Adjust q min/max or S/N.")
            return

        gfit, gfit_err = None, None
        try:
            gfit = _guinier_fit(q, I_obs, sigma,
                                qRg_max=self.sp_g_qrg.value(),
                                snr_min=self.sp_g_snr.value())
        except ValueError as exc:
            gfit_err = str(exc)

        self._data.update({
            "q": q, "I_obs": I_obs, "sigma": sigma,
            "gfit": gfit, "gfit_err": gfit_err,
        })

        auto_icf = (np.pi / (2.0 * float(q.max()))) * r_to_out
        self.lbl_icf.setText(
            f"ICF width:\n(auto = {auto_icf:.3f} {d['r_unit']})")

        # Set ICF scan range: 0.25× to 4× the natural ICF width
        self.sp_wmin.setValue(round(0.25 * auto_icf, 3))
        self.sp_wmax.setValue(round(4.0  * auto_icf, 3))

        self.btn_scan.setEnabled(True)
        self.btn_solve.setEnabled(True)
        self.btn_icfscan.setEnabled(True)

        msg = (f"Filtered {int(filt.sum())} / {len(q_raw)} points, "
               f"q = {q[0]*q_to_out:.4f}–{q[-1]*q_to_out:.4f} {d['q_unit']}")
        if gfit is not None:
            msg += (f"   |   Rg = {gfit['Rg']*r_to_out:.2f} {d['r_unit']}, "
                    f"I₀ = {gfit['I0']:.4g}")
        elif gfit_err:
            msg += f"   |   Guinier failed: {gfit_err}"
        self.statusBar().showMessage(msg)

        self._refresh_data_tab()
        self._refresh_mf_tab()

    def _do_scan(self):
        if self._data is None:
            return
        d = self._data
        r_to_out = d["r_to_out"]
        step_out = self.sp_dstep.value()
        grid_out = np.arange(
            self.sp_dmin.value(),
            self.sp_dlim.value() + step_out * 0.5,
            step_out,
        )
        icf_val = self.sp_icf.value()
        params = {
            "q": d["q"], "I_obs": d["I_obs"], "sigma": d["sigma"],
            "dmax_grid_nm": grid_out / r_to_out,
            "n_r": self.sp_nr.value(),
            "max_iter": self.sp_maxiter.value(),
            "tol": self.sp_tol.value(),
            "icf_nm": (icf_val / r_to_out) if icf_val > 0 else None,
            "r_to_out": r_to_out,
            "r_unit": d["r_unit"],
        }
        self._run_worker(ScanWorker(params), self._on_scan_done)

    def _do_icfscan(self):
        if self._data is None or self._data.get("q") is None:
            return
        d = self._data
        r_to_out = d["r_to_out"]
        w_min_nm = self.sp_wmin.value() / r_to_out
        w_max_nm = self.sp_wmax.value() / r_to_out
        w_grid_nm = np.exp(
            np.linspace(np.log(w_min_nm), np.log(w_max_nm), self.sp_nw.value())
        )
        params = {
            "q": d["q"], "I_obs": d["I_obs"], "sigma": d["sigma"],
            "dmax_nm": self.sp_dmax.value() / r_to_out,
            "w_grid_nm": w_grid_nm,
            "n_r": self.sp_nr.value(),
            "max_iter": self.sp_maxiter.value(),
            "tol": self.sp_tol.value(),
            "r_to_out": r_to_out,
            "r_unit": d["r_unit"],
        }
        self._run_worker(IcfScanWorker(params), self._on_icfscan_done)

    def _do_solve(self):
        if self._data is None:
            return
        d = self._data
        r_to_out = d["r_to_out"]
        icf_val = self.sp_icf.value()
        params = {
            "q": d["q"], "I_obs": d["I_obs"], "sigma": d["sigma"],
            "dmax_nm": self.sp_dmax.value() / r_to_out,
            "n_r": self.sp_nr.value(),
            "max_iter": self.sp_maxiter.value(),
            "tol": self.sp_tol.value(),
            "n_samples": self.sp_samples.value(),
            "icf_nm": (icf_val / r_to_out) if icf_val > 0 else None,
            "r_to_out": r_to_out,
            "q_to_out": d["q_to_out"],
            "r_unit": d["r_unit"],
            "q_unit": d["q_unit"],
        }
        self._run_worker(SolveWorker(params), self._on_solve_done)

    # ------------------------------------------------------------------
    # Done callbacks
    # ------------------------------------------------------------------

    def _on_load_done(self, data: dict):
        # Raw load only — no filtering or Guinier yet
        data.update({"q": None, "I_obs": None, "sigma": None,
                     "gfit": None, "gfit_err": None})
        self._data = data
        self._pr_out = None
        self.btn_guinier.setEnabled(True)
        self.btn_scan.setEnabled(False)
        self.btn_solve.setEnabled(False)

        q_raw = data["q_raw"]
        q_to_out = data["q_to_out"]

        for sb in (self.sp_view_qmin, self.sp_view_qmax):
            sb.blockSignals(True)
        self.sp_view_qmin.setValue(round(float(q_raw.min()) * q_to_out, 4))
        self.sp_view_qmax.setValue(round(float(q_raw.max()) * q_to_out, 4))
        for sb in (self.sp_view_qmin, self.sp_view_qmax):
            sb.blockSignals(False)

        self.statusBar().showMessage(
            f"Loaded {len(q_raw)} points, "
            f"q = {q_raw.min()*q_to_out:.4f}–{q_raw.max()*q_to_out:.4f}"
            f" {data['q_unit']}  –  set q range then click Run Guinier."
        )
        self._refresh_data_tab()

    def _on_scan_done(self, out: dict):
        self._scan_out = out
        arr = out["scan"]
        best_idx = out["best_idx"]
        best_dm = arr[best_idx, 0]
        r_unit = out["r_unit"]
        self.sp_dmax.setValue(best_dm)
        self.statusBar().showMessage(
            f"Dmax scan complete – best Dmax = {best_dm:.2f} {r_unit}  "
            f"(log evid = {arr[best_idx, 1]:.4f})"
        )
        self._refresh_scan_tab()
        self._tabs.setCurrentIndex(2)

    def _on_icfscan_done(self, out: dict):
        self._icfscan_out = out
        arr = out["scan"]
        best_idx = out["best_idx"]
        best_w = arr[best_idx, 0]
        r_unit = out["r_unit"]
        # Populate the ICF width spinbox with the optimal w
        self.sp_icf.setValue(best_w)
        self.statusBar().showMessage(
            f"ICF scan complete – best w = {best_w:.3f} {r_unit}  "
            f"(log evid = {arr[best_idx, 1]:.4f},  "
            f"G = {arr[best_idx, 3]:.1f},  "
            f"{'converged' if arr[best_idx, 2] > 0.5 else 'NOT converged'})"
        )
        self._refresh_icfscan_tab()
        self._tabs.setCurrentIndex(3)

    def _on_solve_done(self, out: dict):
        self._pr_out = out
        res = out["result"]
        status = "converged" if res.converged else "NOT converged"
        mw = out["vc"]["MW_kDa"]
        mw_str = f"{mw:.0f} kDa" if np.isfinite(mw) else "n/a"
        self.statusBar().showMessage(
            f"P(r) {status} in {res.iterations} iter  |  "
            f"Rg = {out['Rg_pr']:.2f} {out['r_unit']},  "
            f"I₀ = {out['I0_pr']:.4g}  |  MW ≈ {mw_str}"
        )
        self.btn_save.setEnabled(True)
        self._refresh_data_tab()
        self._refresh_pr_tab()
        self._refresh_diag_tab()
        self._tabs.setCurrentIndex(4)

    # ------------------------------------------------------------------
    # Plot refresh methods
    # ------------------------------------------------------------------

    def _refresh_data_tab(self):
        ax = self.cvs_data.ax
        ax.cla()
        if self._data is None:
            self.cvs_data.draw()
            return

        d = self._data
        q_to_out = d["q_to_out"]

        # Raw (unfiltered) data in grey
        q_raw_out = d["q_raw"] * q_to_out
        ax.errorbar(q_raw_out, d["I_raw"], yerr=d["s_raw"],
                    fmt="none", ecolor="#dddddd", elinewidth=0.5, capsize=0, zorder=1)
        ax.plot(q_raw_out, d["I_raw"], "o", ms=2, color="#bbbbbb",
                label="Raw", zorder=2)

        # Filtered data on top (only after Guinier has been run)
        if d["q"] is not None:
            q_out = d["q"] * q_to_out
            I_obs = d["I_obs"]
            s = d["sigma"]
            ax.errorbar(q_out, I_obs, yerr=s, fmt="none", ecolor="#aac8e0",
                        elinewidth=0.6, capsize=0, zorder=3)
            ax.plot(q_out, I_obs, "o", ms=2.5, color="steelblue", label="Filtered", zorder=4)

        if self._pr_out is not None:
            p = self._pr_out
            ax.plot(p["q_fine"], p["Iq_fine"], "-", color="tomato",
                    lw=1.5, label="I(q) from P(r)", zorder=5)

        ax.set_yscale("log")
        ax.set_xlabel(f"q  [{d['q_unit']}]")
        ax.set_ylabel("I(q)")
        ax.set_title("SAXS Scattering Profile")
        ax.legend(fontsize=8, loc="upper right")
        self._apply_data_view()

    def _apply_data_view(self, *_):
        """Update x-scale and x-limits on the Data tab without replotting."""
        if self._data is None:
            return
        ax = self.cvs_data.ax
        scale = "linear" if self.cb_xscale.isChecked() else "log"
        ax.set_xscale(scale)
        qmin = self.sp_view_qmin.value()
        qmax = self.sp_view_qmax.value()
        if qmax > qmin > 0:
            ax.set_xlim(qmin, qmax)
        self.cvs_data.draw()

    def _refresh_mf_tab(self):
        axes = self.cvs_mf.axes
        ax_g, ax_k = axes[0], axes[1]
        ax_g.cla()
        ax_k.cla()

        if self._data is None or self._data.get("q") is None:
            self.cvs_mf.draw()
            return

        d = self._data
        q_to_out = d["q_to_out"]
        r_to_out = d["r_to_out"]
        q_unit = d["q_unit"]
        r_unit = d["r_unit"]
        q = d["q"]
        I_obs = d["I_obs"]
        gfit = d["gfit"]

        # ---- Guinier ----
        if gfit is not None:
            n_used = gfit["n_pts"]
            q2_out = gfit["q2_full"] * q_to_out ** 2
            lnI = gfit["lnI_full"]
            dlnI = gfit["dlnI_full"]

            # Show the plot from 0 to ~4× the Guinier limit so the fit
            # line is visible without the far-field data collapsing it.
            q2_guinier_max = q2_out[n_used - 1]
            x_lim = q2_guinier_max * 4.0

            outside_mask = q2_out[n_used:] <= x_lim
            if outside_mask.any():
                ax_g.errorbar(
                    q2_out[n_used:][outside_mask], lnI[n_used:][outside_mask],
                    yerr=dlnI[n_used:][outside_mask],
                    fmt="o", ms=3, color="#aaaaaa", ecolor="#cccccc",
                    elinewidth=0.5, capsize=0, label="Outside range",
                )
            ax_g.errorbar(
                q2_out[:n_used], lnI[:n_used], yerr=dlnI[:n_used],
                fmt="o", ms=4, color="C1", ecolor="#cccccc",
                elinewidth=0.5, capsize=0, label="Guinier range",
            )
            b_user = gfit["b"] / q_to_out ** 2
            q2_fit = np.linspace(0, q2_guinier_max * 1.2, 200)
            Rg_out = gfit["Rg"] * r_to_out
            ax_g.plot(q2_fit, gfit["a"] + b_user * q2_fit, "r-", lw=1.5,
                      label=f"Rg = {Rg_out:.2f} {r_unit},  R² = {gfit['r2']:.4f}")
            ax_g.set_xlim(left=0, right=x_lim)
        else:
            with np.errstate(invalid="ignore", divide="ignore"):
                lnI_raw = np.log(np.where(I_obs > 0, I_obs, np.nan))
            ax_g.plot(q ** 2 * q_to_out ** 2, lnI_raw, "o", ms=2, color="gray")

        ax_g.set_xlabel(f"q²  [{q_unit}²]")
        ax_g.set_ylabel("ln I(q)")
        ax_g.set_title("Guinier Plot")
        if gfit is not None:
            ax_g.legend(fontsize=8)

        # ---- Dimensionless Kratky ----
        if gfit is not None:
            # qRg is dimensionless: q_nm * Rg_nm (q_to_out * r_to_out = 1)
            qRg = q * gfit["Rg"]
            yK = qRg ** 2 * I_obs / gfit["I0"]
            qRg_max = float(qRg[-1])

            # Debye (Gaussian chain) reference: fully flexible/disordered
            # f_D(x) = 2(exp(-x) - 1 + x)/x²,  x = (qRg)²
            # Dimensionless Kratky = x * f_D(x) = 2(exp(-x) - 1 + x)/x
            qRg_ref = np.linspace(0.01, max(qRg_max, 5.0), 400)
            x = qRg_ref ** 2
            # small-x: expm1(-x)+x ≈ x²/2, so 2*(x²/2)/x = x → 0
            debye_kratky = np.where(x > 1e-4, 2.0 * (np.expm1(-x) + x) / x, x)

            ax_k.plot(qRg, yK, "o", ms=2.5, color="steelblue", label="Data", zorder=4)
            ax_k.plot(qRg_ref, debye_kratky, "--", color="#e07b39", lw=1.2,
                      label="Gaussian chain (disordered)", zorder=2)
            ax_k.axhline(2.0, color="#e07b39", ls=":", lw=0.7, alpha=0.6, zorder=1)
            ax_k.axvline(np.sqrt(3), color="gray", ls="--", lw=0.8,
                         label=f"√3 ≈ {np.sqrt(3):.3f}  (globular peak)")
            ax_k.axhline(3.0 / np.e, color="gray", ls=":", lw=0.8,
                         label=f"3/e ≈ {3/np.e:.3f}  (globular peak)")
            ax_k.set_xlabel("q · Rg")
            ax_k.set_ylabel("(q · Rg)² · I / I₀")
            ax_k.set_title("Dimensionless Kratky")
            # Annotate plateau value for disordered reference
            ax_k.annotate("2 (disordered plateau)", xy=(max(qRg_max * 0.7, 3.5), 2.0),
                          xytext=(0, 4), textcoords="offset points",
                          fontsize=7, color="#e07b39", va="bottom")
        else:
            q_out = q * q_to_out
            ax_k.plot(q_out, q_out ** 2 * I_obs, "o", ms=2.5, color="steelblue")
            ax_k.set_xlabel(f"q  [{q_unit}]")
            ax_k.set_ylabel(f"q² · I(q)")
            ax_k.set_title("Kratky Plot  (Guinier fit unavailable)")

        if gfit is not None:
            ax_k.legend(fontsize=7, loc="upper right")
        self.cvs_mf.draw()

    def _refresh_scan_tab(self):
        ax = self.cvs_scan.ax
        ax.cla()
        if self._scan_out is None:
            self.cvs_scan.draw()
            return

        out = self._scan_out
        arr = out["scan"]
        best_idx = out["best_idx"]
        r_unit = out["r_unit"]
        dmax_vals = arr[:, 0]
        log_ev = arr[:, 1]
        best_dm = dmax_vals[best_idx]

        ax.plot(dmax_vals, log_ev, "o-", ms=5, color="steelblue", lw=1.2)
        ax.axvline(best_dm, color="tomato", ls="--", lw=1.5,
                   label=f"Best  Dmax = {best_dm:.2f} {r_unit}")
        ax.set_xlabel(f"Dmax  [{r_unit}]")
        ax.set_ylabel("log evidence")
        ax.set_title("Dmax Scan")
        ax.legend(fontsize=9)
        self.cvs_scan.draw()

    def _refresh_icfscan_tab(self):
        axes = self.cvs_icfscan.axes
        ax_e, ax_g = axes[0], axes[1]
        ax_e.cla()
        ax_g.cla()

        if self._icfscan_out is None:
            self.cvs_icfscan.draw()
            return

        out = self._icfscan_out
        arr = out["scan"]
        best_idx = out["best_idx"]
        r_unit = out["r_unit"]

        w       = arr[:, 0]
        log_ev  = arr[:, 1]
        conv    = arr[:, 2] > 0.5
        G       = arr[:, 3]
        best_w  = w[best_idx]

        # ---- left: log evidence ----
        if conv.any():
            ax_e.plot(w[conv], log_ev[conv], "o-", ms=5,
                      color="steelblue", lw=1.2, label="converged")
        if (~conv).any():
            ax_e.plot(w[~conv], log_ev[~conv], "x", ms=7,
                      color="#aaaaaa", mew=1.5, label="not converged")
        ax_e.axvline(best_w, color="tomato", ls="--", lw=1.5,
                     label=f"best  w = {best_w:.3f} {r_unit}")
        ax_e.set_xscale("log")
        ax_e.set_xlabel(f"ICF width  [{r_unit}]")
        ax_e.set_ylabel("log evidence")
        ax_e.set_title("Evidence vs ICF width")
        ax_e.legend(fontsize=8)

        # ---- right: G (good measurements) ----
        ax_g.plot(w[conv],  G[conv],  "o-", ms=5, color="steelblue", lw=1.2)
        if (~conv).any():
            ax_g.plot(w[~conv], G[~conv], "x", ms=7,
                      color="#aaaaaa", mew=1.5)
        ax_g.axvline(best_w, color="tomato", ls="--", lw=1.5)
        ax_g.set_xscale("log")
        ax_g.set_xlabel(f"ICF width  [{r_unit}]")
        ax_g.set_ylabel("G  (good measurements)")
        ax_g.set_title("Effective data vs ICF width")

        self.cvs_icfscan.draw()

    def _refresh_diag_tab(self):
        axes = self.cvs_diag.axes
        for row in axes:
            for ax in row:
                ax.cla()

        if self._pr_out is None or not self._pr_out.get("trace"):
            self.cvs_diag.draw()
            return

        trace = self._pr_out["trace"]
        res   = self._pr_out["result"]
        N     = len(self._data["q"])

        its   = [row["it"]               for row in trace]
        alpha = [row["alpha"]            for row in trace]
        omega = [row["Omega"]            for row in trace]
        test  = [row["Test"]             for row in trace]
        asf   = [row["alpha_step_factor"] for row in trace]
        chi2  = [row["chi2"]             for row in trace]
        c2    = [row["c2"]               for row in trace]
        G_tr  = [row["G"]                for row in trace]

        chi2sc = [
            ch / c if (np.isfinite(c) and c > 0) else np.nan
            for ch, c in zip(chi2, c2)
        ]
        ng_tr = [N - g for g in G_tr]

        conv_status = "converged" if res.converged else "NOT converged"
        tol_val = 0.01  # default; we don't re-expose it here

        # ── top-left: alpha ──────────────────────────────────────────────
        ax = axes[0, 0]
        ax.semilogy(its, alpha, "o-", ms=4, color="steelblue", lw=1.2)
        ax.set_ylabel("α")
        ax.set_title(
            f"Alpha schedule  [{conv_status}, {res.iterations} iter, "
            f"α_final = {res.alpha:.4g}]",
            fontsize=9,
        )
        ax.grid(True, which="both", ls=":", alpha=0.4)

        # ── top-right: Omega and Test ────────────────────────────────────
        ax = axes[0, 1]
        om_pts = [(i, o) for i, o in zip(its, omega) if np.isfinite(o) and o > 0]
        if om_pts:
            xi, yi = zip(*om_pts)
            ax.semilogy(xi, yi, "s-", ms=4, color="tomato", lw=1.2, label="Ω")
        ax.axhline(1.0, ls="--", lw=1.0, color="k", label="Ω = 1 target")
        te_pts = [(i, t) for i, t in zip(its, test) if np.isfinite(t) and t > 0]
        if te_pts:
            xi, yi = zip(*te_pts)
            ax.semilogy(xi, yi, "^-", ms=4, color="darkorange", lw=1.2, label="Test")
        ax.axhline(tol_val, ls=":", lw=1.0, color="gray",
                   label=f"tol = {tol_val}")
        ax.set_ylabel("Ω  /  Test")
        ax.set_title("Omega & convergence test", fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, which="both", ls=":", alpha=0.4)

        # ── bottom-left: alpha step factor ───────────────────────────────
        ax = axes[1, 0]
        asf_plot = [a if (np.isfinite(a) and a < 1e6) else np.nan for a in asf]
        ax.plot(its, asf_plot, "D-", ms=4, color="purple", lw=1.2)
        ax.axhline(1.0, ls="--", lw=1.0, color="gray",
                   label="asf = 1  (no α constraint)")
        ax.set_yscale("log")
        ax.set_xlabel("iteration")
        ax.set_ylabel("alpha step factor")
        ax.set_title("Trust-region constraint on α", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, which="both", ls=":", alpha=0.4)

        # ── bottom-right: chi2/c2 vs N-G ────────────────────────────────
        ax = axes[1, 1]
        ax.plot(its, chi2sc, "o-", ms=4, color="darkgreen", lw=1.2,
                label="χ²/c²")
        ax.plot(its, ng_tr, "--", ms=3, color="#888888", lw=1.0,
                label="N − G  (target)")
        ax.set_xlabel("iteration")
        ax.set_ylabel("χ²/c²")
        ax.set_title("Data fit  (χ²/c² → N − G)", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, ls=":", alpha=0.4)

        self.cvs_diag.fig.suptitle(
            f"MaxEnt diagnostics  —  Dmax = {self.sp_dmax.value():.1f} "
            f"{self._data['r_unit']},  "
            f"ICF σ = {res.icf_width * self._data['r_to_out']:.4f} "
            f"{self._data['r_unit']}",
            fontsize=10,
        )
        self.cvs_diag.draw()

    def _refresh_pr_tab(self):
        ax = self.cvs_pr.ax
        ax.cla()
        if self._pr_out is None:
            self.cvs_pr.draw()
            self.te_summary.clear()
            return

        out = self._pr_out
        res = out["result"]
        r_out = res.r * out["r_to_out"]
        r_unit = out["r_unit"]

        ax.fill_between(r_out, out["lo"], out["hi"],
                        alpha=0.25, color="steelblue", label="5–95%")
        ax.plot(r_out, res.pr, "-", color="steelblue", lw=2.0, label="P(r)")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel(f"r  [{r_unit}]")
        ax.set_ylabel("P(r)")
        ax.set_title("Pair Distance Distribution")
        ax.legend(fontsize=9)
        self.cvs_pr.draw()

        chi2_sc = res.chi2 / res.c2
        icf_out = res.icf_width * out["r_to_out"]
        r2_unit = f"{r_unit}²"
        vc = out["vc"]
        Vc_out = vc["Vc_nm2"] * (out["r_to_out"] ** 2)
        mw = vc["MW_kDa"]
        mw_str = f"{mw:.1f}" if np.isfinite(mw) else "n/a"
        self.te_summary.setText(
            f"Converged : {res.converged}  ({res.iterations} iter)\n"
            f"ICF σ     : {icf_out:.4f} {r_unit}\n"
            f"alpha     : {res.alpha:.5g}\n"
            f"c²        : {res.c2:.5f}   (σ_eff = {res.c2**0.5:.4f}·σ)\n"
            f"χ²        : {res.chi2:.3f}\n"
            f"χ²/c²     : {chi2_sc:.3f}\n"
            f"G         : {res.G:.1f}\n"
            f"log evid  : {res.log_evidence:.4f}\n"
            f"Rg [P(r)] : {out['Rg_pr']:.4f} {r_unit}\n"
            f"I₀ [P(r)] : {out['I0_pr']:.5g}\n"
            f"Vc        : {Vc_out:.4g} {r2_unit}\n"
            f"MW [prot] : {mw_str} kDa  (Rambo & Tainer 2013)\n"
        )

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------

    def _do_save(self):
        if self._pr_out is None:
            return
        from pathlib import Path
        default = Path(self.le_path.text()).stem if self.le_path.text() else "prsolve"
        prefix, _ = QFileDialog.getSaveFileName(
            self, "Save results – enter output prefix", default, "All files (*)"
        )
        if not prefix:
            return
        for ext in ("_pr.dat", "_iq.dat", "_obs.dat", "_report.txt", ".dat", ".txt"):
            if prefix.endswith(ext):
                prefix = prefix[: -len(ext)]
                break
        try:
            self._write_results(prefix)
            self.statusBar().showMessage(f"Results saved with prefix: {prefix}")
        except Exception as exc:
            QMessageBox.critical(self, "Save error", str(exc))

    def _write_results(self, prefix: str):
        out = self._pr_out
        d = self._data
        res = out["result"]
        r_to_out = out["r_to_out"]
        q_to_out = out["q_to_out"]
        r_unit = out["r_unit"]
        q_unit = out["q_unit"]
        r2_unit = f"{r_unit}²"

        r_out = res.r * r_to_out
        q_obs_out = d["q"] * q_to_out
        vc = out["vc"]
        Vc_out = vc["Vc_nm2"] * (r_to_out ** 2)

        # P(r) with posterior band
        pr_file = f"{prefix}_pr.dat"
        np.savetxt(
            pr_file,
            np.column_stack([r_out, res.pr, out["lo"], out["hi"]]),
            header=f"r [{r_unit}]  P(r)  P(r)_5pct  P(r)_95pct",
            fmt="%.6e",
        )

        # Observed data + fit
        obs_file = f"{prefix}_obs.dat"
        np.savetxt(
            obs_file,
            np.column_stack([q_obs_out, d["I_obs"], d["sigma"], out["Iq_obs_fit"]]),
            header=f"q [{q_unit}]  I_obs  sigma  I_fit",
            fmt="%.6e",
        )

        # Theoretical I(q) to q=0
        iq_file = f"{prefix}_iq.dat"
        np.savetxt(
            iq_file,
            np.column_stack([out["q_fine"], out["Iq_fine"]]),
            header=(
                f"q [{q_unit}]  I_theory"
                f"  (4pi * integral P(r) sinc(qr) dr, q->0 included)"
            ),
            fmt="%.6e",
        )

        # Guinier plot data
        guinier_file = None
        gfit = d.get("gfit")
        if gfit is not None:
            guinier_file = f"{prefix}_guinier.dat"
            _write_guinier_file(guinier_file, gfit, q_to_out, q_unit)

        # Text report
        report_file = f"{prefix}_report.txt"
        c2 = res.c2
        chi2_sc = res.chi2 / c2
        N = len(d["q"])
        sep = "=" * 62
        with open(report_file, "w") as f:
            f.write(f"{sep}\n  prsolve – P(r) reconstruction report\n{sep}\n\n")
            f.write(f"Input file   : {self.le_path.text()}\n")
            f.write(f"Units        : {d['units']}  ({q_unit}  /  {r_unit})\n")
            f.write(
                f"Data points  : {N}   q = {d['q'][0]*q_to_out:.4f}–"
                f"{d['q'][-1]*q_to_out:.4f} {q_unit}\n"
            )
            f.write(f"Dmax         : {self.sp_dmax.value():.4f} {r_unit}\n")
            f.write(f"n_r          : {self.sp_nr.value()}\n")
            f.write(
                f"ICF width    : {res.icf_width*r_to_out:.4f} {r_unit}\n\n"
            )

            f.write("--- Guinier fit ---\n")
            if gfit is not None:
                Rg_g = gfit["Rg"] * r_to_out
                f.write(f"  Rg         : {Rg_g:.4f} {r_unit}\n")
                f.write(f"  I(0)       : {gfit['I0']:.6g}\n")
                f.write(
                    f"  q range    : {gfit['q_min']*q_to_out:.4f}–"
                    f"{gfit['q_max']*q_to_out:.4f} {q_unit}"
                    f"  ({gfit['n_pts']} pts)\n"
                )
                f.write(f"  R²         : {gfit['r2']:.6f}\n\n")
            else:
                f.write("  Guinier fit unavailable.\n\n")

            f.write("--- MaxEnt P(r) solution ---\n")
            f.write(f"  Converged  : {res.converged}  ({res.iterations} iters)\n")
            f.write(f"  alpha      : {res.alpha:.5g}\n")
            f.write(f"  c²         : {c2:.5f}   (σ_eff = {c2**0.5:.4f}·σ)\n")
            f.write(f"  χ²  (raw)  : {res.chi2:.3f}\n")
            f.write(f"  χ²/c²      : {chi2_sc:.3f}   (≈ N−G = {N - res.G:.1f})\n")
            f.write(f"  χ²/c² + G  : {chi2_sc + res.G:.1f}   (≈ N = {N})\n")
            f.write(f"  G          : {res.G:.1f}  (good measurements)\n")
            f.write(f"  log evid   : {res.log_evidence:.4f}\n\n")

            f.write("--- Derived from P(r) ---\n")
            f.write(f"  Rg  [P(r)] : {out['Rg_pr']:.4f} {r_unit}\n")
            f.write(f"  I(0)[P(r)] : {out['I0_pr']:.6g}   (= 4π ∫P(r) dr)\n")
            f.write(f"  Vc         : {Vc_out:.4g} {r2_unit}  (volume of correlation)\n")
            f.write(
                f"  MW [prot]  : {vc['MW_kDa']:.1f} kDa"
                f"  (Rambo & Tainer 2013; relative scale)\n\n"
            )

            f.write("--- Output files ---\n")
            files = [
                (pr_file,     "r, P(r), 5%, 95% posterior band"),
                (obs_file,    "q, I_obs, sigma, I_fit"),
                (iq_file,     "q, I_theory from P(r) incl. q→0"),
                (report_file, "this report"),
            ]
            if guinier_file:
                files.insert(3, (guinier_file, "q², ln I, Guinier fit"))
            w = max(len(p) for p, _ in files) + 2
            for fp, desc in files:
                f.write(f"  {fp:<{w}} {desc}\n")

    # ------------------------------------------------------------------
    # Worker infrastructure
    # ------------------------------------------------------------------

    def _run_worker(self, worker: _BaseWorker, on_done):
        if self._thread and self._thread.isRunning():
            QMessageBox.warning(self, "Busy", "A computation is already running.")
            return

        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)

        def _finish(_=None):
            thread.quit()
            thread.wait()
            self._thread = None
            self._worker = None
            self._set_busy(False)

        worker.done.connect(on_done)
        worker.done.connect(_finish)
        worker.error.connect(self._on_worker_error)
        worker.error.connect(_finish)

        if hasattr(worker, "progress"):
            worker.progress.connect(self.statusBar().showMessage)

        self._thread = thread
        self._worker = worker
        self._set_busy(True)
        thread.start()

    def _set_busy(self, busy: bool):
        self.btn_load.setEnabled(not busy)
        if self._data is not None:
            self.btn_guinier.setEnabled(not busy)
            if self._data.get("q") is not None:
                self.btn_scan.setEnabled(not busy)
                self.btn_icfscan.setEnabled(not busy)
                self.btn_solve.setEnabled(not busy)
        if self._pr_out is not None:
            self.btn_save.setEnabled(not busy)

    def _on_worker_error(self, msg: str):
        QMessageBox.critical(self, "Error", msg)
        self.statusBar().showMessage("Error – see dialog for details.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def launch():
    import sys
    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
