"""
PyQt6 interactive dashboard — Almgren-Chriss Optimal Execution.

Run:
    python app/dashboard.py
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from typing import Optional

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

from datetime import datetime

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QGroupBox, QLabel, QDoubleSpinBox, QSpinBox, QSlider, QPushButton,
    QTabWidget, QStatusBar, QScrollArea, QSizePolicy, QFrame, QSplitter,
    QTextEdit,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QFont, QColor

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from src.almgren_chriss import AlmgrenChrissModel
from src.simulator import ExecutionSimulator, SimulationResult


# ─────────────────────────────── Worker thread ────────────────────────────────

class MCWorker(QObject):
    finished = pyqtSignal(object)   # SimulationResult dict
    status   = pyqtSignal(str)

    def __init__(self, model: AlmgrenChrissModel, n_sims: int) -> None:
        super().__init__()
        self._model  = model
        self._n_sims = n_sims

    def run(self) -> None:
        self.status.emit(f"Running simulation  ({self._n_sims:,} paths)…")
        sim     = ExecutionSimulator(self._model, S0=100.0)
        results = sim.run_all_strategies_with_paths(n_sims=self._n_sims, seed=42)
        self.finished.emit(results)


# ─────────────────────────────── Matplotlib canvas ───────────────────────────

class Canvas(FigureCanvas):
    def __init__(self, nrows: int = 1, ncols: int = 1, figsize=(6, 4)) -> None:
        self.fig = Figure(figsize=figsize)
        self.fig.set_tight_layout(True)
        self.axes = self.fig.subplots(nrows, ncols)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)


# ─────────────────────────────── Main window ─────────────────────────────────

COLORS = {"Almgren-Chriss": "#1565C0", "TWAP": "#2E7D32", "VWAP": "#C62828"}
MARKERS = {"Almgren-Chriss": "*", "TWAP": "s", "VWAP": "^"}
_N_PATHS_PLOT = 300  # individual path lines rendered per subplot


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Almgren-Chriss Optimal Execution")
        self.resize(1440, 860)

        self._mc_thread: Optional[QThread] = None
        self._mc_worker: Optional[MCWorker] = None

        self._build_ui()
        self._connect_signals()
        self._refresh(log=True)
        self._log("Dashboard ready — adjust parameters or run Monte Carlo.", "success")

    # ── Build ──────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QHBoxLayout(root)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        # Left control panel
        ctrl = self._build_controls()
        ctrl.setFixedWidth(260)
        main_layout.addWidget(ctrl)

        # Right: tabs
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        main_layout.addWidget(self.tabs, stretch=1)

        self._build_tab_model()
        self._build_tab_mc()

        # Status bar
        self.statusBar().showMessage("Ready")

    def _build_controls(self) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(4)

        # ── Position ──────────────────────────────────────────────────────
        g = QGroupBox("Position")
        gl = QVBoxLayout(g)
        gl.setSpacing(2)

        gl.addWidget(QLabel("X  —  shares to liquidate"))
        self.w_X = QSpinBox()
        self.w_X.setRange(1_000, 10_000_000)
        self.w_X.setValue(100_000)
        self.w_X.setSingleStep(5_000)
        gl.addWidget(self.w_X)

        gl.addWidget(QLabel("T  —  horizon (trading days)"))
        self.w_T = QSpinBox()
        self.w_T.setRange(1, 60)
        self.w_T.setValue(5)
        gl.addWidget(self.w_T)

        gl.addWidget(QLabel("N  —  execution steps"))
        self.w_N = QSpinBox()
        self.w_N.setRange(5, 200)
        self.w_N.setValue(50)
        gl.addWidget(self.w_N)

        lay.addWidget(g)

        # ── Market ────────────────────────────────────────────────────────
        g = QGroupBox("Market Parameters")
        gl = QVBoxLayout(g)
        gl.setSpacing(2)

        def dspin(lo, hi, val, step, dec) -> QDoubleSpinBox:
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setValue(val)
            s.setSingleStep(step)
            s.setDecimals(dec)
            return s

        gl.addWidget(QLabel("σ  —  vol ($/share/yr,  30% @ S₀=$100 → σ=30)"))
        self.w_sigma = dspin(0.01, 500.0, 30.0, 0.5, 1)
        gl.addWidget(self.w_sigma)

        gl.addWidget(QLabel("η  —  temporary impact  ($/share·yr)"))
        self.w_eta = dspin(0.0, 0.5, 2e-5, 1e-6, 6)
        gl.addWidget(self.w_eta)

        gl.addWidget(QLabel("γ  —  permanent impact  ($/share)"))
        self.w_gamma = dspin(0.0, 0.1, 1e-5, 1e-6, 6)
        gl.addWidget(self.w_gamma)

        lay.addWidget(g)

        # ── Risk aversion ─────────────────────────────────────────────────
        g = QGroupBox("Risk Aversion")
        gl = QVBoxLayout(g)
        gl.setSpacing(2)

        self.lbl_lam = QLabel("log₁₀(λ) = −5.2   →   λ = 6.3e−06")
        self.lbl_lam.setFont(QFont("Courier", 8))
        gl.addWidget(self.lbl_lam)

        self.sld_lam = QSlider(Qt.Orientation.Horizontal)
        self.sld_lam.setRange(-100, 30)  # ×0.1 → [−10, 3]
        self.sld_lam.setValue(-52)
        gl.addWidget(self.sld_lam)

        lay.addWidget(g)

        # ── Metrics box ───────────────────────────────────────────────────
        g = QGroupBox("Analytical Metrics")
        gl = QVBoxLayout(g)
        gl.setSpacing(1)
        mono = QFont("Courier", 9)

        self.lbl_kappa  = QLabel("κ         =  —")
        self.lbl_ecost  = QLabel("E[C]      =  —")
        self.lbl_risk   = QLabel("√Var[C]   =  —")
        self.lbl_ratio  = QLabel("E[C]/σ    =  —")

        for lbl in (self.lbl_kappa, self.lbl_ecost, self.lbl_risk, self.lbl_ratio):
            lbl.setFont(mono)
            gl.addWidget(lbl)
        lay.addWidget(g)

        # ── Monte Carlo ───────────────────────────────────────────────────
        g = QGroupBox("Monte Carlo")
        gl = QVBoxLayout(g)
        gl.setSpacing(2)

        gl.addWidget(QLabel("Number of simulations"))
        self.w_nsims = QSpinBox()
        self.w_nsims.setRange(500, 200_000)
        self.w_nsims.setValue(10_000)
        self.w_nsims.setSingleStep(500)
        gl.addWidget(self.w_nsims)

        self.btn_mc = QPushButton("▶  Run Monte Carlo")
        self.btn_mc.setStyleSheet(
            "QPushButton{"
            "  background:#1565C0; color:white; border-radius:4px;"
            "  padding:7px; font-weight:bold; font-size:11px;"
            "}"
            "QPushButton:hover{background:#0D47A1;}"
            "QPushButton:disabled{background:#90A4AE; color:#fff;}"
        )
        gl.addWidget(self.btn_mc)

        self.lbl_mc_status = QLabel("")
        self.lbl_mc_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_mc_status.setWordWrap(True)
        self.lbl_mc_status.setFont(QFont("Courier", 8))
        gl.addWidget(self.lbl_mc_status)

        lay.addWidget(g)

        # ── Log ───────────────────────────────────────────────────────────
        g = QGroupBox("Logs")
        gl = QVBoxLayout(g)
        gl.setContentsMargins(4, 4, 4, 4)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFont(QFont("Courier", 8))
        self.log_box.setFixedHeight(180)
        self.log_box.setStyleSheet(
            "QTextEdit {"
            "  background: #1E1E1E;"
            "  color: #D4D4D4;"
            "  border: 1px solid #444;"
            "  border-radius: 3px;"
            "}"
        )
        gl.addWidget(self.log_box)

        lay.addWidget(g)
        lay.addStretch()

        scroll.setWidget(w)
        return scroll

    def _build_tab_model(self) -> None:
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)

        self.cv_traj     = Canvas(2, 1, figsize=(6, 6))
        self.cv_frontier = Canvas(1, 1, figsize=(6, 6))

        lay.addWidget(self.cv_traj, stretch=1)
        lay.addWidget(self.cv_frontier, stretch=1)

        self.tabs.addTab(w, "  Model & Frontier  ")

    def _build_tab_mc(self) -> None:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)

        # ── Price paths (top — larger) ─────────────────────────────────────
        self.cv_mc_paths = Canvas(1, 1, figsize=(14, 4))
        self._init_paths_placeholder()
        lay.addWidget(self.cv_mc_paths, stretch=2)

        # ── IS distribution + frontier (bottom row) ────────────────────────
        mid = QWidget()
        mid_lay = QHBoxLayout(mid)
        mid_lay.setContentsMargins(0, 0, 0, 0)
        mid_lay.setSpacing(4)
        self.cv_mc_dist     = Canvas(1, 1, figsize=(6, 3))
        self.cv_mc_frontier = Canvas(1, 1, figsize=(6, 3))
        mid_lay.addWidget(self.cv_mc_dist,     stretch=1)
        mid_lay.addWidget(self.cv_mc_frontier, stretch=1)
        lay.addWidget(mid, stretch=1)

        # ── Summary table ──────────────────────────────────────────────────
        self.lbl_mc_table = QLabel(
            "Set parameters and click  ▶ Run Monte Carlo  to see results."
        )
        self.lbl_mc_table.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_mc_table.setFont(QFont("Courier", 9))
        self.lbl_mc_table.setFrameStyle(QFrame.Shape.Box)
        lay.addWidget(self.lbl_mc_table)

        self.tabs.addTab(w, "  Monte Carlo  ")

    # ── Logging ───────────────────────────────────────────────────────────────

    _LOG_COLORS = {
        "info":    "#D4D4D4",
        "success": "#4EC9B0",
        "warn":    "#CE9178",
        "mc":      "#9CDCFE",
    }

    def _log(self, msg: str, kind: str = "info") -> None:
        ts    = datetime.now().strftime("%H:%M:%S")
        color = self._LOG_COLORS.get(kind, "#D4D4D4")
        self.log_box.append(
            f'<span style="color:#858585">[{ts}]</span> '
            f'<span style="color:{color}">{msg}</span>'
        )
        self.log_box.verticalScrollBar().setValue(
            self.log_box.verticalScrollBar().maximum()
        )

    # ── Signals ───────────────────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        for w in (self.w_X, self.w_T, self.w_N,
                  self.w_sigma, self.w_eta, self.w_gamma):
            # valueChanged passes the new numeric value — drop it with lambda
            w.valueChanged.connect(lambda _: self._refresh())
            # log only when the user finishes editing (Enter / focus-out)
            w.editingFinished.connect(lambda: self._refresh(log=True))

        self.sld_lam.valueChanged.connect(self._on_lam_change)
        self.btn_mc.clicked.connect(self._start_mc)

    def _on_lam_change(self, v: int) -> None:
        exp = v / 10.0
        lam = 10.0 ** exp
        self.lbl_lam.setText(f"log₁₀(λ) = {exp:+.1f}   →   λ = {lam:.2e}")
        self._log(f"λ = {lam:.2e}  (log₁₀λ = {exp:+.1f})", "info")
        self._refresh()

    # ── Model builder ─────────────────────────────────────────────────────────

    def _make_model(self) -> AlmgrenChrissModel:
        lam = 10.0 ** (self.sld_lam.value() / 10.0)
        return AlmgrenChrissModel(
            X     = float(self.w_X.value()),
            T     = self.w_T.value() / 252.0,
            N     = self.w_N.value(),
            sigma = self.w_sigma.value(),
            eta   = self.w_eta.value(),
            gamma = self.w_gamma.value(),
            lam   = lam,
        )

    # ── Refresh (live update) ─────────────────────────────────────────────────

    def _refresh(self, log: bool = False) -> None:
        model  = self._make_model()
        result = model.solve()

        risk = np.sqrt(result.cost_variance)

        # Metrics labels
        self.lbl_kappa.setText(f"κ         = {result.kappa:.6f}")
        self.lbl_ecost.setText(f"E[C]      = {result.expected_cost:>14,.0f}")
        self.lbl_risk.setText( f"√Var[C]   = {risk:>14,.0f}")
        ratio = result.expected_cost / max(risk, 1.0)
        self.lbl_ratio.setText(f"E[C]/σ    = {ratio:>14,.2f}")

        self.statusBar().showMessage(
            f"  κ = {result.kappa:.4f}   |   E[C] = {result.expected_cost:,.0f}"
            f"   |   √Var[C] = {risk:,.0f}   |   λ = {model.lam:.2e}"
        )

        if log:
            self._log(
                f"σ={model.sigma:.4g}  η={model.eta:.2e}  γ={model.gamma:.2e}"
                f"  X={model.X:,.0f}  T={model.T*252:.0f}d  "
                f"κ={result.kappa:.4f}  E[C]={result.expected_cost:,.0f}",
                "info",
            )

        self._draw_trajectory(model, result)
        self._draw_frontier(model, result)

    # ── Trajectory plot ───────────────────────────────────────────────────────

    def _draw_trajectory(self, m: AlmgrenChrissModel, result) -> None:
        fig = self.cv_traj.fig
        fig.clear()

        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212, sharex=ax1)
        fig.subplots_adjust(hspace=0.04, left=0.13, right=0.97, top=0.93, bottom=0.09)

        t = result.times * 252
        X = m.X
        twap = m.twap_trajectory()
        vwap = m.vwap_trajectory()

        ax1.plot(t, result.holdings / X, color=COLORS["Almgren-Chriss"], lw=2,
                 label="Almgren-Chriss")
        ax1.plot(t, twap / X, color=COLORS["TWAP"], lw=1.5, ls="--", label="TWAP")
        ax1.plot(t, vwap / X, color=COLORS["VWAP"], lw=1.5, ls=":",  label="VWAP")
        ax1.set_ylabel("x(t) / X")
        ax1.set_ylim(-0.02, 1.02)
        ax1.set_title(f"Optimal trajectory   (κ = {result.kappa:.5f})")
        ax1.legend(fontsize=8, loc="upper right")
        ax1.grid(True, alpha=0.25)
        plt.setp(ax1.get_xticklabels(), visible=False)

        bw = (t[1] - t[0]) * 0.8 if len(t) > 1 else 0.05
        ax2.bar(t[1:], result.trades / X, width=bw,
                color=COLORS["Almgren-Chriss"], alpha=0.75)
        ax2.set_xlabel("Trading days")
        ax2.set_ylabel("n_j / X")
        ax2.set_title("Volume sold per step")
        ax2.grid(True, alpha=0.25)

        self.cv_traj.draw()

    # ── Efficient frontier plot ───────────────────────────────────────────────

    def _draw_frontier(self, m: AlmgrenChrissModel, result) -> None:
        from matplotlib.collections import LineCollection
        from matplotlib.colors import LinearSegmentedColormap

        _CMAP = LinearSegmentedColormap.from_list(
            "ac_frontier", ["#1565C0", "#FDD835", "#C62828"]
        )

        fig = self.cv_frontier.fig
        fig.set_tight_layout(False)
        fig.clear()
        ax = fig.add_subplot(111)
        fig.subplots_adjust(left=0.13, right=0.83, top=0.91, bottom=0.12)

        risks_ef, costs_ef = m.efficient_frontier(n_points=400)
        mask = np.isfinite(risks_ef) & np.isfinite(costs_ef)
        risks_ef, costs_ef = risks_ef[mask], costs_ef[mask]
        n = len(risks_ef)

        # ── Gradient line: bleu (λ→∞, conservateur) → rouge (λ→0, agressif) ──
        pts  = np.column_stack([risks_ef, costs_ef]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        t    = np.linspace(0.0, 1.0, n)
        lc   = LineCollection(segs, cmap=_CMAP, linewidth=3.0, zorder=4, alpha=0.95)
        lc.set_array(t)
        ax.add_collection(lc)

        # ── Fill below frontier ───────────────────────────────────────────────
        floor = costs_ef.min() * 0.97
        ax.fill_between(risks_ef, costs_ef, floor, alpha=0.07, color="#1565C0", zorder=1)

        # ── Strategy markers (halo + point + annotation) ──────────────────────
        e_ac  = result.expected_cost
        v_ac  = result.cost_variance
        e_twap, v_twap = m.cost_from_trajectory(m.twap_trajectory())
        e_vwap, v_vwap = m.cost_from_trajectory(m.vwap_trajectory())

        for label, rx, cy, color, mk, sz in [
            (f"AC  λ={m.lam:.1e}", np.sqrt(v_ac),  e_ac,   COLORS["Almgren-Chriss"], "*", 280),
            ("TWAP",               np.sqrt(v_twap), e_twap, COLORS["TWAP"],           "s", 160),
            ("VWAP",               np.sqrt(v_vwap), e_vwap, COLORS["VWAP"],           "^", 160),
        ]:
            ax.scatter(rx, cy, s=sz * 3.5, c=color, marker=mk, alpha=0.15, zorder=5)
            ax.scatter(rx, cy, s=sz,       c=color, marker=mk, zorder=9,
                       edgecolors="white", linewidths=1.5)
            ax.annotate(
                label,
                xy=(rx, cy), xytext=(10, 4), textcoords="offset points",
                fontsize=8, fontweight="bold", color=color,
                bbox=dict(boxstyle="round,pad=0.30", fc="white",
                          ec=color, alpha=0.88, lw=1.2),
            )

        # ── Colorbar ──────────────────────────────────────────────────────────
        sm = plt.cm.ScalarMappable(cmap=_CMAP, norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.03)
        cbar.set_ticks([0.0, 1.0])
        cbar.set_ticklabels(["Conservative\n(λ → ∞)", "Aggressive\n(λ → 0)"], fontsize=7)
        cbar.ax.set_title("λ", fontsize=9, pad=4)

        # ── Endpoint annotations ──────────────────────────────────────────────
        ax.annotate(
            "Immediate\n(λ→∞)",
            xy=(risks_ef[0], costs_ef[0]), xytext=(6, -28),
            textcoords="offset points", fontsize=7, color="#78909C", style="italic",
            arrowprops=dict(arrowstyle="-", color="#B0BEC5", lw=0.8),
        )
        ax.annotate(
            "TWAP\n(λ→0)",
            xy=(risks_ef[-1], costs_ef[-1]), xytext=(-54, 12),
            textcoords="offset points", fontsize=7, color="#78909C", style="italic",
            arrowprops=dict(arrowstyle="-", color="#B0BEC5", lw=0.8),
        )

        # ── Style axes ────────────────────────────────────────────────────────
        dx = risks_ef.max() - risks_ef.min()
        dy = costs_ef.max() - costs_ef.min()
        ax.set_xlim(risks_ef.min() - 0.06 * dx, risks_ef.max() + 0.10 * dx)
        ax.set_ylim(costs_ef.min() - 0.10 * dy, costs_ef.max() + 0.22 * dy)
        ax.set_xlabel("√Var[C]  —  Execution risk", fontsize=10)
        ax.set_ylabel("E[C]  —  Expected cost", fontsize=10)
        ax.set_title("Efficient Frontier  Cost–Risk",
                     fontsize=11, fontweight="bold", pad=6)
        ax.grid(True, alpha=0.20, linestyle="--", color="#9E9E9E")
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        self.cv_frontier.draw()

    # ── Monte Carlo ───────────────────────────────────────────────────────────

    def _start_mc(self) -> None:
        self.btn_mc.setEnabled(False)
        self.lbl_mc_status.setText("Initialising…")
        self.tabs.setCurrentIndex(1)

        model = self._make_model()
        n     = self.w_nsims.value()

        self._log(
            f"▶ Monte Carlo started — {n:,} simulations"
            f"  (σ={model.sigma:.4g}  η={model.eta:.2e}  λ={model.lam:.2e})",
            "mc",
        )

        self._mc_thread = QThread()
        self._mc_worker = MCWorker(model, n)
        self._mc_worker.moveToThread(self._mc_thread)

        self._mc_thread.started.connect(self._mc_worker.run)
        self._mc_worker.status.connect(self.lbl_mc_status.setText)
        self._mc_worker.status.connect(lambda msg: self._log(msg, "mc"))
        self._mc_worker.finished.connect(self._on_mc_done)
        self._mc_worker.finished.connect(self._mc_thread.quit)
        self._mc_worker.finished.connect(self._mc_worker.deleteLater)
        self._mc_thread.finished.connect(self._mc_thread.deleteLater)

        self._mc_thread.start()

    def _on_mc_done(self, results: dict) -> None:
        self.btn_mc.setEnabled(True)
        n = list(results.values())[0].n_sims
        self.lbl_mc_status.setText(f"✓  {n:,} simulations complete")
        self._log(f"✓ Done — {n:,} paths", "success")
        for r in results.values():
            self._log(
                f"  {r.strategy:<18}  E[IS]={r.mean_cost:>10,.0f}"
                f"  σ={r.std_cost:>10,.0f}  CVaR95={r.cvar_95:>10,.0f}",
                "success",
            )

        self._draw_mc_paths(results)
        self._draw_mc_distributions(results)
        self._draw_mc_frontier(results)
        self._update_mc_table(results)

    def _init_paths_placeholder(self) -> None:
        fig = self.cv_mc_paths.fig
        fig.clear()
        fig.text(
            0.5, 0.5,
            "Run Monte Carlo to see price paths",
            ha="center", va="center",
            fontsize=12, color="#90A4AE", style="italic",
        )
        self.cv_mc_paths.draw()

    def _draw_mc_paths(self, results: dict) -> None:
        fig = self.cv_mc_paths.fig
        fig.clear()

        strategies = list(results.keys())
        axes = fig.subplots(1, len(strategies))
        fig.subplots_adjust(left=0.07, right=0.98, top=0.84, bottom=0.14, wspace=0.32)

        for i, (ax, (name, r)) in enumerate(zip(axes, results.items())):
            paths = r.price_paths  # (n_kept, N+1)
            if paths is None:
                ax.text(0.5, 0.5, "No paths", ha="center", va="center",
                        transform=ax.transAxes)
                continue

            base_color = COLORS[name]
            n_kept, n_steps = paths.shape
            t = np.arange(n_steps)

            n_plot = min(n_kept, _N_PATHS_PLOT)
            for path in paths[:n_plot]:
                ax.plot(t, path, color=base_color, alpha=0.10, lw=0.6,
                        rasterized=True)

            q1     = np.percentile(paths, 25, axis=0)
            q3     = np.percentile(paths, 75, axis=0)
            median = np.median(paths, axis=0)
            mean   = np.mean(paths, axis=0)

            ax.fill_between(t, q1, q3,
                            color=base_color, alpha=0.28,
                            label="Q1 – Q3", zorder=2)
            ax.plot(t, q1, color=base_color, lw=1.2, ls=":",
                    alpha=0.90, zorder=3)
            ax.plot(t, q3, color=base_color, lw=1.2, ls=":",
                    alpha=0.90, zorder=3)

            ax.plot(t, median,
                    color="#111111", lw=2.6, ls="-",
                    label="Median", zorder=5)

            ax.plot(t, mean,
                    color="#E53935", lw=2.0, ls="--",
                    label="Mean", zorder=4)

            ax.set_title(name, fontsize=10, fontweight="bold", color=base_color)
            ax.set_xlabel("Execution step", fontsize=8)
            if i == 0:
                ax.set_ylabel("Mid price S ($)", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=8, loc="lower left",
                      framealpha=0.85, edgecolor="#cccccc")
            ax.grid(True, alpha=0.18)
            ax.set_xlim(0, n_steps - 1)

        n_sims = list(results.values())[0].n_sims
        n_kept = list(results.values())[0].price_paths.shape[0]
        fig.suptitle(
            f"Price paths  —  {n_sims:,} simulations  "
            f"({min(n_kept, _N_PATHS_PLOT):,} paths displayed / strategy)",
            fontsize=10, y=0.99,
        )
        self.cv_mc_paths.draw()

    def _draw_mc_distributions(self, results: dict) -> None:
        fig = self.cv_mc_dist.fig
        fig.clear()
        ax = fig.add_subplot(111)
        fig.subplots_adjust(left=0.09, right=0.97, top=0.90, bottom=0.12)

        for name, r in results.items():
            ax.hist(r.implementation_shortfalls, bins=90, density=True,
                    alpha=0.50, color=COLORS[name], label=name, edgecolor="none")
            ax.axvline(r.mean_cost, color=COLORS[name], lw=1.8, ls="--")
            ax.axvline(r.cvar_95,   color=COLORS[name], lw=1.2, ls=":")

        ax.set_xlabel("Implementation Shortfall (IS)")
        ax.set_ylabel("Density")
        ax.set_title("Execution cost distribution   (- - E[IS]   ···  CVaR 95%)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)

        self.cv_mc_dist.draw()

    def _draw_mc_frontier(self, results: dict) -> None:
        from matplotlib.collections import LineCollection
        from matplotlib.colors import LinearSegmentedColormap

        _CMAP = LinearSegmentedColormap.from_list(
            "ac_frontier", ["#1565C0", "#FDD835", "#C62828"]
        )

        m = self._make_model()
        risks_ef, costs_ef = m.efficient_frontier(n_points=300)
        mask = np.isfinite(risks_ef) & np.isfinite(costs_ef)
        risks_ef, costs_ef = risks_ef[mask], costs_ef[mask]

        fig = self.cv_mc_frontier.fig
        fig.set_tight_layout(False)
        fig.clear()
        ax = fig.add_subplot(111)
        fig.subplots_adjust(left=0.11, right=0.83, top=0.90, bottom=0.14)

        # ── Analytical frontier (gradient, thin line) ─────────────────────────
        n    = len(risks_ef)
        pts  = np.column_stack([risks_ef, costs_ef]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc   = LineCollection(segs, cmap=_CMAP, linewidth=2.0, zorder=3, alpha=0.60)
        lc.set_array(np.linspace(0.0, 1.0, n))
        ax.add_collection(lc)

        # ── Simulated strategy markers ─────────────────────────────────────────
        for name, r in results.items():
            color, mk = COLORS[name], MARKERS[name]
            ax.scatter(r.std_cost, r.mean_cost, s=350,
                       c=color, marker=mk, alpha=0.18, zorder=5)
            ax.scatter(r.std_cost, r.mean_cost, s=140,
                       c=color, marker=mk, zorder=8,
                       edgecolors="white", linewidths=1.4,
                       label=f"{name}  CVaR={r.cvar_95:,.0f}")

        # ── Style ─────────────────────────────────────────────────────────────
        sm = plt.cm.ScalarMappable(cmap=_CMAP, norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_ticks([0.0, 1.0])
        cbar.set_ticklabels(["λ→∞", "λ→0"], fontsize=7)
        cbar.ax.set_title("λ", fontsize=8, pad=3)

        dx = risks_ef.max() - risks_ef.min()
        dy = costs_ef.max() - costs_ef.min()
        ax.set_xlim(risks_ef.min() - 0.06 * dx, risks_ef.max() + 0.10 * dx)
        ax.set_ylim(costs_ef.min() - 0.10 * dy, costs_ef.max() + 0.25 * dy)
        ax.set_xlabel("σ[IS]  —  Realised risk", fontsize=9)
        ax.set_ylabel("E[IS]  —  Realised mean cost", fontsize=9)
        ax.set_title("Analytical frontier vs simulated strategies", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8, loc="upper left",
                  framealpha=0.88, edgecolor="#cccccc")
        ax.grid(True, alpha=0.20, linestyle="--", color="#9E9E9E")
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        self.cv_mc_frontier.draw()

    def _update_mc_table(self, results: dict) -> None:
        hdr = (
            f"{'Strategy':<22}"
            f"{'E[IS]':>14}  {'σ[IS]':>14}  {'VaR 95%':>14}  {'CVaR 95%':>14}"
        )
        sep = "─" * len(hdr)
        rows = [hdr, sep]
        for r in results.values():
            rows.append(
                f"{r.strategy:<22}"
                f"{r.mean_cost:>14,.0f}  {r.std_cost:>14,.0f}  "
                f"{r.var_95:>14,.0f}  {r.cvar_95:>14,.0f}"
            )
        self.lbl_mc_table.setText("\n".join(rows))


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Slightly greyed palette to soften the pure-white background
    pal = app.palette()
    pal.setColor(pal.ColorRole.Window,      pal.color(pal.ColorRole.Window).lighter(103))
    app.setPalette(pal)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
