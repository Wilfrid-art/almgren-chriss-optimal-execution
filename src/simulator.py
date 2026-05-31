"""
Monte Carlo simulator for the Almgren-Chriss execution model.

Price dynamics (discrete, per step k = 1…N):
    S_k = S_{k-1} − γ·v_{k-1}·τ + σ·√τ·W_k

Execution price at step k (temporary impact reduces received price):
    P_k = S_{k-1} − η·v_{k-1}

Implementation Shortfall:
    IS = S_0·X − Σ_k P_k · n_k
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .almgren_chriss import AlmgrenChrissModel


@dataclass
class SimulationResult:
    """Aggregated output from a Monte Carlo run."""

    strategy: str
    n_sims: int
    implementation_shortfalls: np.ndarray  # IS for each simulated path
    mean_cost: float
    std_cost: float
    var_95: float    # VaR at 95th percentile
    cvar_95: float   # Expected Shortfall (CVaR) at 95%
    price_paths: Optional[np.ndarray] = None  # shape (n_kept, N+1) — mid-price paths

    @classmethod
    def from_is(
        cls,
        strategy: str,
        IS: np.ndarray,
        price_paths: Optional[np.ndarray] = None,
    ) -> "SimulationResult":
        sorted_IS = np.sort(IS)
        idx_95 = int(np.ceil(0.95 * len(IS))) - 1
        tail = sorted_IS[idx_95:]
        return cls(
            strategy=strategy,
            n_sims=len(IS),
            implementation_shortfalls=IS,
            mean_cost=float(np.mean(IS)),
            std_cost=float(np.std(IS)),
            var_95=float(sorted_IS[idx_95]),
            cvar_95=float(tail.mean()),
            price_paths=price_paths,
        )

    def summary(self) -> str:
        return (
            f"{self.strategy:<20s} | "
            f"E[IS]={self.mean_cost:>12,.1f} | "
            f"σ[IS]={self.std_cost:>12,.1f} | "
            f"VaR95={self.var_95:>12,.1f} | "
            f"CVaR95={self.cvar_95:>12,.1f}"
        )


class ExecutionSimulator:
    """
    Vectorised Monte Carlo simulator for execution strategies.

    Supports three built-in strategies:
      - Almgren-Chriss optimal trajectory
      - TWAP (uniform)
      - VWAP (U-shaped volume profile)
    """

    def __init__(
        self,
        model: "AlmgrenChrissModel",
        S0: float = 100.0,
    ) -> None:
        self.model = model
        self.S0 = S0

    # ──────────────────────────────── Core engine ─────────────────────────────

    def _run_mc(
        self,
        holdings: np.ndarray,
        n_sims: int,
        seed: Optional[int],
        keep_paths: int = 0,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Core MC engine.  Returns (IS, S_mid[:keep_paths]) or (IS, None).

        keep_paths > 0 → store the first keep_paths price trajectories
        for downstream visualisation.
        """
        m = self.model
        N = len(holdings) - 1
        tau = m.T / N

        n = holdings[:-1] - holdings[1:]   # trades  (N,)
        v = n / tau                         # rates   (N,)

        rng = np.random.default_rng(seed)
        W = rng.standard_normal((n_sims, N))

        delta_S = -m.gamma * v[np.newaxis, :] * tau + m.sigma * np.sqrt(tau) * W
        S_mid = np.empty((n_sims, N + 1))
        S_mid[:, 0] = self.S0
        np.cumsum(delta_S, axis=1, out=S_mid[:, 1:])
        S_mid[:, 1:] += self.S0

        exec_prices = S_mid[:, :-1] - m.eta * v[np.newaxis, :]
        proceeds = (exec_prices * n[np.newaxis, :]).sum(axis=1)
        IS = self.S0 * holdings[0] - proceeds

        paths = S_mid[:keep_paths].copy() if keep_paths > 0 else None
        return IS, paths

    def _simulate_is(
        self,
        holdings: np.ndarray,
        n_sims: int,
        seed: Optional[int],
    ) -> np.ndarray:
        """Vectorised IS computation. Returns IS array of shape (n_sims,)."""
        IS, _ = self._run_mc(holdings, n_sims, seed)
        return IS

    # ──────────────────────────────── Public API ──────────────────────────────

    def run(
        self,
        holdings: Optional[np.ndarray] = None,
        n_sims: int = 10_000,
        seed: Optional[int] = 42,
        strategy_name: str = "Almgren-Chriss",
    ) -> SimulationResult:
        """
        Simulate IS distribution for a single holdings trajectory.
        If holdings is None, uses the optimal AC trajectory.
        """
        if holdings is None:
            holdings = self.model.optimal_trajectory()
        IS = self._simulate_is(holdings, n_sims, seed)
        return SimulationResult.from_is(strategy_name, IS)

    def run_all_strategies(
        self,
        n_sims: int = 10_000,
        seed: Optional[int] = 42,
        volume_profile: Optional[np.ndarray] = None,
    ) -> dict[str, SimulationResult]:
        """
        Simulate Almgren-Chriss, TWAP, and VWAP side by side.
        Same random seed → directly comparable paths.
        """
        results: dict[str, SimulationResult] = {}

        trajectories = {
            "Almgren-Chriss": self.model.optimal_trajectory(),
            "TWAP": self.model.twap_trajectory(),
            "VWAP": self.model.vwap_trajectory(volume_profile),
        }
        for name, traj in trajectories.items():
            IS = self._simulate_is(traj, n_sims, seed)
            results[name] = SimulationResult.from_is(name, IS)

        return results

    def run_all_strategies_with_paths(
        self,
        n_sims: int = 10_000,
        seed: Optional[int] = 42,
        volume_profile: Optional[np.ndarray] = None,
        n_paths_keep: int = 2_000,
    ) -> dict[str, SimulationResult]:
        """
        Like run_all_strategies but stores price paths in each result.

        n_paths_keep : number of S_mid paths to store per strategy for
                       visualisation (capped to n_sims).
        """
        keep = min(n_sims, n_paths_keep)
        results: dict[str, SimulationResult] = {}
        trajectories = {
            "Almgren-Chriss": self.model.optimal_trajectory(),
            "TWAP": self.model.twap_trajectory(),
            "VWAP": self.model.vwap_trajectory(volume_profile),
        }
        for name, traj in trajectories.items():
            IS, paths = self._run_mc(traj, n_sims, seed, keep_paths=keep)
            results[name] = SimulationResult.from_is(name, IS, price_paths=paths)
        return results

    def print_summary(self, results: dict[str, SimulationResult]) -> None:
        header = (
            f"{'Strategy':<20s} | {'E[IS]':>14} | {'σ[IS]':>14} | "
            f"{'VaR95':>14} | {'CVaR95':>14}"
        )
        print(header)
        print("─" * len(header))
        for r in results.values():
            print(r.summary())
