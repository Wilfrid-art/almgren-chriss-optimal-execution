"""
Extensions to the base Almgren-Chriss model:
  - Piecewise (regime-switching) volatility schedule
  - Intraday participation-rate constraint
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from typing import Optional


class VolatilityRegimeModel:
    """
    Almgren-Chriss with step-wise σ(t): no closed-form → numerical optimisation.

    Minimises: E[C] + λ·Var[C] subject to Σ n_k = X, n_k ≥ 0.

    E[C]   = (γ/2)X² + (η/τ) Σ n_k²
    Var[C] = τ Σ σ_k² · x_{k-1}²
    """

    def __init__(
        self,
        X: float,
        T: float,
        N: int,
        sigma_schedule: np.ndarray,
        eta: float,
        gamma: float,
        lam: float,
    ) -> None:
        if len(sigma_schedule) != N:
            raise ValueError(f"sigma_schedule must have length N={N}")
        self.X = X
        self.T = T
        self.N = N
        self.sigma = np.asarray(sigma_schedule, dtype=float)
        self.eta = eta
        self.gamma = gamma
        self.lam = lam
        self.tau = T / N

    def _objective(self, trades: np.ndarray) -> float:
        holdings = np.concatenate([[self.X], self.X - np.cumsum(trades)])
        e_cost = (
            0.5 * self.gamma * self.X**2
            + (self.eta / self.tau) * float(np.dot(trades, trades))
        )
        var_cost = float(
            self.tau * np.dot(self.sigma**2, holdings[:-1] ** 2)
        )
        return e_cost + self.lam * var_cost

    def optimal_trajectory(self) -> np.ndarray:
        """Returns holdings array of length N+1."""
        n0 = np.full(self.N, self.X / self.N)
        result = minimize(
            self._objective,
            n0,
            method="SLSQP",
            bounds=[(0.0, self.X)] * self.N,
            constraints=[{"type": "eq", "fun": lambda n: n.sum() - self.X}],
            options={"ftol": 1e-12, "maxiter": 2000},
        )
        if not result.success:
            raise RuntimeError(f"SLSQP did not converge: {result.message}")
        return np.concatenate([[self.X], self.X - np.cumsum(result.x)])


class IntradayConstrainedModel:
    """
    Almgren-Chriss with a maximum participation rate:  n_k ≤ pmax · V_k.

    This prevents the trader from taking more than pmax fraction of each
    interval's expected volume, limiting market impact on thinly-traded bars.
    """

    def __init__(
        self,
        X: float,
        T: float,
        N: int,
        sigma: float,
        eta: float,
        gamma: float,
        lam: float,
        volume_profile: np.ndarray,
        pmax: float = 0.10,
    ) -> None:
        if len(volume_profile) != N:
            raise ValueError(f"volume_profile must have length N={N}")
        self.X = X
        self.T = T
        self.N = N
        self.sigma = sigma
        self.eta = eta
        self.gamma = gamma
        self.lam = lam
        self.volume_profile = np.asarray(volume_profile, dtype=float)
        self.pmax = pmax
        self.tau = T / N

    def optimal_trajectory(self) -> np.ndarray:
        """Returns holdings array of length N+1."""
        tau = self.tau

        def objective(trades: np.ndarray) -> float:
            holdings = np.concatenate([[self.X], self.X - np.cumsum(trades)])
            e_cost = (
                0.5 * self.gamma * self.X**2
                + (self.eta / tau) * float(np.dot(trades, trades))
            )
            var_cost = self.sigma**2 * tau * float(np.sum(holdings[:-1] ** 2))
            return e_cost + self.lam * var_cost

        upper = self.pmax * self.volume_profile
        n0 = np.clip(np.full(self.N, self.X / self.N), 0.0, upper)

        result = minimize(
            objective,
            n0,
            method="SLSQP",
            bounds=[(0.0, float(u)) for u in upper],
            constraints=[{"type": "eq", "fun": lambda n: n.sum() - self.X}],
            options={"ftol": 1e-12, "maxiter": 2000},
        )
        if not result.success:
            raise RuntimeError(f"SLSQP did not converge: {result.message}")
        return np.concatenate([[self.X], self.X - np.cumsum(result.x)])


def two_regime_sigma(
    N: int,
    sigma_low: float,
    sigma_high: float,
    switch_step: int,
) -> np.ndarray:
    """
    Build a two-regime σ schedule.

    Steps [0, switch_step) → sigma_low
    Steps [switch_step, N) → sigma_high
    """
    if not (0 < switch_step < N):
        raise ValueError("switch_step must be in (0, N)")
    sched = np.full(N, sigma_low)
    sched[switch_step:] = sigma_high
    return sched
