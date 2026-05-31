"""
Almgren-Chriss (2001) optimal execution model — closed-form solution.

Reference:
    Almgren, R. & Chriss, N. (2001). Optimal Execution of Portfolio Transactions.
    Journal of Risk, 3, 5-40.

Units convention: T and sigma must be in the same time unit.
  - If T is in years  → sigma is annualized vol.
  - If T is in days   → sigma is daily vol (= annual / sqrt(252)).
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class ACResult:
    """Full output of a model solve."""

    times: np.ndarray       # t_0 … t_N  (length N+1)
    holdings: np.ndarray    # x*(t_j) shares held at each grid point (N+1)
    trades: np.ndarray      # n_j = x_{j-1} - x_j  shares sold per step (N)
    rates: np.ndarray       # v_j = n_j / τ  trading rate (N)
    expected_cost: float    # E[C]   analytical
    cost_variance: float    # Var[C] analytical
    kappa: float            # decay parameter κ


class AlmgrenChrissModel:
    """
    Almgren-Chriss optimal liquidation model.

    Minimises: U = E[C] + λ · Var[C]

    Analytical solution (continuous-time limit):
        x*(t) = X · sinh(κ(T−t)) / sinh(κT)
        v*(t) = X · κ · cosh(κ(T−t)) / sinh(κT)
        κ     = √(λσ²/η)

    Expected cost (Proposition 3.1 in the paper):
        E[C] = (γ/2)X² + η X² κ [sinh(κT)cosh(κT) + κT] / (2 sinh²(κT))

    Variance of cost:
        Var[C] = σ² X² [sinh(κT)cosh(κT) − κT] / (2κ sinh²(κT))

    Parameters
    ----------
    X     : initial position (shares to liquidate).
    T     : liquidation horizon (same time unit as sigma).
    N     : number of execution intervals.
    sigma : asset volatility per unit time.
    eta   : temporary market impact coefficient (price·time/share).
    gamma : permanent market impact coefficient (price/share).
    lam   : risk-aversion λ ≥ 0.  0 → TWAP,  +∞ → immediate.
    """

    def __init__(
        self,
        X: float,
        T: float,
        N: int,
        sigma: float,
        eta: float,
        gamma: float,
        lam: float = 1e-6,
    ) -> None:
        if X <= 0:
            raise ValueError("X must be positive")
        if T <= 0:
            raise ValueError("T must be positive")
        if N < 1:
            raise ValueError("N must be >= 1")
        if any(v < 0 for v in (sigma, eta, gamma, lam)):
            raise ValueError("sigma, eta, gamma, lam must be non-negative")

        self.X = float(X)
        self.T = float(T)
        self.N = int(N)
        self.sigma = float(sigma)
        self.eta = float(eta)
        self.gamma = float(gamma)
        self.lam = float(lam)

    @property
    def tau(self) -> float:
        """Time step τ = T/N."""
        return self.T / self.N

    @property
    def kappa(self) -> float:
        """Decay parameter κ = √(λσ²/η). Returns 0 when η=0 or λ=0."""
        if self.eta == 0.0 or self.lam == 0.0:
            return 0.0
        return float(np.sqrt(self.lam * self.sigma**2 / self.eta))

    # ──────────────────────────────── Trajectories ────────────────────────────

    def optimal_trajectory(self) -> np.ndarray:
        """
        Holdings x*(t_j) = X · sinh(κ(T−t_j)) / sinh(κT),  j = 0…N.

        Limits:
          κ→0  (λ→0):   linear / TWAP   x*(t) = X(1 − t/T)
          κ→∞  (λ→∞):   immediate        x*(0) = X, x*(t>0) = 0
        """
        kappa = self.kappa
        times = np.linspace(0.0, self.T, self.N + 1)
        kT = kappa * self.T

        if kT < 1e-8:
            return self.X * (1.0 - times / self.T)
        if kT > 700.0:
            # sinh overflows float64 for arg > ~710
            traj = np.zeros(self.N + 1)
            traj[0] = self.X
            return traj

        return self.X * np.sinh(kappa * (self.T - times)) / np.sinh(kT)

    def optimal_trading_rate(self) -> np.ndarray:
        """
        Trading rate v*(t_j) = Xκ · cosh(κ(T−t_j)) / sinh(κT),  j = 0…N.
        """
        kappa = self.kappa
        times = np.linspace(0.0, self.T, self.N + 1)
        kT = kappa * self.T

        if kT < 1e-8:
            return np.full(self.N + 1, self.X / self.T)
        if kT > 700.0:
            rates = np.zeros(self.N + 1)
            rates[0] = self.X / self.tau
            return rates

        return self.X * kappa * np.cosh(kappa * (self.T - times)) / np.sinh(kT)

    def twap_trajectory(self) -> np.ndarray:
        """Uniform holdings: x(t_j) = X(1 − j/N)."""
        return self.X * np.linspace(1.0, 0.0, self.N + 1)

    def vwap_trajectory(
        self, volume_profile: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        VWAP trajectory: liquidate proportional to intraday volume profile.
        Default: U-shaped profile (heavier volume at open and close).
        """
        if volume_profile is None:
            t = np.linspace(0.0, 1.0, self.N)
            v = 1.5 - np.cos(2.0 * np.pi * t)
            volume_profile = v / v.sum()

        if len(volume_profile) != self.N:
            raise ValueError(f"volume_profile must have length N={self.N}")

        holdings = np.empty(self.N + 1)
        holdings[0] = self.X
        holdings[1:] = self.X - self.X * np.cumsum(volume_profile)
        holdings[-1] = 0.0
        return holdings

    # ──────────────────────────────── Cost / Variance ─────────────────────────

    def solve(self) -> ACResult:
        """Compute optimal trajectory and analytical E[C], Var[C]."""
        times = np.linspace(0.0, self.T, self.N + 1)
        holdings = self.optimal_trajectory()
        trades = holdings[:-1] - holdings[1:]
        rates = trades / self.tau
        e_cost, var_cost = self._closed_form_cost_variance()

        return ACResult(
            times=times,
            holdings=holdings,
            trades=trades,
            rates=rates,
            expected_cost=e_cost,
            cost_variance=var_cost,
            kappa=self.kappa,
        )

    def _closed_form_cost_variance(self) -> tuple[float, float]:
        """
        Analytical E[C] and Var[C] for the optimal trajectory.

        E[C]   = (γ/2)X² + η X² κ (sinh(κT)cosh(κT) + κT) / (2 sinh²(κT))
        Var[C] = σ² X²   (sinh(κT)cosh(κT) − κT)          / (2κ sinh²(κT))
        """
        kappa = self.kappa
        X, T = self.X, self.T
        sigma, eta, gamma = self.sigma, self.eta, self.gamma
        kT = kappa * T

        if kT < 1e-8:
            # Taylor expansion κ→0 (TWAP)
            e_cost = 0.5 * gamma * X**2 + eta * X**2 / T
            var_cost = sigma**2 * X**2 * T / 3.0
            return float(e_cost), float(var_cost)

        if kT > 100.0:
            # For kT > ~100, cosh/sinh → 1 and kT/sinh² → 0 to machine precision.
            # The intermediate products sinh·cosh overflow float64 around kT≈352;
            # using the exact asymptotic here avoids the overflow with zero loss
            # in accuracy (relative error < 1e-43).
            #   (sinh·cosh + kT) / sinh² → 1   ⇒  E[C] → γX²/2 + ηX²κ/2
            #   (sinh·cosh − kT) / sinh² → 1   ⇒  Var[C] → σ²X²/(2κ)
            e_cost = 0.5 * gamma * X**2 + 0.5 * eta * X**2 * kappa
            var_cost = sigma**2 * X**2 / (2.0 * kappa)
            return float(e_cost), float(var_cost)

        sinh_kT = np.sinh(kT)
        cosh_kT = np.cosh(kT)
        sinh2_kT = sinh_kT**2

        e_cost = (
            0.5 * gamma * X**2
            + eta * X**2 * kappa * (sinh_kT * cosh_kT + kT) / (2.0 * sinh2_kT)
        )
        var_cost = (
            sigma**2 * X**2 * (sinh_kT * cosh_kT - kT) / (2.0 * kappa * sinh2_kT)
        )
        return float(e_cost), float(max(var_cost, 0.0))

    def cost_from_trajectory(
        self, holdings: np.ndarray
    ) -> tuple[float, float]:
        """
        Discrete-time E[C] and Var[C] for any holdings schedule.

        E[C]   = (γ/2)X² + (η/τ) Σ n_j²
        Var[C] = σ² τ Σ x_{j-1}²
        """
        n = holdings[:-1] - holdings[1:]
        e_cost = 0.5 * self.gamma * self.X**2 + (self.eta / self.tau) * float(np.sum(n**2))
        var_cost = float(self.sigma**2 * self.tau * np.sum(holdings[:-1]**2))
        return float(e_cost), max(var_cost, 0.0)

    # ──────────────────────────────── Efficient Frontier ──────────────────────

    def efficient_frontier(
        self, n_points: int = 200
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Efficient frontier in (√Var[C], E[C]) space by sweeping λ ∈ [0, +∞).

        Returns
        -------
        risks : √Var[C] values (sorted ascending)
        costs : E[C] values
        """
        lambdas = np.concatenate([[0.0], np.logspace(-9, 15, n_points - 1)])
        costs = np.empty(n_points)
        risks = np.empty(n_points)

        for i, lam in enumerate(lambdas):
            tmp = AlmgrenChrissModel(
                self.X, self.T, self.N, self.sigma, self.eta, self.gamma, float(lam)
            )
            e_cost, var_cost = tmp._closed_form_cost_variance()
            costs[i] = e_cost
            risks[i] = np.sqrt(var_cost)

        idx = np.argsort(risks)
        return risks[idx], costs[idx]
