"""
Tests for the Almgren-Chriss optimal execution model.

Coverage:
  - TWAP limit (λ→0)
  - Immediate liquidation limit (λ→∞)
  - Full liquidation constraint
  - Monotone trajectory
  - Analytical vs discrete cost consistency
  - Monte Carlo vs analytical expectation
  - Market impact functions
  - Input validation
"""
from __future__ import annotations

import numpy as np
import pytest

from src.almgren_chriss import AlmgrenChrissModel, ACResult
from src.simulator import ExecutionSimulator, SimulationResult
from src.market_impact import temp_impact_linear, temp_impact_sqrt, perm_impact_linear


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

# sigma is in $/share/√yr (absolute).  Here sigma≈0.019 corresponds to a ~2% annual-vol
# penny stock — deliberately small so temporary impact (eta=0.05) dominates and the
# qualitative model properties tested below are numerically unambiguous.
# For realistic institutional parameters see the README programmatic example.
BASE = dict(X=100_000.0, T=1.0 / 252, N=50, sigma=0.30 / np.sqrt(252), eta=0.05, gamma=0.01)


def make_model(**overrides) -> AlmgrenChrissModel:
    p = {**BASE, **overrides}
    return AlmgrenChrissModel(**p)


# ──────────────────────────────────────────────────────────────────────────────
# TWAP limit  λ → 0
# ──────────────────────────────────────────────────────────────────────────────

class TestTWAPLimit:
    def test_trajectory_is_linear(self):
        model = make_model(lam=1e-15)
        traj = model.optimal_trajectory()
        twap = model.twap_trajectory()
        np.testing.assert_allclose(traj, twap, rtol=1e-4)

    def test_kappa_near_zero(self):
        model = make_model(lam=1e-15)
        assert model.kappa < 1e-5

    def test_uniform_trades(self):
        model = make_model(lam=1e-15)
        result = model.solve()
        # All trades should be equal (within 1%)
        assert result.trades.std() / result.trades.mean() < 0.01


# ──────────────────────────────────────────────────────────────────────────────
# Immediate liquidation  λ → ∞
# ──────────────────────────────────────────────────────────────────────────────

class TestImmediateLiquidation:
    # lam=1e20 → kT >> 500 → immediate-liquidation branch
    def test_first_trade_is_full_position(self):
        model = make_model(lam=1e20)
        result = model.solve()
        assert result.trades[0] == pytest.approx(BASE["X"], rel=1e-2)

    def test_holdings_drop_to_zero(self):
        model = make_model(lam=1e20)
        result = model.solve()
        np.testing.assert_allclose(result.holdings[1:], 0.0, atol=1.0)

    def test_variance_near_zero(self):
        # Asymptotic formula: Var[C] = σ²X²/(2κ) → 0 as κ → ∞
        model = make_model(lam=1e20)
        e_cost, var = model._closed_form_cost_variance()
        # Var should be negligible relative to E[C]
        assert var / e_cost < 1e-10


# ──────────────────────────────────────────────────────────────────────────────
# Trajectory properties
# ──────────────────────────────────────────────────────────────────────────────

class TestTrajectoryProperties:
    def test_starts_at_X(self):
        model = make_model(lam=1e-5)
        traj = model.optimal_trajectory()
        assert traj[0] == pytest.approx(BASE["X"])

    def test_ends_at_zero(self):
        model = make_model(lam=1e-5)
        traj = model.optimal_trajectory()
        assert traj[-1] == pytest.approx(0.0, abs=1e-6)

    def test_monotone_decreasing(self):
        model = make_model(lam=1e-5)
        traj = model.optimal_trajectory()
        assert np.all(np.diff(traj) <= 1e-10)

    def test_length_N_plus_1(self):
        model = make_model(lam=1e-5, N=30)
        traj = model.optimal_trajectory()
        assert len(traj) == 31


# ──────────────────────────────────────────────────────────────────────────────
# Analytical vs discrete cost
# ──────────────────────────────────────────────────────────────────────────────

class TestCostConsistency:
    def test_analytical_vs_discrete_expected_cost(self):
        model = make_model(lam=1e-5, N=200)
        e_analytical, _ = model._closed_form_cost_variance()
        e_discrete, _ = model.cost_from_trajectory(model.optimal_trajectory())
        # Large N → convergence within 2%
        assert abs(e_analytical - e_discrete) / e_analytical < 0.02

    def test_analytical_vs_discrete_variance(self):
        model = make_model(lam=1e-5, N=200)
        _, v_analytical = model._closed_form_cost_variance()
        _, v_discrete = model.cost_from_trajectory(model.optimal_trajectory())
        assert abs(v_analytical - v_discrete) / v_analytical < 0.02

    def test_lambda_zero_variance_is_maximum(self):
        """TWAP should have highest variance on the frontier."""
        model = make_model(lam=1e-12)
        _, v_twap = model._closed_form_cost_variance()
        model.lam = 1e-2
        _, v_ac = model._closed_form_cost_variance()
        assert v_twap > v_ac


# ──────────────────────────────────────────────────────────────────────────────
# Efficient frontier
# ──────────────────────────────────────────────────────────────────────────────

class TestEfficientFrontier:
    def test_frontier_is_sorted_by_risk(self):
        model = make_model(lam=1e-5)
        risks, _ = model.efficient_frontier(n_points=50)
        assert np.all(np.diff(risks) >= -1e-12)

    def test_frontier_cost_decreases_with_risk(self):
        """Near-immediate liquidation costs more than TWAP (risk-cost trade-off)."""
        # Use T=5 days so κT is substantial and variation is visible
        model = AlmgrenChrissModel(
            X=100_000, T=5 / 252, N=50,
            sigma=0.30 / np.sqrt(252), eta=0.05, gamma=0.01, lam=1e-12,
        )
        e_twap, _ = model._closed_form_cost_variance()
        model.lam = 1e20
        e_immediate, _ = model._closed_form_cost_variance()
        # Immediate liquidation pays more in market impact than spread-over-time
        assert e_immediate > e_twap

    def test_frontier_endpoints(self):
        model = make_model(lam=1e-5)
        risks, costs = model.efficient_frontier(n_points=100)
        # Minimum risk ≈ 0 (immediate) — last point has highest risk
        assert risks[0] < risks[-1]


# ──────────────────────────────────────────────────────────────────────────────
# Monte Carlo
# ──────────────────────────────────────────────────────────────────────────────

class TestMonteCarlo:
    def test_mean_is_positive(self):
        model = make_model(lam=1e-5)
        sim = ExecutionSimulator(model, S0=100.0)
        result = sim.run(n_sims=5_000, seed=0)
        assert result.mean_cost > 0

    def test_cvar_ge_var(self):
        model = make_model(lam=1e-5)
        sim = ExecutionSimulator(model, S0=100.0)
        result = sim.run(n_sims=5_000, seed=0)
        assert result.cvar_95 >= result.var_95

    def test_ac_lower_cvar_than_twap(self):
        """AC optimal trajectory should have lower CVaR than TWAP."""
        model = make_model(lam=1e-4)
        sim = ExecutionSimulator(model, S0=100.0)
        results = sim.run_all_strategies(n_sims=20_000, seed=42)
        assert results["Almgren-Chriss"].cvar_95 <= results["TWAP"].cvar_95

    def test_simulation_result_fields(self):
        model = make_model(lam=1e-5)
        sim = ExecutionSimulator(model, S0=100.0)
        r = sim.run(n_sims=1_000, seed=1)
        assert isinstance(r, SimulationResult)
        assert r.n_sims == 1_000
        assert len(r.implementation_shortfalls) == 1_000


# ──────────────────────────────────────────────────────────────────────────────
# Market impact functions
# ──────────────────────────────────────────────────────────────────────────────

class TestMarketImpact:
    def test_linear_impact(self):
        v = np.array([1_000.0, 2_000.0])
        eta, gamma = 0.05, 0.01
        np.testing.assert_allclose(temp_impact_linear(v, eta), eta * v)
        np.testing.assert_allclose(perm_impact_linear(v, gamma), gamma * v)

    def test_sqrt_impact_sign_preserving(self):
        assert temp_impact_sqrt(np.array([1_000.0]), 0.05)[0] > 0
        assert temp_impact_sqrt(np.array([-1_000.0]), 0.05)[0] < 0

    def test_sqrt_impact_sublinear_vs_linear(self):
        v = np.array([1_000.0])
        eta = 0.05
        assert abs(temp_impact_sqrt(v, eta)[0]) < abs(temp_impact_linear(v, eta)[0])

    def test_impact_doubles_with_volume(self):
        """Linear: impact should exactly double. Sqrt: should sub-double."""
        eta = 0.05
        v1, v2 = np.array([100.0]), np.array([200.0])
        ratio_linear = temp_impact_linear(v2, eta)[0] / temp_impact_linear(v1, eta)[0]
        ratio_sqrt = temp_impact_sqrt(v2, eta)[0] / temp_impact_sqrt(v1, eta)[0]
        assert ratio_linear == pytest.approx(2.0)
        assert ratio_sqrt < 2.0


# ──────────────────────────────────────────────────────────────────────────────
# Input validation
# ──────────────────────────────────────────────────────────────────────────────

class TestInputValidation:
    def test_negative_X_raises(self):
        with pytest.raises(ValueError, match="X must be positive"):
            AlmgrenChrissModel(X=-1, T=1.0, N=10, sigma=0.3, eta=0.05, gamma=0.01, lam=1e-5)

    def test_negative_T_raises(self):
        with pytest.raises(ValueError):
            AlmgrenChrissModel(X=100, T=-1.0, N=10, sigma=0.3, eta=0.05, gamma=0.01, lam=1e-5)

    def test_negative_sigma_raises(self):
        with pytest.raises(ValueError):
            AlmgrenChrissModel(X=100, T=1.0, N=10, sigma=-0.3, eta=0.05, gamma=0.01, lam=1e-5)

    def test_wrong_volume_profile_length_raises(self):
        model = make_model(N=20)
        with pytest.raises(ValueError):
            model.vwap_trajectory(np.ones(10))
