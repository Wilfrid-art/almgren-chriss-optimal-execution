"""
Market impact models: temporary and permanent impact functions + calibration.

Temporary impact h(v): captures the bid-ask spread widening and short-term
liquidity absorption when trading at rate v.

Permanent impact g(v): lasting shift in the equilibrium price from informed
trading or supply/demand pressure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Functional forms
# ──────────────────────────────────────────────────────────────────────────────

def temp_impact_linear(v: np.ndarray, eta: float) -> np.ndarray:
    """Linear temporary impact: h(v) = η·v."""
    return eta * np.asarray(v)


def temp_impact_sqrt(
    v: np.ndarray, eta: float, beta: float = 0.6
) -> np.ndarray:
    """
    Power-law temporary impact: h(v) = η · sign(v) · |v|^β.

    β=0.6 is the empirically-validated exponent (Almgren et al. 2005).
    Sublinear in volume: large orders have proportionally smaller impact per share.
    """
    v = np.asarray(v)
    return eta * np.sign(v) * np.abs(v) ** beta


def perm_impact_linear(v: np.ndarray, gamma: float) -> np.ndarray:
    """Linear permanent impact: g(v) = γ·v."""
    return gamma * np.asarray(v)


# ──────────────────────────────────────────────────────────────────────────────
# Calibration
# ──────────────────────────────────────────────────────────────────────────────

class MarketImpactCalibrator:
    """
    Calibrate η (temporary) and γ (permanent) from price/volume history.

    Permanent impact: linear regression  ΔS_t = γ·(V_t/ADV) + ε
    Temporary impact: power-law fit      slip_t = η·(V_t/ADV)^β

    where ADV = Average Daily Volume.
    """

    def __init__(
        self,
        adv: float,
        beta: float = 0.6,
    ) -> None:
        """
        Parameters
        ----------
        adv  : average daily volume (shares/day)
        beta : power-law exponent for temporary impact
        """
        self.adv = adv
        self.beta = beta
        self.eta_hat: Optional[float] = None
        self.gamma_hat: Optional[float] = None

    def calibrate_permanent(
        self,
        volumes_signed: np.ndarray,
        price_changes: np.ndarray,
    ) -> float:
        """
        OLS: ΔS = γ·(V/ADV) → returns γ̂ (clamped ≥ 0).

        Parameters
        ----------
        volumes_signed : signed volumes (+buy, −sell) per bar
        price_changes  : price return per bar (same length)
        """
        x = (volumes_signed / self.adv).reshape(-1, 1)
        coef, *_ = np.linalg.lstsq(x, price_changes, rcond=None)
        self.gamma_hat = float(max(coef[0], 0.0))
        return self.gamma_hat

    def calibrate_temporary(
        self,
        volumes: np.ndarray,
        slippages: np.ndarray,
    ) -> float:
        """
        Nonlinear fit: slippage = η·(V/ADV)^β → returns η̂.

        Parameters
        ----------
        volumes    : unsigned volumes per bar
        slippages  : execution slippage per bar (price units, ≥ 0)
        """
        x = volumes / self.adv
        mask = (x > 0) & (slippages >= 0)
        x, slippages = x[mask], slippages[mask]

        def _model(x_: np.ndarray, eta: float) -> np.ndarray:
            return eta * x_**self.beta

        popt, _ = curve_fit(
            _model, x, slippages, p0=[0.1], bounds=(0, np.inf), maxfev=10_000
        )
        self.eta_hat = float(popt[0])
        return self.eta_hat

    def calibrate_from_ohlcv(self, df: pd.DataFrame) -> dict[str, float]:
        """
        Full calibration from a OHLCV DataFrame (daily bars).
        Columns required: open, close, volume.

        Returns dict with keys: eta, gamma, adv.
        """
        prices = df["close"].values.astype(float)
        opens = df["open"].values.astype(float)
        volumes = df["volume"].values.astype(float)

        delta_S = np.diff(prices)
        # Intrabar direction proxy avoids signing volumes by the price change we
        # are trying to predict (which would make gamma trivially positive).
        bar_dir = np.sign(prices[1:] - opens[1:])
        bar_dir = np.where(bar_dir == 0, 1.0, bar_dir)
        v_signed = volumes[1:] * bar_dir
        self.calibrate_permanent(v_signed, delta_S)

        # Use |close − open| / open as a coarse intraday slippage proxy.
        slippage = np.abs(prices - opens) / opens
        self.calibrate_temporary(volumes, slippage)

        return {
            "eta": self.eta_hat,
            "gamma": self.gamma_hat,
            "adv": self.adv,
        }
