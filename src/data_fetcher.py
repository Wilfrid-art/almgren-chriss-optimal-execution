"""
Market data retrieval (yfinance) and parameter calibration for AC model.

Estimates:
  σ     : annualized realized volatility from log-returns
  η     : temporary impact (from spread and ADV)
  γ     : permanent impact (OLS regression ΔP ~ signed volume)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


def fetch_ohlcv(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Fetch adjusted OHLCV from Yahoo Finance.

    Parameters
    ----------
    ticker   : e.g. "AAPL"
    period   : yfinance period string — "1y", "6mo", "3mo", "1mo"
    interval : "1d", "1h", "30m", "5m", etc.

    Returns
    -------
    DataFrame with lowercase columns: open, high, low, close, volume.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("pip install yfinance") from exc

    raw = yf.download(
        ticker, period=period, interval=interval,
        progress=False, auto_adjust=True,
    )
    if raw.empty:
        raise ValueError(f"No data returned for ticker={ticker!r}")

    # yfinance ≥0.2.31 returns MultiIndex columns even for single tickers
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw.columns = [c.lower() for c in raw.columns]
    return raw.dropna()


def realized_vol(
    prices: pd.Series,
    window: int = 20,
    annualize: bool = True,
) -> pd.Series:
    """Rolling realized volatility from log-returns."""
    log_ret = np.log(prices / prices.shift(1))
    rv = log_ret.rolling(window).std()
    if annualize:
        rv = rv * np.sqrt(252)
    return rv


def estimate_hl_spread(df: pd.DataFrame) -> float:
    """
    Corwin-Schultz (2012) high-low spread estimator.
    Returns average spread as fraction of price.
    """
    beta = (np.log(df["high"] / df["low"]) ** 2).rolling(2).sum().dropna()
    gamma_cs = (
        np.log(
            df["high"].rolling(2).max() / df["low"].rolling(2).min()
        ) ** 2
    ).dropna()
    k = np.sqrt(2) - 1
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / (3.0 - 2.0 * np.sqrt(2)) - np.sqrt(
        gamma_cs / (3.0 - 2.0 * np.sqrt(2))
    )
    spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    return float(spread.clip(lower=0.0).mean())


class MarketDataCalibrator:
    """
    End-to-end calibration of AC parameters from live market data.

    Usage
    -----
    >>> cal = MarketDataCalibrator("AAPL")
    >>> params = cal.fetch_and_calibrate()
    >>> print(params)
    """

    def __init__(self, ticker: str, period: str = "1y") -> None:
        self.ticker = ticker
        self.period = period
        self.df: Optional[pd.DataFrame] = None
        self._params: Optional[dict[str, float]] = None

    def fetch_and_calibrate(self) -> dict[str, float]:
        """Fetch data then run calibration. Returns full parameter dict."""
        self.df = fetch_ohlcv(self.ticker, period=self.period)
        return self.calibrate(self.df)

    def calibrate(self, df: pd.DataFrame) -> dict[str, float]:
        """
        Calibrate σ, η, γ from a OHLCV DataFrame.

        Returns
        -------
        dict with keys: sigma, eta, gamma, adv, spread, price_last, sigma_daily
        """
        prices = df["close"].astype(float)
        volumes = df["volume"].astype(float)

        # ── σ (annualized daily vol) ──────────────────────────────────────────
        if (prices <= 0).any():
            raise ValueError("Non-positive prices detected — check data quality.")
        log_ret = np.log(prices / prices.shift(1)).dropna()
        sigma_daily = float(log_ret.std())
        sigma_annual = sigma_daily * np.sqrt(252)

        # ── ADV ──────────────────────────────────────────────────────────────
        adv = float(volumes.mean())

        # ── Spread (Corwin-Schultz) ───────────────────────────────────────────
        try:
            spread = estimate_hl_spread(df)
        except Exception:
            spread = float(((df["high"] - df["low"]) / df["close"]).mean())

        # ── γ (permanent impact) : OLS  ΔS = γ·(V_signed/ADV) ───────────────
        # Direction proxy: close > open → buying pressure (avoids using delta_S
        # to sign volumes, which would make γ positive by construction).
        delta_S = np.diff(prices.values)
        bar_direction = np.sign(prices.values[1:] - df["open"].values.astype(float)[1:])
        bar_direction = np.where(bar_direction == 0, 1.0, bar_direction)
        V_signed = volumes.values[1:] * bar_direction
        x_perm = (V_signed / adv).reshape(-1, 1)
        gamma_hat = float(np.linalg.lstsq(x_perm, delta_S, rcond=None)[0][0])
        gamma_hat = max(gamma_hat, 0.0)

        # ── η (temporary impact) : heuristic from spread + ADV + vol ─────────
        price_last = float(prices.iloc[-1])
        # Approximation: η ≈ 0.1 · spread_cost / (σ_daily · √ADV).
        # This is a rough heuristic — proper estimation requires tick-level data.
        # Units: [price · day / share] when T is in trading days.
        eta_hat = 0.1 * spread * price_last / (sigma_daily * np.sqrt(adv))
        eta_hat = max(eta_hat, 1e-8)

        self._params = {
            "sigma": sigma_annual,
            "eta": eta_hat,
            "gamma": gamma_hat,
            "adv": adv,
            "spread": spread,
            "price_last": price_last,
            "sigma_daily": sigma_daily,
        }
        return self._params

    def summary(self) -> str:
        if self._params is None:
            return "Not calibrated yet — call fetch_and_calibrate() first."
        p = self._params
        return (
            f"Ticker  : {self.ticker}\n"
            f"σ (ann) : {p['sigma']:.2%}\n"
            f"σ (day) : {p['sigma_daily']:.4%}\n"
            f"η       : {p['eta']:.6f}\n"
            f"γ       : {p['gamma']:.6f}\n"
            f"ADV     : {p['adv']:,.0f} shares\n"
            f"Spread  : {p['spread']:.4%}\n"
            f"Price   : ${p['price_last']:.2f}\n"
        )
