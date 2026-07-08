"""
US Stock Long/Short Scanner
============================
Scans a liquid universe of US stocks (default: S&P 500) and ranks them into
the Top 10 "Long" candidates (strong fundamentals + bullish trend) and
Top 10 "Short" candidates (weak fundamentals + bearish trend).

Data source: Yahoo Finance via `yfinance` (free, no API key required).

Usage:
    python stock_scanner.py
    python stock_scanner.py --universe nasdaq100 --top-n 15
    python stock_scanner.py --tickers AAPL,MSFT,TSLA,NVDA,XOM

See the accompanying setup instructions for environment setup.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# Wikipedia rejects the default urllib user-agent that pd.read_html sends
# under the hood (403 Forbidden), so fetch the HTML ourselves with a
# browser-like User-Agent and hand the markup to read_html directly.
WIKI_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; stock-scanner/1.0)"}

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------

MIN_MARKET_CAP = 2_000_000_000     # $2B liquidity/quality floor
MIN_AVG_VOLUME = 1_000_000         # 1M shares/day liquidity floor
PRICE_HISTORY_PERIOD = "1y"        # enough history for a 200-day SMA
SMA_SHORT, SMA_LONG = 50, 200
RSI_PERIOD = 14

LONG_RSI_BAND = (50, 65)           # strong but not overbought
SHORT_RSI_BAND = (35, 50)          # weak but not oversold

MAX_WORKERS = 5                    # thread pool size for per-ticker .info calls
MIN_REQUEST_INTERVAL = 0.35        # global min seconds between any two Yahoo requests
FUNDAMENTALS_RETRIES = 4           # retries per ticker before giving up
RATE_LIMIT_BACKOFF_BASE = 8.0      # seconds; doubles each retry on a 429

PRICE_CHUNK_SIZE = 40               # tickers per yf.download() batch call
PRICE_CHUNK_PAUSE = 2.0             # seconds between batch calls
PRICE_DOWNLOAD_RETRIES = 3          # retry rounds for tickers still missing data
PRICE_RATE_LIMIT_BACKOFF_BASE = 20.0  # seconds; doubles each retry round


class RateLimiter:
    """Enforces a minimum spacing between requests *across all threads*.

    A per-thread sleep isn't enough to stay under Yahoo Finance's
    undocumented rate limit: with N threads each pausing independently,
    the aggregate request rate is still N times too fast. This serializes
    the *start* of each request behind a shared clock so the whole pool
    respects one global pace.
    """

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._lock = threading.Lock()
        self._last_call = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self._last_call = time.monotonic()


_rate_limiter = RateLimiter(MIN_REQUEST_INTERVAL)


# --------------------------------------------------------------------------
# UNIVERSE BUILDERS
# --------------------------------------------------------------------------

def get_sp500_tickers() -> list[str]:
    """Scrape the current S&P 500 constituent list from Wikipedia.

    Falls back to a small hardcoded liquid-name list if the scrape fails
    (e.g. no internet access, Wikipedia layout change), so the script
    always has something to run against.
    """
    try:
        resp = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers=WIKI_HEADERS, timeout=10,
        )
        resp.raise_for_status()
        tables = pd.read_html(io.StringIO(resp.text))
        df = tables[0]
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        return tickers
    except Exception as exc:
        print(f"[WARN] Could not fetch S&P 500 list from Wikipedia ({exc}). "
              f"Falling back to a small default universe.", file=sys.stderr)
        return [
            "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "JPM",
            "XOM", "UNH", "V", "PG", "HD", "MA", "COST", "PFE", "INTC",
            "KO", "PEP", "DIS", "BA", "WMT", "CVX", "ABBV", "MRK",
        ]


def get_nasdaq100_tickers() -> list[str]:
    """Scrape the current Nasdaq-100 constituent list from Wikipedia."""
    try:
        resp = requests.get(
            "https://en.wikipedia.org/wiki/Nasdaq-100",
            headers=WIKI_HEADERS, timeout=10,
        )
        resp.raise_for_status()
        tables = pd.read_html(io.StringIO(resp.text))
        for t in tables:
            if "Ticker" in t.columns:
                return t["Ticker"].str.replace(".", "-", regex=False).tolist()
            if "Symbol" in t.columns:
                return t["Symbol"].str.replace(".", "-", regex=False).tolist()
        raise ValueError("Ticker column not found in any table")
    except Exception as exc:
        print(f"[WARN] Could not fetch Nasdaq-100 list ({exc}). "
              f"Falling back to S&P 500.", file=sys.stderr)
        return get_sp500_tickers()


# --------------------------------------------------------------------------
# TECHNICAL INDICATORS
# --------------------------------------------------------------------------

def compute_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Wilder's RSI, the standard formulation used by most charting platforms."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)  # neutral RSI when there's no loss/gain history yet


@dataclass
class TechnicalSnapshot:
    price: float
    sma50: float
    sma200: float
    rsi14: float
    avg_volume_3m: float
    dist_from_200sma_pct: float
    dist_from_50sma_pct: float


def compute_technicals(hist: pd.DataFrame) -> Optional[TechnicalSnapshot]:
    """Derive price/trend/momentum stats from a single ticker's OHLCV history."""
    if hist is None or hist.empty or "Close" not in hist:
        return None
    close = hist["Close"].dropna()
    volume = hist["Volume"].dropna()
    if len(close) < SMA_LONG:  # need a full 200-day window to be meaningful
        return None

    sma50 = close.rolling(SMA_SHORT).mean().iloc[-1]
    sma200 = close.rolling(SMA_LONG).mean().iloc[-1]
    rsi14 = compute_rsi(close).iloc[-1]
    price = close.iloc[-1]
    avg_vol_3m = volume.tail(63).mean()  # ~3 trading months

    if pd.isna(sma50) or pd.isna(sma200) or price <= 0:
        return None

    return TechnicalSnapshot(
        price=float(price),
        sma50=float(sma50),
        sma200=float(sma200),
        rsi14=float(rsi14),
        avg_volume_3m=float(avg_vol_3m),
        dist_from_200sma_pct=float((price - sma200) / sma200 * 100),
        dist_from_50sma_pct=float((price - sma50) / sma50 * 100),
    )


# --------------------------------------------------------------------------
# FUNDAMENTAL DATA
# --------------------------------------------------------------------------

@dataclass
class Fundamentals:
    name: str = ""
    sector: str = ""                    # principal business activity (GICS sector)
    industry: str = ""                  # finer-grained industry within the sector
    market_cap: float = np.nan
    pe_ratio: float = np.nan
    eps: float = np.nan
    eps_growth: float = np.nan          # trailing quarterly EPS growth YoY
    revenue_growth: float = np.nan      # quarterly revenue growth YoY
    debt_to_equity: float = np.nan


def fetch_fundamentals(ticker: str, retries: int = FUNDAMENTALS_RETRIES) -> Optional[Fundamentals]:
    """Pull fundamental metrics for one ticker, tolerating missing fields
    and transient API/rate-limit errors.

    Every attempt passes through the shared `_rate_limiter` so the whole
    thread pool paces its requests together. A 429 gets a much longer,
    exponentially growing backoff than other errors since Yahoo's rate
    limit window is on the order of tens of seconds, not milliseconds.
    """
    for attempt in range(retries + 1):
        _rate_limiter.wait()
        try:
            info = yf.Ticker(ticker).get_info()
            return Fundamentals(
                name=info.get("shortName") or info.get("longName") or ticker,
                sector=info.get("sector") or "",
                industry=info.get("industry") or "",
                market_cap=info.get("marketCap", np.nan),
                pe_ratio=info.get("trailingPE", np.nan),
                eps=info.get("trailingEps", np.nan),
                eps_growth=info.get("earningsQuarterlyGrowth", np.nan),
                revenue_growth=info.get("revenueGrowth", np.nan),
                debt_to_equity=info.get("debtToEquity", np.nan),
            )
        except Exception as exc:
            is_rate_limited = "Too Many Requests" in str(exc) or "Rate limited" in str(exc)
            if attempt < retries:
                if is_rate_limited:
                    time.sleep(RATE_LIMIT_BACKOFF_BASE * (2 ** attempt))
                else:
                    time.sleep(0.5 * (attempt + 1))
                continue
            print(f"[WARN] fundamentals failed for {ticker}: {exc}", file=sys.stderr)
            return None


# --------------------------------------------------------------------------
# SCORING
# --------------------------------------------------------------------------

def _clip01(x: float) -> float:
    return float(np.clip(x, 0, 100))


def _fmt_pct(v: float, signed: bool = True) -> str:
    if pd.isna(v):
        return "no data"
    return f"{v * 100:+.1f}%" if signed else f"{v * 100:.1f}%"


def _fmt_num(v: float, decimals: int = 1) -> str:
    return "no data" if pd.isna(v) else f"{v:.{decimals}f}"


def _component(label: str, weight: float, score: float, detail: str) -> dict:
    """One scored ingredient of a composite: what it measures, how much it
    counts, the 0-100 score it earned, and a human-readable why."""
    return {"label": label, "weight": weight, "score": round(_clip01(score), 1), "detail": detail}


def _weighted(components: list[dict]) -> float:
    return _clip01(sum(c["weight"] * c["score"] for c in components))


def fundamental_breakdown_long(f: Fundamentals) -> dict:
    """0-100 with per-component drill-down: rewards revenue growth,
    positive/growing EPS, low leverage, and a reasonable valuation."""
    growth = 0 if pd.isna(f.revenue_growth) else np.clip(f.revenue_growth * 100, -30, 30)
    growth_score = (growth + 30) / 60 * 100  # -30%..+30% -> 0..100

    eps_growth = 0 if pd.isna(f.eps_growth) else np.clip(f.eps_growth * 100, -50, 50)
    eps_score = (eps_growth + 50) / 100 * 100
    eps_detail = f"EPS growth {_fmt_pct(f.eps_growth)}, trailing EPS {_fmt_num(f.eps, 2)}"
    if pd.notna(f.eps) and f.eps <= 0:
        eps_score *= 0.3  # heavily penalize unprofitable companies for "long" quality
        eps_detail += " (unprofitable: score cut 70%)"

    de = 100 if pd.isna(f.debt_to_equity) else f.debt_to_equity
    leverage_score = 100 - np.clip(de, 0, 300) / 300 * 100  # lower D/E -> higher score

    if pd.isna(f.pe_ratio) or f.pe_ratio <= 0:
        valuation_score = 40  # unknown/negative earnings -> mildly penalized, not disqualified
        valuation_detail = "P/E unavailable or negative"
    else:
        # Sweet spot ~10-30x; score decays as PE drifts far from that band.
        valuation_score = _clip01(100 - abs(f.pe_ratio - 20) * 1.5)
        valuation_detail = f"P/E {f.pe_ratio:.1f} vs ~20x sweet spot"

    components = [
        _component("Revenue growth", 0.35, growth_score, f"YoY revenue {_fmt_pct(f.revenue_growth)}"),
        _component("EPS quality", 0.25, eps_score, eps_detail),
        _component("Leverage", 0.20, leverage_score, f"Debt/Equity {_fmt_num(f.debt_to_equity, 0)} (lower is better)"),
        _component("Valuation", 0.20, valuation_score, valuation_detail),
    ]
    return {"score": round(_weighted(components), 1), "components": components}


def fundamental_breakdown_short(f: Fundamentals) -> dict:
    """Mirror of the long score: rewards revenue decline, negative/collapsing
    EPS, high leverage, and stretched/unsustainable valuation."""
    growth = 0 if pd.isna(f.revenue_growth) else np.clip(f.revenue_growth * 100, -30, 30)
    growth_score = 100 - (growth + 30) / 60 * 100  # declining revenue -> high score

    eps_growth = 0 if pd.isna(f.eps_growth) else np.clip(f.eps_growth * 100, -50, 50)
    eps_score = 100 - (eps_growth + 50) / 100 * 100
    eps_detail = f"EPS growth {_fmt_pct(f.eps_growth)}, trailing EPS {_fmt_num(f.eps, 2)}"
    if pd.notna(f.eps) and f.eps <= 0:
        eps_score = min(100, eps_score + 30)  # negative earnings strengthens the short case
        eps_detail += " (unprofitable: +30 to short case)"

    de = 0 if pd.isna(f.debt_to_equity) else f.debt_to_equity
    leverage_score = np.clip(de, 0, 300) / 300 * 100  # higher D/E -> higher score

    if pd.isna(f.pe_ratio) or f.pe_ratio <= 0:
        valuation_score = 60  # no/negative earnings while still priced richly is a red flag
        valuation_detail = "P/E unavailable or negative (red flag)"
    else:
        # Very high multiples (expensive/priced-for-perfection) score higher as short candidates.
        valuation_score = _clip01((f.pe_ratio - 20) * 1.5)
        valuation_detail = f"P/E {f.pe_ratio:.1f}; multiples above ~20x add to the short case"

    components = [
        _component("Revenue weakness", 0.35, growth_score, f"YoY revenue {_fmt_pct(f.revenue_growth)}"),
        _component("EPS deterioration", 0.25, eps_score, eps_detail),
        _component("Leverage risk", 0.20, leverage_score, f"Debt/Equity {_fmt_num(f.debt_to_equity, 0)} (higher is worse)"),
        _component("Valuation stretch", 0.20, valuation_score, valuation_detail),
    ]
    return {"score": round(_weighted(components), 1), "components": components}


def _band_score(value: float, low: float, high: float) -> float:
    """100 at the center of [low, high], decaying to 0 outside a symmetric margin."""
    mid = (low + high) / 2
    half_width = (high - low) / 2
    margin = half_width * 2  # allow a bit of room outside the band before hitting 0
    return _clip01(100 - abs(value - mid) / (half_width + margin) * 100)


def technical_breakdown_long(t: TechnicalSnapshot) -> dict:
    """0-100 with drill-down: rewards price above both SMAs, a healthy
    uptrend (50>200), and RSI in the bullish-but-not-overbought band."""
    above = t.price > t.sma50 and t.price > t.sma200
    golden = t.sma50 > t.sma200
    components = [
        _component("Above both SMAs", 0.30, 100 if above else 0,
                    f"Price {t.price:,.2f} vs SMA50 {t.sma50:,.2f} / SMA200 {t.sma200:,.2f}"),
        _component("Trend alignment", 0.20, 100 if golden else 30,
                    "SMA50 above SMA200 (uptrend confirmed)" if golden else "SMA50 below SMA200 (trend not confirmed)"),
        _component("Distance vs 200SMA", 0.20, 50 + t.dist_from_200sma_pct * 2,
                    f"{t.dist_from_200sma_pct:+.1f}% from 200-day SMA"),
        _component("RSI momentum", 0.30, _band_score(t.rsi14, *LONG_RSI_BAND),
                    f"RSI(14) {t.rsi14:.1f}; ideal band {LONG_RSI_BAND[0]}-{LONG_RSI_BAND[1]}"),
    ]
    return {"score": round(_weighted(components), 1), "components": components}


def technical_breakdown_short(t: TechnicalSnapshot) -> dict:
    """0-100 with drill-down: rewards price below both SMAs, a confirmed
    downtrend (50<200), and RSI in the bearish-but-not-oversold band."""
    below = t.price < t.sma50 and t.price < t.sma200
    death = t.sma50 < t.sma200
    components = [
        _component("Below both SMAs", 0.30, 100 if below else 0,
                    f"Price {t.price:,.2f} vs SMA50 {t.sma50:,.2f} / SMA200 {t.sma200:,.2f}"),
        _component("Trend alignment", 0.20, 100 if death else 30,
                    "SMA50 below SMA200 (downtrend confirmed)" if death else "SMA50 above SMA200 (breakdown not confirmed)"),
        _component("Distance vs 200SMA", 0.20, 50 - t.dist_from_200sma_pct * 2,
                    f"{t.dist_from_200sma_pct:+.1f}% from 200-day SMA"),
        _component("RSI momentum", 0.30, _band_score(t.rsi14, *SHORT_RSI_BAND),
                    f"RSI(14) {t.rsi14:.1f}; ideal band {SHORT_RSI_BAND[0]}-{SHORT_RSI_BAND[1]}"),
    ]
    return {"score": round(_weighted(components), 1), "components": components}


# Thin scalar wrappers keep the original API for anything that only needs the number.
def fundamental_score_long(f: Fundamentals) -> float:
    return fundamental_breakdown_long(f)["score"]


def fundamental_score_short(f: Fundamentals) -> float:
    return fundamental_breakdown_short(f)["score"]


def technical_score_long(t: TechnicalSnapshot) -> float:
    return technical_breakdown_long(t)["score"]


def technical_score_short(t: TechnicalSnapshot) -> float:
    return technical_breakdown_short(t)["score"]


# --------------------------------------------------------------------------
# PIPELINE
# --------------------------------------------------------------------------

def _extract_hist(price_data: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
    """Pulls one ticker's OHLCV frame out of a (possibly multi-ticker) yf.download() result."""
    try:
        hist = price_data[ticker] if isinstance(price_data.columns, pd.MultiIndex) else price_data
    except KeyError:
        return None
    if hist is None or hist.empty or "Close" not in hist or hist["Close"].dropna().empty:
        return None
    return hist


def fetch_price_history(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Downloads OHLCV history in small batches instead of one call for the
    whole universe. yf.download() silently drops rate-limited tickers rather
    than raising, so after each batch we check which tickers actually came
    back with data and retry only the missing ones, backing off further each
    round -- this survives partial rate-limiting without re-hammering names
    that already succeeded.
    """
    pending = list(tickers)
    results: dict[str, pd.DataFrame] = {}

    for attempt in range(PRICE_DOWNLOAD_RETRIES + 1):
        if not pending:
            break
        still_missing: list[str] = []
        for i in range(0, len(pending), PRICE_CHUNK_SIZE):
            chunk = pending[i : i + PRICE_CHUNK_SIZE]
            try:
                price_data = yf.download(
                    tickers=chunk,
                    period=PRICE_HISTORY_PERIOD,
                    interval="1d",
                    group_by="ticker",
                    auto_adjust=True,
                    threads=True,
                    progress=False,
                )
            except Exception as exc:
                print(f"[WARN] price batch download raised ({exc}); will retry", file=sys.stderr)
                still_missing.extend(chunk)
                time.sleep(PRICE_CHUNK_PAUSE)
                continue

            for ticker in chunk:
                hist = _extract_hist(price_data, ticker)
                if hist is not None:
                    results[ticker] = hist
                else:
                    still_missing.append(ticker)
            time.sleep(PRICE_CHUNK_PAUSE)  # courtesy gap between batches

        pending = still_missing
        if pending and attempt < PRICE_DOWNLOAD_RETRIES:
            wait = PRICE_RATE_LIMIT_BACKOFF_BASE * (2 ** attempt)
            print(f"[INFO] {len(pending)} tickers still missing price data; "
                  f"retrying in {wait:.0f}s (round {attempt + 2}/{PRICE_DOWNLOAD_RETRIES + 1})...")
            time.sleep(wait)

    if pending:
        preview = ", ".join(pending[:10]) + ("..." if len(pending) > 10 else "")
        print(f"[WARN] Giving up on {len(pending)} tickers after repeated rate limiting: {preview}",
              file=sys.stderr)

    return results


def fetch_universe_data(tickers: list[str]) -> dict[str, tuple[Fundamentals, TechnicalSnapshot]]:
    """Fetches price history + fundamentals for `tickers` ONCE and returns a
    ticker -> (Fundamentals, TechnicalSnapshot) map that both the long and
    short scoring passes can reuse. Fetching once (rather than once per
    long/short pass) halves network calls and avoids hammering yfinance's
    local sqlite response cache with duplicate concurrent requests."""

    print(f"Downloading price history for {len(tickers)} tickers...")
    price_histories = fetch_price_history(tickers)

    technicals: dict[str, TechnicalSnapshot] = {}
    for ticker, hist in price_histories.items():
        snap = compute_technicals(hist)
        if snap is not None:
            technicals[ticker] = snap

    print(f"Fetched technicals for {len(technicals)}/{len(tickers)} tickers.")
    print(f"Fetching fundamentals for {len(technicals)} tickers (threaded)...")

    fundamentals: dict[str, Fundamentals] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(fetch_fundamentals, t): t for t in technicals}
        for future in as_completed(futures):
            ticker = futures[future]
            result = future.result()
            if result is not None:
                fundamentals[ticker] = result

    data: dict[str, tuple[Fundamentals, TechnicalSnapshot]] = {}
    for ticker, tech in technicals.items():
        fund = fundamentals.get(ticker)
        if fund is None:
            continue
        # Liquidity gate: skip small/illiquid names regardless of scores.
        if pd.isna(fund.market_cap) or fund.market_cap < MIN_MARKET_CAP:
            continue
        if pd.isna(tech.avg_volume_3m) or tech.avg_volume_3m < MIN_AVG_VOLUME:
            continue
        data[ticker] = (fund, tech)

    return data


def score_candidates(
    data: dict[str, tuple[Fundamentals, TechnicalSnapshot]],
    side: str,
) -> list[dict]:
    """Scores already-fetched (fundamentals, technicals) pairs for one side
    ("long"/"short") and returns candidate dicts carrying the full drill-down:
    component scores, filter checklist, and the raw metrics behind them."""
    cfg = SIDE_CONFIG[side]
    rows: list[dict] = []
    for ticker, (fund, tech) in data.items():
        fund_bd = cfg["fund_breakdown"](fund)
        tech_bd = cfg["tech_breakdown"](tech)
        checks = cfg["filter_checks"](fund, tech)
        composite = round(0.5 * fund_bd["score"] + 0.5 * tech_bd["score"], 1)

        rows.append({
            "ticker": ticker,
            "name": fund.name,
            "sector": fund.sector,
            "industry": fund.industry,
            "price": tech.price,
            "rsi14": tech.rsi14,
            "dist_from_200sma_pct": tech.dist_from_200sma_pct,
            "dist_from_50sma_pct": tech.dist_from_50sma_pct,
            "pe_ratio": fund.pe_ratio,
            "market_cap": fund.market_cap,
            "revenue_growth": fund.revenue_growth,
            "eps": fund.eps,
            "eps_growth": fund.eps_growth,
            "debt_to_equity": fund.debt_to_equity,
            "avg_volume_3m": tech.avg_volume_3m,
            "fundamental_score": fund_bd["score"],
            "technical_score": tech_bd["score"],
            "composite_score": composite,
            "passes_hard_filter": checks_verdict(checks),
            "fundamental_components": fund_bd["components"],
            "technical_components": tech_bd["components"],
            "filter_checks": checks,
        })
    return rows


def _check(label: str, passed: bool, detail: str, group: str = "required") -> dict:
    """One hard-filter criterion. group="required" must pass individually;
    group="any_one" criteria pass as a set if at least one of them is true."""
    return {"label": label, "passed": bool(passed), "detail": detail, "group": group}


def checks_verdict(checks: list[dict]) -> bool:
    required = [c for c in checks if c["group"] == "required"]
    any_one = [c for c in checks if c["group"] == "any_one"]
    return all(c["passed"] for c in required) and (any(c["passed"] for c in any_one) if any_one else True)


def long_filter_checks(f: Fundamentals, t: TechnicalSnapshot) -> list[dict]:
    """Strict long checklist per the spec: growth, positive EPS, healthy
    leverage, price above both SMAs, RSI in the bullish band. All must pass."""
    return [
        _check("Revenue growing (YoY)", pd.notna(f.revenue_growth) and f.revenue_growth > 0,
               f"YoY revenue {_fmt_pct(f.revenue_growth)}"),
        _check("EPS positive", pd.notna(f.eps) and f.eps > 0,
               f"trailing EPS {_fmt_num(f.eps, 2)}"),
        _check("Debt/Equity under 150", pd.isna(f.debt_to_equity) or f.debt_to_equity < 150,
               f"D/E {_fmt_num(f.debt_to_equity, 0)}"),
        _check("Price above 50 & 200-day SMA", t.price > t.sma50 and t.price > t.sma200,
               f"price {t.price:,.2f}, SMA50 {t.sma50:,.2f}, SMA200 {t.sma200:,.2f}"),
        _check(f"RSI in {LONG_RSI_BAND[0]}–{LONG_RSI_BAND[1]} band",
               LONG_RSI_BAND[0] <= t.rsi14 <= LONG_RSI_BAND[1],
               f"RSI(14) {t.rsi14:.1f}"),
    ]


def short_filter_checks(f: Fundamentals, t: TechnicalSnapshot) -> list[dict]:
    """Strict short checklist per the spec: at least one fundamental weakness
    (declining revenue, negative EPS, stretched valuation, or heavy debt),
    plus price below both SMAs and RSI in the bearish band."""
    return [
        _check("Revenue declining (YoY)", pd.notna(f.revenue_growth) and f.revenue_growth < 0,
               f"YoY revenue {_fmt_pct(f.revenue_growth)}", group="any_one"),
        _check("EPS negative", pd.notna(f.eps) and f.eps < 0,
               f"trailing EPS {_fmt_num(f.eps, 2)}", group="any_one"),
        _check("P/E above 40 (stretched)", pd.notna(f.pe_ratio) and f.pe_ratio > 40,
               f"P/E {_fmt_num(f.pe_ratio)}", group="any_one"),
        _check("Debt/Equity above 200", pd.notna(f.debt_to_equity) and f.debt_to_equity > 200,
               f"D/E {_fmt_num(f.debt_to_equity, 0)}", group="any_one"),
        _check("Price below 50 & 200-day SMA", t.price < t.sma50 and t.price < t.sma200,
               f"price {t.price:,.2f}, SMA50 {t.sma50:,.2f}, SMA200 {t.sma200:,.2f}"),
        _check(f"RSI in {SHORT_RSI_BAND[0]}–{SHORT_RSI_BAND[1]} band",
               SHORT_RSI_BAND[0] <= t.rsi14 <= SHORT_RSI_BAND[1],
               f"RSI(14) {t.rsi14:.1f}"),
    ]


def long_hard_filter(f: Fundamentals, t: TechnicalSnapshot) -> bool:
    return checks_verdict(long_filter_checks(f, t))


def short_hard_filter(f: Fundamentals, t: TechnicalSnapshot) -> bool:
    return checks_verdict(short_filter_checks(f, t))


SIDE_CONFIG = {
    "long": {
        "fund_breakdown": fundamental_breakdown_long,
        "tech_breakdown": technical_breakdown_long,
        "filter_checks": long_filter_checks,
    },
    "short": {
        "fund_breakdown": fundamental_breakdown_short,
        "tech_breakdown": technical_breakdown_short,
        "filter_checks": short_filter_checks,
    },
}


def rank_top_n(candidates: list[dict], n: int) -> list[dict]:
    """Ranks by composite score. Prefers rows that pass the strict hard
    filter; if fewer than `n` pass, backfills with the next-best scores so
    the output still has up to `n` rows. Each returned row gets a 1-based
    "rank" and keeps "passes_hard_filter" so the UI can flag backfills."""
    strict = sorted((c for c in candidates if c["passes_hard_filter"]),
                    key=lambda c: c["composite_score"], reverse=True)
    if len(strict) < n:
        loose = sorted((c for c in candidates if not c["passes_hard_filter"]),
                       key=lambda c: c["composite_score"], reverse=True)
        if candidates:
            print(f"[INFO] Only {len(strict)} names passed the strict filter; "
                  f"backfilling to {min(n, len(candidates))} with the next-highest composite scores.")
        strict = strict + loose

    ranked = strict[:n]
    for i, row in enumerate(ranked, start=1):
        row["rank"] = i
    return ranked


def _json_safe(obj):
    """Recursively replaces NaN/inf with None so the payload is valid JSON."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


RESULTS_CACHE = Path(__file__).parent / "last_scan.json"


def filter_by_sector(candidates: list[dict], sector: Optional[str]) -> list[dict]:
    """Keeps only candidates whose principal business activity (GICS sector)
    matches. None/"" /"all" means no filtering; matching is case-insensitive."""
    if not sector or sector.lower() == "all":
        return candidates
    return [c for c in candidates if (c.get("sector") or "").lower() == sector.lower()]


def run_scan(tickers: list[str], top_n: int = 10, universe_label: str = "custom",
             sector: Optional[str] = None, progress=None) -> dict:
    """Full pipeline: fetch -> score both sides -> rank -> JSON-safe payload.
    The payload keeps the FULL scored candidate lists ("candidates") alongside
    the ranked top-N, so a sector filter can re-rank later without hitting the
    Yahoo API again. Writes the payload to RESULTS_CACHE so the web UI can show
    the latest scan without refetching. `progress` is an optional callable(str)
    for status updates (used by the web UI; the CLI just prints)."""
    notify = progress or (lambda msg: None)

    notify(f"Downloading price history for {len(tickers)} tickers...")
    data = fetch_universe_data(tickers)
    notify(f"Scoring {len(data)} tickers that passed the liquidity gate...")

    all_long = score_candidates(data, "long")
    all_short = score_candidates(data, "short")
    sectors = sorted({c["sector"] for c in all_long if c["sector"]})

    long_ranked = rank_top_n(filter_by_sector(all_long, sector), top_n)
    short_ranked = rank_top_n(filter_by_sector(all_short, sector), top_n)

    payload = _json_safe({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "universe": universe_label,
        "universe_size": len(tickers),
        "scanned": len(data),
        "top_n": top_n,
        "sector_filter": sector or None,
        "sectors": sectors,
        "criteria": {
            "long": {"rsi_band": list(LONG_RSI_BAND)},
            "short": {"rsi_band": list(SHORT_RSI_BAND)},
            "min_market_cap": MIN_MARKET_CAP,
            "min_avg_volume": MIN_AVG_VOLUME,
        },
        "long": long_ranked,
        "short": short_ranked,
        "candidates": {"long": all_long, "short": all_short},
    })

    try:
        RESULTS_CACHE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as exc:
        print(f"[WARN] could not write results cache: {exc}", file=sys.stderr)

    return payload


# --------------------------------------------------------------------------
# OUTPUT
# --------------------------------------------------------------------------

DISPLAY_COLUMNS = {
    "ticker": "Ticker",
    "name": "Company Name",
    "price": "Price",
    "rsi14": "RSI(14)",
    "dist_from_200sma_pct": "Dist from 200SMA (%)",
    "pe_ratio": "P/E",
    "composite_score": "Composite Score",
}


def format_output(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    out = df[list(DISPLAY_COLUMNS.keys())].rename(columns=DISPLAY_COLUMNS)
    out["Price"] = out["Price"].map(lambda v: f"{v:,.2f}")
    out["RSI(14)"] = out["RSI(14)"].map(lambda v: f"{v:.1f}")
    out["Dist from 200SMA (%)"] = out["Dist from 200SMA (%)"].map(lambda v: f"{v:+.1f}%")
    out["P/E"] = out["P/E"].map(lambda v: f"{v:.1f}" if v is not None and pd.notna(v) and v > 0 else "N/A")
    out["Composite Score"] = out["Composite Score"].map(lambda v: f"{v:.1f}")
    out.index = range(1, len(out) + 1)
    return out


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="US stock long/short scanner")
    parser.add_argument("--universe", choices=["sp500", "nasdaq100"], default="sp500",
                        help="Predefined ticker universe to scan (default: sp500)")
    parser.add_argument("--tickers", type=str, default=None,
                        help="Comma-separated custom ticker list, overrides --universe")
    parser.add_argument("--top-n", type=int, default=10, help="Number of long/short candidates to output")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional cap on universe size, useful for quick test runs")
    parser.add_argument("--sector", type=str, default=None,
                        help='Only rank companies in this GICS sector, e.g. "Technology", '
                             '"Healthcare", "Financial Services" (case-insensitive)')
    return parser.parse_args()


def resolve_universe(universe: str, tickers_csv: Optional[str] = None,
                     limit: Optional[int] = None) -> tuple[list[str], str]:
    """Turns CLI/web parameters into a concrete ticker list + display label."""
    if tickers_csv:
        tickers = [t.strip().upper() for t in tickers_csv.split(",") if t.strip()]
        label = "custom"
    elif universe == "nasdaq100":
        tickers, label = get_nasdaq100_tickers(), "nasdaq100"
    else:
        tickers, label = get_sp500_tickers(), "sp500"
    if limit:
        tickers = tickers[:limit]
    return tickers, label


def main() -> None:
    args = parse_args()
    tickers, label = resolve_universe(args.universe, args.tickers, args.limit)
    results = run_scan(tickers, top_n=args.top_n, universe_label=label,
                       sector=args.sector, progress=print)

    if args.sector and not any((results[s] for s in ("long", "short"))):
        print(f"\nNo candidates in sector '{args.sector}'. "
              f"Sectors found in this scan: {', '.join(results['sectors']) or 'none'}")

    scope = f" — sector: {args.sector}" if args.sector else ""
    for side, title in (("long", "LONG CANDIDATES (bullish fundamentals + momentum)"),
                        ("short", "SHORT CANDIDATES (bearish fundamentals + breakdown)")):
        print("\n" + "=" * 80)
        print(f"TOP {args.top_n} {title}{scope}")
        print("=" * 80)
        rows = results[side]
        print(format_output(rows).to_string() if rows else "No candidates found.")


if __name__ == "__main__":
    main()
