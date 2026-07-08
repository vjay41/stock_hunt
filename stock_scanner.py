"""
Stock Long/Short Scanner
========================
Scans a liquid universe of stocks from a selected market (US, India, Singapore)
and ranks them into the Top 10 "Long" and "Short" candidates.

Data source: Yahoo Finance via `yfinance` (free, no API key required).

Usage:
    python stock_scanner.py --market us --universe sp500
    python stock_scanner.py --market india --universe nifty50 --top-n 15
    python stock_scanner.py --market us --tickers AAPL,MSFT,TSLA,NVDA,XOM
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
from typing import List, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

WIKI_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; stock-scanner/1.0)"}
warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------

MIN_MARKET_CAP = 2_000_000_000
MIN_AVG_VOLUME = 1_000_000
PRICE_HISTORY_PERIOD = "1y"
SMA_SHORT, SMA_LONG = 50, 200
RSI_PERIOD = 14

LONG_RSI_BAND = (55, 70)
SHORT_RSI_BAND = (30, 45)

MAX_WORKERS = 5
MIN_REQUEST_INTERVAL = 0.35
FUNDAMENTALS_RETRIES = 5  # Increased retries
RATE_LIMIT_BACKOFF_BASE = 10.0 # Increased backoff

PRICE_CHUNK_SIZE = 40
PRICE_CHUNK_PAUSE = 2.0
PRICE_DOWNLOAD_RETRIES = 3
PRICE_RATE_LIMIT_BACKOFF_BASE = 20.0

RESULTS_CACHE = Path(__file__).parent / "last_scan.json"
SECTORS_CACHE = Path(__file__).parent / "sectors_cache.json"


class RateLimiter:
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

def _scrape_wiki_tickers(url: str, symbol_cols: List[str], suffix: str = "") -> list[str]:
    resp = requests.get(url, headers=WIKI_HEADERS, timeout=10)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    for df in tables:
        for col in symbol_cols:
            if col in df.columns:
                symbols = df[col].dropna()
                tickers = symbols.str.replace(".", "-", regex=False).tolist()
                if suffix:
                    tickers = [f"{t}{suffix}" for t in tickers]
                return tickers
    raise ValueError(f"Could not find any of {symbol_cols} in any table at {url}")

def get_sp500_tickers() -> list[str]:
    try:
        return _scrape_wiki_tickers("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", ["Symbol"])
    except Exception as exc:
        print(f"[WARN] S&P 500 scrape failed ({exc}). Using small default list.", file=sys.stderr)
        return ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "JPM", "XOM", "UNH"]

def get_nasdaq100_tickers() -> list[str]:
    try:
        return _scrape_wiki_tickers("https://en.wikipedia.org/wiki/Nasdaq-100", ["Ticker", "Symbol"])
    except Exception as exc:
        print(f"[WARN] Nasdaq-100 scrape failed ({exc}). Falling back to S&P 500.", file=sys.stderr)
        return get_sp500_tickers()

def get_nifty50_tickers() -> list[str]:
    try:
        return _scrape_wiki_tickers("https://en.wikipedia.org/wiki/NIFTY_50", ["Symbol"], suffix=".NS")
    except Exception as exc:
        print(f"[WARN] NIFTY 50 scrape failed ({exc}). Using small default list.", file=sys.stderr)
        return ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]

def get_nifty500_tickers() -> list[str]:
    try:
        return _scrape_wiki_tickers("https://en.wikipedia.org/wiki/NIFTY_500", ["Symbol"], suffix=".NS")
    except Exception as exc:
        print(f"[WARN] NIFTY 500 scrape failed ({exc}). Falling back to NIFTY 50.", file=sys.stderr)
        return get_nifty50_tickers()

def get_sti_tickers() -> list[str]:
    """Returns a hardcoded list of major Straits Times Index (STI) constituents."""
    print("[INFO] Using a hardcoded default list for the Straits Times Index (STI).", file=sys.stderr)
    return [
        "D05.SI", "O39.SI", "U11.SI", "Z74.SI", "C38U.SI",
        "G13.SI", "C09.SI", "S68.SI", "A17U.SI", "9CI.SI"
    ]

def get_ftse_all_share_tickers() -> list[str]:
    try:
        return _scrape_wiki_tickers("https://en.wikipedia.org/wiki/FTSE_ST_All-Share_Index", ["Ticker"], suffix=".SI")
    except Exception as exc:
        print(f"[WARN] FTSE ST All-Share scrape failed ({exc}). Falling back to STI.", file=sys.stderr)
        return get_sti_tickers()

def get_us_composite_tickers() -> list[str]:
    """Combines S&P 500 and Nasdaq 100 for a broader US market view."""
    sp500 = get_sp500_tickers()
    nasdaq100 = get_nasdaq100_tickers()
    return sorted(list(set(sp500 + nasdaq100)))

UNIVERSE_BUILDERS = {
    "us": {
        "us_composite": get_us_composite_tickers,
        "sp500": get_sp500_tickers, 
        "nasdaq100": get_nasdaq100_tickers
    },
    "india": {
        "nifty500": get_nifty500_tickers,
        "nifty50": get_nifty50_tickers
    },
    "singapore": {
        "ftse_all_share": get_ftse_all_share_tickers,
        "sti": get_sti_tickers
    },
}

# --------------------------------------------------------------------------
# TECHNICAL INDICATORS
# --------------------------------------------------------------------------

def compute_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

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
    if hist is None or hist.empty or "Close" not in hist:
        return None
    close = hist["Close"].dropna()
    volume = hist["Volume"].dropna()
    if len(close) < SMA_LONG:
        return None

    sma50 = close.rolling(SMA_SHORT).mean().iloc[-1]
    sma200 = close.rolling(SMA_LONG).mean().iloc[-1]
    rsi14 = compute_rsi(close).iloc[-1]
    price = close.iloc[-1]
    avg_vol_3m = volume.tail(63).mean()

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
    sector: str = ""
    industry: str = ""
    market_cap: float = np.nan
    pe_ratio: float = np.nan
    eps: float = np.nan
    eps_growth: float = np.nan
    revenue_growth: float = np.nan
    debt_to_equity: float = np.nan

def fetch_fundamentals(ticker: str, retries: int = FUNDAMENTALS_RETRIES) -> Optional[Fundamentals]:
    for attempt in range(retries + 1):
        _rate_limiter.wait()
        try:
            info = yf.Ticker(ticker).get_info()
            
            # If essential data is missing, raise an error to trigger a retry.
            if not info or info.get("marketCap") is None or info.get("sector") is None:
                raise ValueError(f"Incomplete data for {ticker}, retrying...")

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
                    time.sleep(1.0 * (attempt + 1)) # Increased backoff
                continue
            if "Not Found" not in str(exc):
                print(f"[WARN] fundamentals failed for {ticker}: {exc}", file=sys.stderr)
            return None

def get_all_market_sectors() -> list[str]:
    if SECTORS_CACHE.exists():
        print("[INFO] Loading sectors from local cache.")
        try:
            return json.loads(SECTORS_CACHE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"[WARN] Could not read sectors cache: {e}. Re-fetching.", file=sys.stderr)

    print("[INFO] Compiling a list of all sectors from all markets...")
    all_tickers = []
    for market, universes in UNIVERSE_BUILDERS.items():
        primary_universe = list(universes.keys())[0]
        try:
            all_tickers.extend(universes[primary_universe]()[:50])
        except Exception as exc:
            print(f"[WARN] Could not fetch tickers for {market}/{primary_universe}: {exc}", file=sys.stderr)

    sectors = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(fetch_fundamentals, t): t for t in all_tickers}
        for future in as_completed(futures):
            result = future.result()
            if result and result.sector:
                sectors[result.sector] = 1
    
    sorted_sectors = sorted(list(sectors.keys()))
    print(f"[INFO] Found {len(sorted_sectors)} unique sectors. Caching locally.")
    try:
        SECTORS_CACHE.write_text(json.dumps(sorted_sectors, indent=2), encoding="utf-8")
    except OSError as e:
        print(f"[WARN] Could not write sectors cache: {e}", file=sys.stderr)
        
    return sorted_sectors

# --------------------------------------------------------------------------
# SCORING & PIPELINE
# --------------------------------------------------------------------------

def _check(label: str, passed: bool, detail: str, group: str = "required") -> dict:
    return {"label": label, "passed": bool(passed), "detail": detail, "group": group}

def checks_verdict(checks: list[dict]) -> bool:
    required = [c["passed"] for c in checks if c["group"] == "required"]
    any_one = [c["passed"] for c in checks if c["group"] == "any_one"]
    return all(required) and (any(any_one) if any_one else True)

def long_filter_checks(f: Fundamentals, t: TechnicalSnapshot) -> list[dict]:
    return [
        _check("Revenue growth > 5% (YoY)", pd.notna(f.revenue_growth) and f.revenue_growth > 0.05, f"YoY revenue {_fmt_pct(f.revenue_growth)}"),
        _check("EPS growth > 5% (YoY)", pd.notna(f.eps_growth) and f.eps_growth > 0.05, f"YoY EPS growth {_fmt_pct(f.eps_growth)}"),
        _check("EPS positive", pd.notna(f.eps) and f.eps > 0, f"trailing EPS {_fmt_num(f.eps, 2)}"),
        _check("Debt/Equity under 100", pd.isna(f.debt_to_equity) or f.debt_to_equity < 100, f"D/E {_fmt_num(f.debt_to_equity, 0)}"),
        _check("Price above 50 & 200-day SMA", t.price > t.sma50 and t.price > t.sma200, f"price {t.price:,.2f}, SMA50 {t.sma50:,.2f}, SMA200 {t.sma200:,.2f}"),
        _check(f"RSI in {LONG_RSI_BAND[0]}–{LONG_RSI_BAND[1]} band", LONG_RSI_BAND[0] <= t.rsi14 <= LONG_RSI_BAND[1], f"RSI(14) {t.rsi14:.1f}"),
    ]

def short_filter_checks(f: Fundamentals, t: TechnicalSnapshot) -> list[dict]:
    return [
        _check("Revenue declining > 5% (YoY)", pd.notna(f.revenue_growth) and f.revenue_growth < -0.05, f"YoY revenue {_fmt_pct(f.revenue_growth)}", group="any_one"),
        _check("EPS negative or declining > 5%", (pd.notna(f.eps) and f.eps < 0) or (pd.notna(f.eps_growth) and f.eps_growth < -0.05), f"trailing EPS {_fmt_num(f.eps, 2)}, YoY growth {_fmt_pct(f.eps_growth)}", group="any_one"),
        _check("P/E above 50 (very stretched)", pd.notna(f.pe_ratio) and f.pe_ratio > 50, f"P/E {_fmt_num(f.pe_ratio)}", group="any_one"),
        _check("Debt/Equity above 250 (high risk)", pd.notna(f.debt_to_equity) and f.debt_to_equity > 250, f"D/E {_fmt_num(f.debt_to_equity, 0)}", group="any_one"),
        _check("Price below 50 & 200-day SMA", t.price < t.sma50 and t.price < t.sma200, f"price {t.price:,.2f}, SMA50 {t.sma50:,.2f}, SMA200 {t.sma200:,.2f}"),
        _check(f"RSI in {SHORT_RSI_BAND[0]}–{SHORT_RSI_BAND[1]} band", SHORT_RSI_BAND[0] <= t.rsi14 <= SHORT_RSI_BAND[1], f"RSI(14) {t.rsi14:.1f}"),
    ]

def _fmt_pct(v: float, signed: bool = True) -> str:
    if pd.isna(v): return "no data"
    return f"{v * 100:+.1f}%" if signed else f"{v * 100:.1f}%"

def _fmt_num(v: float, decimals: int = 1) -> str:
    return "no data" if pd.isna(v) else f"{v:.{decimals}f}"

def _component(label: str, weight: float, score: float, detail: str) -> dict:
    return {"label": label, "weight": weight, "score": round(np.clip(score, 0, 100), 1), "detail": detail}

def _weighted(components: list[dict]) -> float:
    return np.clip(sum(c["weight"] * c["score"] for c in components), 0, 100)

def fundamental_breakdown_long(f: Fundamentals) -> dict:
    growth = 0 if pd.isna(f.revenue_growth) else np.clip(f.revenue_growth * 100, -30, 30)
    growth_score = (growth + 30) / 60 * 100
    eps_growth = 0 if pd.isna(f.eps_growth) else np.clip(f.eps_growth * 100, -50, 50)
    eps_score = (eps_growth + 50) / 100 * 100
    eps_detail = f"EPS growth {_fmt_pct(f.eps_growth)}, trailing EPS {_fmt_num(f.eps, 2)}"
    if pd.notna(f.eps) and f.eps <= 0:
        eps_score *= 0.3
        eps_detail += " (unprofitable: score cut 70%)"
    de = 100 if pd.isna(f.debt_to_equity) else f.debt_to_equity
    leverage_score = 100 - np.clip(de, 0, 300) / 300 * 100
    if pd.isna(f.pe_ratio) or f.pe_ratio <= 0:
        valuation_score = 40
        valuation_detail = "P/E unavailable or negative"
    else:
        valuation_score = np.clip(100 - abs(f.pe_ratio - 20) * 1.5, 0, 100)
        valuation_detail = f"P/E {f.pe_ratio:.1f} vs ~20x sweet spot"
    components = [
        _component("Revenue growth", 0.35, growth_score, f"YoY revenue {_fmt_pct(f.revenue_growth)}"),
        _component("EPS quality", 0.25, eps_score, eps_detail),
        _component("Leverage", 0.20, leverage_score, f"Debt/Equity {_fmt_num(f.debt_to_equity, 0)} (lower is better)"),
        _component("Valuation", 0.20, valuation_score, valuation_detail),
    ]
    return {"score": round(_weighted(components), 1), "components": components}

def fundamental_breakdown_short(f: Fundamentals) -> dict:
    growth = 0 if pd.isna(f.revenue_growth) else np.clip(f.revenue_growth * 100, -30, 30)
    growth_score = 100 - (growth + 30) / 60 * 100
    eps_growth = 0 if pd.isna(f.eps_growth) else np.clip(f.eps_growth * 100, -50, 50)
    eps_score = 100 - (eps_growth + 50) / 100 * 100
    eps_detail = f"EPS growth {_fmt_pct(f.eps_growth)}, trailing EPS {_fmt_num(f.eps, 2)}"
    if pd.notna(f.eps) and f.eps <= 0:
        eps_score = min(100, eps_score + 30)
        eps_detail += " (unprofitable: +30 to short case)"
    de = 0 if pd.isna(f.debt_to_equity) else f.debt_to_equity
    leverage_score = np.clip(de, 0, 300) / 300 * 100
    if pd.isna(f.pe_ratio) or f.pe_ratio <= 0:
        valuation_score = 60
        valuation_detail = "P/E unavailable or negative (red flag)"
    else:
        valuation_score = np.clip((f.pe_ratio - 20) * 1.5, 0, 100)
        valuation_detail = f"P/E {f.pe_ratio:.1f}; multiples above ~20x add to the short case"
    components = [
        _component("Revenue weakness", 0.35, growth_score, f"YoY revenue {_fmt_pct(f.revenue_growth)}"),
        _component("EPS deterioration", 0.25, eps_score, eps_detail),
        _component("Leverage risk", 0.20, leverage_score, f"Debt/Equity {_fmt_num(f.debt_to_equity, 0)} (higher is worse)"),
        _component("Valuation stretch", 0.20, valuation_score, valuation_detail),
    ]
    return {"score": round(_weighted(components), 1), "components": components}

def _band_score(value: float, low: float, high: float) -> float:
    mid = (low + high) / 2
    half_width = (high - low) / 2
    margin = half_width * 2
    return np.clip(100 - abs(value - mid) / (half_width + margin) * 100, 0, 100)

def technical_breakdown_long(t: TechnicalSnapshot) -> dict:
    above = t.price > t.sma50 and t.price > t.sma200
    golden = t.sma50 > t.sma200
    components = [
        _component("Above both SMAs", 0.30, 100 if above else 0, f"Price {t.price:,.2f} vs SMA50 {t.sma50:,.2f} / SMA200 {t.sma200:,.2f}"),
        _component("Trend alignment", 0.20, 100 if golden else 30, "SMA50 > SMA200 (uptrend)" if golden else "SMA50 < SMA200 (downtrend)"),
        _component("Distance vs 200SMA", 0.20, 50 + t.dist_from_200sma_pct * 2, f"{t.dist_from_200sma_pct:+.1f}% from 200-day SMA"),
        _component("RSI momentum", 0.30, _band_score(t.rsi14, *LONG_RSI_BAND), f"RSI(14) {t.rsi14:.1f}; ideal band {LONG_RSI_BAND[0]}-{LONG_RSI_BAND[1]}"),
    ]
    return {"score": round(_weighted(components), 1), "components": components}

def technical_breakdown_short(t: TechnicalSnapshot) -> dict:
    below = t.price < t.sma50 and t.price < t.sma200
    death = t.sma50 < t.sma200
    components = [
        _component("Below both SMAs", 0.30, 100 if below else 0, f"Price {t.price:,.2f} vs SMA50 {t.sma50:,.2f} / SMA200 {t.sma200:,.2f}"),
        _component("Trend alignment", 0.20, 100 if death else 30, "SMA50 < SMA200 (downtrend)" if death else "SMA50 > SMA200 (uptrend)"),
        _component("Distance vs 200SMA", 0.20, 50 - t.dist_from_200sma_pct * 2, f"{t.dist_from_200sma_pct:+.1f}% from 200-day SMA"),
        _component("RSI momentum", 0.30, _band_score(t.rsi14, *SHORT_RSI_BAND), f"RSI(14) {t.rsi14:.1f}; ideal band {SHORT_RSI_BAND[0]}-{SHORT_RSI_BAND[1]}"),
    ]
    return {"score": round(_weighted(components), 1), "components": components}

SIDE_CONFIG = {"long": {"fund_breakdown": fundamental_breakdown_long, "tech_breakdown": technical_breakdown_long, "filter_checks": long_filter_checks},
               "short": {"fund_breakdown": fundamental_breakdown_short, "tech_breakdown": technical_breakdown_short, "filter_checks": short_filter_checks}}

def _extract_hist(price_data: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
    try:
        hist = price_data[ticker] if isinstance(price_data.columns, pd.MultiIndex) else price_data
    except KeyError:
        return None
    return hist if hist is not None and not hist.empty and "Close" in hist and not hist["Close"].dropna().empty else None

def fetch_price_history(tickers: list[str]) -> dict[str, pd.DataFrame]:
    pending = list(tickers)
    results: dict[str, pd.DataFrame] = {}
    for attempt in range(PRICE_DOWNLOAD_RETRIES + 1):
        if not pending: break
        still_missing: list[str] = []
        for i in range(0, len(pending), PRICE_CHUNK_SIZE):
            chunk = pending[i : i + PRICE_CHUNK_SIZE]
            try:
                price_data = yf.download(tickers=chunk, period=PRICE_HISTORY_PERIOD, interval="1d", group_by="ticker", auto_adjust=True, threads=True, progress=False)
            except Exception as exc:
                print(f"[WARN] price batch download raised ({exc}); will retry", file=sys.stderr)
                still_missing.extend(chunk)
                time.sleep(PRICE_CHUNK_PAUSE)
                continue
            for ticker in chunk:
                hist = _extract_hist(price_data, ticker)
                if hist is not None: results[ticker] = hist
                else: still_missing.append(ticker)
            time.sleep(PRICE_CHUNK_PAUSE)
        pending = still_missing
        if pending and attempt < PRICE_DOWNLOAD_RETRIES:
            wait = PRICE_RATE_LIMIT_BACKOFF_BASE * (2 ** attempt)
            print(f"[INFO] {len(pending)} tickers still missing price data; retrying in {wait:.0f}s...", file=sys.stderr)
            time.sleep(wait)
    if pending:
        preview = ", ".join(pending[:10]) + ("..." if len(pending) > 10 else "")
        print(f"[WARN] Giving up on {len(pending)} tickers: {preview}", file=sys.stderr)
    return results

def fetch_universe_data(tickers: list[str], min_price: Optional[float], max_price: Optional[float]) -> dict[str, tuple[Fundamentals, TechnicalSnapshot]]:
    print(f"Downloading price history for {len(tickers)} tickers...")
    price_histories = fetch_price_history(tickers)
    technicals: dict[str, TechnicalSnapshot] = {t: s for t, h in price_histories.items() if (s := compute_technicals(h))}
    
    # Price filter
    if min_price is not None or max_price is not None:
        print(f"Applying price filter (min: {min_price}, max: {max_price})...")
        technicals = {t: s for t, s in technicals.items() 
                      if (min_price is None or s.price >= min_price) and \
                         (max_price is None or s.price <= max_price)}

    print(f"Fetched technicals for {len(technicals)}/{len(tickers)} tickers.")
    print(f"Fetching fundamentals for {len(technicals)} tickers (threaded)...")
    fundamentals: dict[str, Fundamentals] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(fetch_fundamentals, t): t for t in technicals}
        for future in as_completed(futures):
            if result := future.result():
                fundamentals[futures[future]] = result
    data: dict[str, tuple[Fundamentals, TechnicalSnapshot]] = {}
    for ticker, tech in technicals.items():
        if (fund := fundamentals.get(ticker)) and not (pd.isna(fund.market_cap) or fund.market_cap < MIN_MARKET_CAP or pd.isna(tech.avg_volume_3m) or tech.avg_volume_3m < MIN_AVG_VOLUME):
            data[ticker] = (fund, tech)
    return data

def score_candidates(data: dict[str, tuple[Fundamentals, TechnicalSnapshot]], side: str) -> list[dict]:
    cfg = SIDE_CONFIG[side]
    rows: list[dict] = []
    for ticker, (fund, tech) in data.items():
        fund_bd = cfg["fund_breakdown"](fund)
        tech_bd = cfg["tech_breakdown"](tech)
        checks = cfg["filter_checks"](fund, tech)
        rows.append({
            "ticker": ticker, "name": fund.name, "sector": fund.sector, "industry": fund.industry,
            "price": tech.price, "rsi14": tech.rsi14, "dist_from_200sma_pct": tech.dist_from_200sma_pct,
            "dist_from_50sma_pct": tech.dist_from_50sma_pct, "pe_ratio": fund.pe_ratio, "market_cap": fund.market_cap,
            "revenue_growth": fund.revenue_growth, "eps": fund.eps, "eps_growth": fund.eps_growth,
            "debt_to_equity": fund.debt_to_equity, "avg_volume_3m": tech.avg_volume_3m,
            "fundamental_score": fund_bd["score"], "technical_score": tech_bd["score"],
            "composite_score": round(0.5 * fund_bd["score"] + 0.5 * tech_bd["score"], 1),
            "passes_hard_filter": checks_verdict(checks),
            "fundamental_components": fund_bd["components"], "technical_components": tech_bd["components"],
            "filter_checks": checks,
        })
    return rows

def rank_top_n(candidates: list[dict], n: int) -> list[dict]:
    strict = sorted((c for c in candidates if c["passes_hard_filter"]), key=lambda c: c["composite_score"], reverse=True)
    ranked = strict[:n]
    for i, row in enumerate(ranked, start=1):
        row["rank"] = i
    return ranked

def _json_safe(obj):
    if isinstance(obj, dict): return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list): return [_json_safe(v) for v in obj]
    if isinstance(obj, (float, np.floating)): return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.bool_): return bool(obj)
    return obj

def run_scan(tickers: list[str], all_sectors: list[str], top_n: int = 10, universe_label: str = "custom",
             sector: Optional[str] = None, progress=None, market: str = "us", 
             min_price: Optional[float] = None, max_price: Optional[float] = None) -> dict:
    notify = progress or (lambda msg: None)
    
    if sector:
        notify(f"Pre-filtering for sector: {sector}...")
        filtered_tickers = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(fetch_fundamentals, t): t for t in tickers}
            for future in as_completed(futures):
                if (result := future.result()) and result.sector == sector:
                    filtered_tickers.append(futures[future])
        tickers = filtered_tickers
        if not tickers: print(f"[INFO] No tickers found in sector '{sector}' for this universe.")

    full_universe_size = len(tickers)
    data = fetch_universe_data(tickers, min_price, max_price)
    notify(f"Scoring {len(data)} tickers that passed the liquidity gate...")
    
    all_long = score_candidates(data, "long")
    all_short = score_candidates(data, "short")
    long_ranked = rank_top_n(all_long, top_n)
    short_ranked = rank_top_n(all_short, top_n)

    payload = _json_safe({
        "generated_at": datetime.now(timezone.utc).isoformat(), "market": market,
        "universe": universe_label, "universe_size": full_universe_size, "scanned": len(data),
        "top_n": top_n, "sector_filter": sector or None, "sectors": all_sectors,
        "min_price": min_price, "max_price": max_price,
        "criteria": {"long": {"rsi_band": list(LONG_RSI_BAND)}, "short": {"rsi_band": list(SHORT_RSI_BAND)},
                     "min_market_cap": MIN_MARKET_CAP, "min_avg_volume": MIN_AVG_VOLUME},
        "long": long_ranked, "short": short_ranked,
    })
    try:
        RESULTS_CACHE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as exc:
        print(f"[WARN] could not write results cache: {exc}", file=sys.stderr)
    return payload

def resolve_universe(market: str, universe: str | None, tickers_csv: str | None, limit: int | None) -> tuple[list[str], str]:
    if tickers_csv:
        return [t.strip().upper() for t in tickers_csv.split(",") if t.strip()], "custom"
    
    market_universes = UNIVERSE_BUILDERS.get(market)
    if not market_universes: raise ValueError(f"Invalid market: {market}")
    
    universe_key = universe or list(market_universes.keys())[0]
    if universe_key not in market_universes:
        raise ValueError(f"Invalid universe '{universe_key}' for market '{market}'")
        
    tickers = market_universes[universe_key]()
    if limit: tickers = tickers[:limit]
    return tickers, universe_key

def main() -> None:
    args = parse_args()
    all_sectors = get_all_market_sectors()
    tickers, label = resolve_universe(args.market, args.universe, args.tickers, args.limit)
    results = run_scan(tickers, all_sectors, top_n=args.top_n, universe_label=label,
                       sector=args.sector, progress=print, market=args.market,
                       min_price=args.min_price, max_price=args.max_price)
    
    scope = f"market: {args.market}, universe: {label}"
    if args.sector: scope += f", sector: {args.sector}"
    
    for side, title in (("long", "LONG CANDIDATES"), ("short", "SHORT CANDIDATES")):
        print("\n" + "=" * 80)
        print(f"TOP {args.top_n} {title} — {scope}")
        print("=" * 80)
        rows = results[side]
        if rows:
            print(f"{len(rows)} candidates found.")
        else:
            print("No candidates found.")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stock long/short scanner for multiple markets.")
    parser.add_argument("--market", choices=UNIVERSE_BUILDERS.keys(), default="us", help="Market to scan")
    parser.add_argument("--universe", default=None, help="Predefined ticker universe (e.g., sp500, nifty50)")
    parser.add_argument("--tickers", type=str, default=None, help="Comma-separated custom ticker list")
    parser.add_argument("--top-n", type=int, default=10, help="Number of candidates to output")
    parser.add_argument("--limit", type=int, default=None, help="Cap on universe size for quick tests")
    parser.add_argument("--sector", type=str, default=None, help="Filter by GICS sector")
    parser.add_argument("--min-price", type=float, default=None, help="Minimum stock price")
    parser.add_argument("--max-price", type=float, default=None, help="Maximum stock price")
    return parser.parse_args()

if __name__ == "__main__":
    main()