"""
Web dashboard for the stock long/short scanner.

Serves a single-page GUI on top of stock_scanner.run_scan(): kick off a scan,
watch its progress, and drill into each Top-10 candidate to see exactly which
criteria and score components put it on the list.

Run:
    python app.py            # http://127.0.0.1:5000
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from functools import lru_cache

from flask import Flask, jsonify, render_template, request

import stock_scanner as scanner

app = Flask(__name__)

_state_lock = threading.Lock()
_state = {
    "status": "idle",
    "message": "",
    "started_at": None,
    "error": None,
}
_results: dict | None = None
_all_sectors_cache: list[str] | None = None

def _load_cached_results() -> None:
    global _results
    try:
        _results = json.loads(scanner.RESULTS_CACHE.read_text(encoding="utf-8"))
        _state["status"] = "done"
        _state["message"] = "Loaded last saved scan."
    except (OSError, json.JSONDecodeError):
        _results = None

@lru_cache(maxsize=1)
def get_all_sectors() -> list[str]:
    """Caches the expensive sector fetch operation."""
    global _all_sectors_cache
    if _all_sectors_cache is None:
        _all_sectors_cache = scanner.get_all_market_sectors()
    return _all_sectors_cache

def _scan_worker(market: str, universe: str, tickers_csv: str | None, limit: int | None, 
                 top_n: int, sector: str | None, min_price: float | None, max_price: float | None) -> None:
    global _results

    def progress(msg: str) -> None:
        with _state_lock:
            _state["message"] = str(msg)

    try:
        all_sectors = get_all_sectors()
        tickers, label = scanner.resolve_universe(market, universe, tickers_csv, limit)
        payload = scanner.run_scan(tickers, all_sectors, top_n=top_n, universe_label=label, 
                                   sector=sector, progress=progress, market=market,
                                   min_price=min_price, max_price=max_price)
        with _state_lock:
            _results = payload
            _state["status"] = "done"
            _state["message"] = f"Scan complete: {payload['scanned']} tickers scored."
    except Exception as exc:
        with _state_lock:
            _state["status"] = "error"
            _state["error"] = str(exc)
            _state["message"] = f"Scan failed: {exc}"

@app.get("/")
def index():
    return render_template("index.html")

@app.get("/api/sectors")
def all_sectors():
    return jsonify(get_all_sectors())

@app.post("/api/scan")
def start_scan():
    body = request.get_json(silent=True) or {}
    with _state_lock:
        if _state["status"] == "running":
            return jsonify({"error": "A scan is already in progress."}), 409
        _state.update(status="running", error=None, message="Starting scan...",
                      started_at=datetime.now(timezone.utc).isoformat())

    limit = body.get("limit")
    min_price = body.get("min_price")
    max_price = body.get("max_price")
    
    thread = threading.Thread(
        target=_scan_worker,
        kwargs={
            "market": body.get("market", "us"),
            "universe": body.get("universe"),
            "tickers_csv": body.get("tickers") or None,
            "limit": int(limit) if limit else None,
            "top_n": int(body.get("top_n", 10)),
            "sector": body.get("sector") or None,
            "min_price": float(min_price) if min_price else None,
            "max_price": float(max_price) if max_price else None,
        },
        daemon=True,
    )
    thread.start()
    return jsonify({"ok": True})

@app.get("/api/status")
def status():
    with _state_lock:
        return jsonify(dict(_state, has_results=_results is not None))

@app.get("/api/results")
def results():
    with _state_lock:
        if _results is None:
            return jsonify({"error": "No scan results yet."}), 404
        return jsonify(_results)

_load_cached_results()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)