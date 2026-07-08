"""
Web dashboard for the US stock long/short scanner.

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

from flask import Flask, jsonify, render_template, request

import stock_scanner as scanner

app = Flask(__name__)

# One scan at a time; results are also persisted by run_scan() to last_scan.json
# so a server restart still has the previous scan to show.
_state_lock = threading.Lock()
_state = {
    "status": "idle",        # idle | running | done | error
    "message": "",
    "started_at": None,
    "error": None,
}
_results: dict | None = None


def _load_cached_results() -> None:
    global _results
    try:
        _results = json.loads(scanner.RESULTS_CACHE.read_text(encoding="utf-8"))
        _state["status"] = "done"
        _state["message"] = "Loaded last saved scan."
    except (OSError, json.JSONDecodeError):
        _results = None


def _scan_worker(universe: str, tickers_csv: str | None, limit: int | None, top_n: int) -> None:
    global _results

    def progress(msg: str) -> None:
        with _state_lock:
            _state["message"] = str(msg)

    try:
        tickers, label = scanner.resolve_universe(universe, tickers_csv, limit)
        payload = scanner.run_scan(tickers, top_n=top_n, universe_label=label, progress=progress)
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


@app.post("/api/scan")
def start_scan():
    body = request.get_json(silent=True) or {}
    with _state_lock:
        if _state["status"] == "running":
            return jsonify({"error": "A scan is already running."}), 409
        _state.update(status="running", error=None, message="Starting scan...",
                      started_at=datetime.now(timezone.utc).isoformat())

    limit = body.get("limit")
    thread = threading.Thread(
        target=_scan_worker,
        kwargs={
            "universe": body.get("universe", "sp500"),
            "tickers_csv": body.get("tickers") or None,
            "limit": int(limit) if limit else None,
            "top_n": int(body.get("top_n", 10)),
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
    sector = request.args.get("sector") or None
    top_n = request.args.get("top_n", type=int)
    with _state_lock:
        if _results is None:
            return jsonify({"error": "No scan results yet."}), 404
        payload = dict(_results)  # shallow copy; we only replace ranked lists

        # Re-rank from the full candidate lists so a sector change never needs
        # a new Yahoo fetch. Older cache files without "candidates" fall back
        # to the stored ranked lists (sector filter unavailable there).
        candidates = payload.pop("candidates", None)  # full lists stay server-side
        if candidates and (sector or top_n):
            n = top_n or payload.get("top_n", 10)
            payload["long"] = scanner.rank_top_n(scanner.filter_by_sector(candidates["long"], sector), n)
            payload["short"] = scanner.rank_top_n(scanner.filter_by_sector(candidates["short"], sector), n)
            payload["sector_filter"] = sector
            payload["top_n"] = n
    return jsonify(payload)


_load_cached_results()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
