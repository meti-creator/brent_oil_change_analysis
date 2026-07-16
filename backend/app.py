"""
Brent Oil Price Analysis Dashboard - Flask Backend
====================================================
Serves the precomputed analysis results from Tasks 1-2:
- Real Brent price history (1987-2022)
- Real curated events (data/events.csv -> events.py)
- Real PELT change-point scan (ruptures)
- Real Bayesian single-change-point case studies (PyMC), with full
  posterior summaries and credible intervals

IMPORTANT: This app does NOT run any modeling live. All heavy computation
(PyMC sampling, PELT scanning) happens once, offline, via
`generate_results.py`, which writes data/results.json. This file only
reads and serves that cache. If you change the underlying analysis,
re-run `generate_results.py` before restarting this server.

There are no ARIMA / LSTM / Prophet / GARCH endpoints here, because none
of those models were built in this project -- only PELT and a Bayesian
change-point model. Do not add fabricated forecast/metrics endpoints;
extend generate_results.py with a real model first if that's needed.
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import json
import os
from datetime import datetime
from collections import defaultdict

app = Flask(__name__)
CORS(app)

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "data", "results.json")

with open(RESULTS_PATH) as f:
    RESULTS = json.load(f)

PRICE_SERIES = RESULTS["price_series"]
EVENTS = RESULTS["events"]
PELT_CHANGE_POINTS = RESULTS["pelt_change_points"]
BAYESIAN_CASE_STUDIES = RESULTS["bayesian_case_studies"]
METADATA = RESULTS["metadata"]


def filter_by_date(records, date_key, start, end):
    if not start and not end:
        return records
    out = records
    if start:
        out = [r for r in out if r[date_key] >= start]
    if end:
        out = [r for r in out if r[date_key] <= end]
    return out


# ============================================================================
# HISTORICAL PRICE DATA
# ============================================================================

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "data_range": METADATA["data_range"],
        "n_observations": METADATA["n_observations"],
        "results_generated_at": METADATA["generated_at"],
    })


@app.route("/api/historical", methods=["GET"])
def get_historical():
    """
    Real Brent price series, 1987-2022.

    Query Parameters:
        start_date (str, YYYY-MM-DD, optional)
        end_date (str, YYYY-MM-DD, optional)
    """
    start = request.args.get("start_date")
    end = request.args.get("end_date")
    data = filter_by_date(PRICE_SERIES, "date", start, end)
    return jsonify({
        "data": data,
        "count": len(data),
        "full_range": METADATA["data_range"],
    })


# ============================================================================
# EVENTS
# ============================================================================

@app.route("/api/events", methods=["GET"])
def get_events():
    """
    Real curated event dataset (29 events, 1990-2024).

    Query Parameters:
        category (str, optional) -- filter by category
        start_date, end_date (str, optional)
    """
    category = request.args.get("category")
    start = request.args.get("start_date")
    end = request.args.get("end_date")

    data = EVENTS
    if category:
        data = [e for e in data if e["category"].lower() == category.lower()]
    data = filter_by_date(data, "date", start, end)

    categories = sorted({e["category"] for e in EVENTS})
    return jsonify({"data": data, "count": len(data), "available_categories": categories})


# ============================================================================
# CHANGE POINTS
# ============================================================================

@app.route("/api/changepoints", methods=["GET"])
def get_changepoints():
    """
    Change-point detection results.

    Query Parameters:
        method (str, optional): 'pelt' or 'bayesian' (default: both)

    PELT results come from a fast full-history scan (point estimates only).
    Bayesian results come from a focused single-change-point PyMC model
    run on specific windows (COVID, 2008 crisis) -- these include full
    posterior summaries: credible intervals and before/after parameter
    distributions, not just a point estimate.
    """
    method = request.args.get("method", "all")

    response = {}
    if method in ("all", "pelt"):
        response["pelt"] = PELT_CHANGE_POINTS
    if method in ("all", "bayesian"):
        response["bayesian"] = BAYESIAN_CASE_STUDIES
    return jsonify(response)


@app.route("/api/changepoints/<date>", methods=["GET"])
def get_changepoint_detail(date):
    """Detail for a single PELT-detected change point, by date (YYYY-MM-DD)."""
    match = next((cp for cp in PELT_CHANGE_POINTS if cp["date"] == date), None)
    if match is None:
        return jsonify({"error": f"No PELT change point found at {date}"}), 404
    return jsonify(match)


# ============================================================================
# EVENT CORRELATION DATA
# ============================================================================

@app.route("/api/correlations", methods=["GET"])
def get_correlations():
    """
    Category-level correlation between event types and detected price
    shifts, aggregated from real PELT change points matched to real
    events (within a 21-day tolerance window -- see src/analysis.py).

    NOTE: sample sizes here are small (single digits per category) --
    this reflects the true number of PELT-detected change points that
    happened to fall near a curated event, not a large statistical
    sample. Treat these as directional/illustrative, not conclusive.
    """
    cat_changes = defaultdict(list)
    cat_events = defaultdict(int)

    for cp in PELT_CHANGE_POINTS:
        seen_categories_this_cp = set()
        for m in cp["matched_events"]:
            cat = m["category"]
            cat_events[cat] += 1
            if cp["pct_price_change"] is not None and cat not in seen_categories_this_cp:
                cat_changes[cat].append(cp["pct_price_change"])
                seen_categories_this_cp.add(cat)

    correlations = []
    for cat in sorted(cat_events.keys()):
        changes = cat_changes.get(cat, [])
        correlations.append({
            "category": cat,
            "avg_pct_price_change": round(sum(changes) / len(changes), 2) if changes else None,
            "sample_size": len(changes),
            "matched_event_count": cat_events[cat],
        })

    return jsonify({
        "data": correlations,
        "note": "Aggregated from real PELT change points matched to curated events (21-day window). Small sample sizes -- see src/analysis.py match_change_points_to_events.",
    })


# ============================================================================
# EVENT IMPACT DETAIL
# ============================================================================

@app.route("/api/event-impact/<event_date>", methods=["GET"])
def get_event_impact(event_date):
    """
    For a given curated event date, find the nearest PELT change point
    (if any) and return its real before/after impact statistics.

    Path Parameters:
        event_date (str): YYYY-MM-DD
    """
    event = next((e for e in EVENTS if e["date"] == event_date), None)
    if event is None:
        return jsonify({"error": f"No curated event found at {event_date}"}), 404

    from datetime import date as _date
    ed = _date.fromisoformat(event_date)

    nearest = None
    nearest_gap = None
    for cp in PELT_CHANGE_POINTS:
        cd = _date.fromisoformat(cp["date"])
        gap = abs((cd - ed).days)
        if nearest_gap is None or gap < nearest_gap:
            nearest = cp
            nearest_gap = gap

    return jsonify({
        "event": event,
        "nearest_change_point": nearest,
        "days_between_event_and_change_point": nearest_gap,
        "caveat": "Proximity in time is not evidence of causation -- see Task 1 report, Section 2.",
    })


# ============================================================================
# DASHBOARD SUMMARY
# ============================================================================

@app.route("/api/dashboard/summary", methods=["GET"])
def get_dashboard_summary():
    """Aggregated real summary statistics for the dashboard landing view."""
    prices = [p["price"] for p in PRICE_SERIES]
    latest = PRICE_SERIES[-1]

    return jsonify({
        "latest_price": latest["price"],
        "latest_date": latest["date"],
        "all_time_max": max(prices),
        "all_time_min": min(prices),
        "data_range": METADATA["data_range"],
        "n_observations": METADATA["n_observations"],
        "total_events": len(EVENTS),
        "pelt_change_points_detected": len(PELT_CHANGE_POINTS),
        "bayesian_case_studies_available": len(BAYESIAN_CASE_STUDIES),
        "last_updated": datetime.now().isoformat(),
    })


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "available_endpoints": [
            "/api/health",
            "/api/historical",
            "/api/events",
            "/api/changepoints",
            "/api/changepoints/<date>",
            "/api/correlations",
            "/api/event-impact/<event_date>",
            "/api/dashboard/summary",
        ],
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    print("=" * 60)
    print("Brent Oil Price Analysis Dashboard API")
    print("Serving REAL analysis results from data/results.json")
    print("=" * 60)
    print(f"\nData range: {METADATA['data_range']}, {METADATA['n_observations']} observations")
    print(f"Events: {len(EVENTS)}, PELT change points: {len(PELT_CHANGE_POINTS)}, Bayesian case studies: {len(BAYESIAN_CASE_STUDIES)}")
    print("\nAvailable endpoints:")
    print("  GET /api/health")
    print("  GET /api/historical?start_date=&end_date=")
    print("  GET /api/events?category=&start_date=&end_date=")
    print("  GET /api/changepoints?method=pelt|bayesian|all")
    print("  GET /api/changepoints/<date>")
    print("  GET /api/correlations")
    print("  GET /api/event-impact/<event_date>")
    print("  GET /api/dashboard/summary")
    print("=" * 60)
    print("\nStarting server on http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)