"""
Precompute analysis results for the dashboard backend.

Run this once (and re-run whenever the underlying analysis changes) to
regenerate backend/data/results.json. The Flask app only ever reads this
file -- it never re-runs PyMC sampling on a live request, since MCMC is
far too slow for interactive use.

Usage:
    python generate_results.py
"""

import os
os.environ["PYTENSOR_FLAGS"] = "cxx="

import json
import sys
sys.path.append("..")

import numpy as np
import pandas as pd
import pymc as pm

from src.data_loader import load_brent_prices, clean_prices, add_log_returns
from src.change_point import ruptures_change_points
from src.events import get_events_df
from src.analysis import summarize_regime_shift, match_change_points_to_events

DATA_PATH = "../data/BrentOilPrices.csv"
OUTPUT_PATH = "data/results.json"


def run_bayesian_window(returns, start, end, label):
    """Run the single-change-point Bayesian model on a focused window."""
    window = returns.loc[start:end, "LogReturn"]
    y = window.values
    n = len(y)
    idx = np.arange(n)

    with pm.Model():
        tau = pm.DiscreteUniform("tau", lower=0, upper=n - 1)
        mu_1 = pm.Normal("mu_1", mu=0, sigma=0.1)
        mu_2 = pm.Normal("mu_2", mu=0, sigma=0.1)
        mu = pm.math.switch(tau >= idx, mu_1, mu_2)
        sigma_1 = pm.HalfNormal("sigma_1", sigma=0.1)
        sigma_2 = pm.HalfNormal("sigma_2", sigma=0.1)
        sigma = pm.math.switch(tau >= idx, sigma_1, sigma_2)
        pm.Normal("obs", mu=mu, sigma=sigma, observed=y)
        trace = pm.sample(
            draws=4000, tune=4000, chains=4, cores=1,
            target_accept=0.95, random_seed=42, progressbar=False,
        )

    tau_samples = trace.posterior["tau"].values.flatten()
    tau_mode = int(np.bincount(tau_samples).argmax())
    tau_date = window.index[tau_mode]
    lb = int(np.percentile(tau_samples, 5))
    ub = min(int(np.percentile(tau_samples, 95)), n - 1)

    mu_1_s = trace.posterior["mu_1"].values.flatten()
    mu_2_s = trace.posterior["mu_2"].values.flatten()
    sigma_1_s = trace.posterior["sigma_1"].values.flatten()
    sigma_2_s = trace.posterior["sigma_2"].values.flatten()

    return {
        "label": label,
        "window_start": str(window.index.min().date()),
        "window_end": str(window.index.max().date()),
        "change_point_date": str(tau_date.date()),
        "credible_interval": [str(window.index[lb].date()), str(window.index[ub].date())],
        "mu_before": float(mu_1_s.mean()),
        "mu_after": float(mu_2_s.mean()),
        "sigma_before": float(sigma_1_s.mean()),
        "sigma_after": float(sigma_2_s.mean()),
        "p_volatility_decreased": float((sigma_2_s < sigma_1_s).mean()),
        "p_return_increased": float((mu_2_s > mu_1_s).mean()),
    }


def main():
    print("Loading price data...")
    prices = load_brent_prices(DATA_PATH)
    prices = clean_prices(prices, method="ffill")
    returns = add_log_returns(prices)

    print("Running PELT full-history scan...")
    pelt_dates = ruptures_change_points(returns["LogReturn"], model="l2", penalty=0.01)

    print("Running Bayesian model: COVID window...")
    covid_result = run_bayesian_window(returns, "2019-06-01", "2021-06-01", "COVID-19 / OPEC+ Emergency Cuts")

    print("Running Bayesian model: 2008 crisis window...")
    crisis_result = run_bayesian_window(returns, "2008-01-01", "2009-12-31", "Global Financial Crisis")

    bayesian_results = [covid_result, crisis_result]

    # Attach dollar-impact quantification + matched events to each Bayesian result
    for r in bayesian_results:
        cd = pd.Timestamp(r["change_point_date"])
        impact = summarize_regime_shift(prices["Price"], cd, window_days=30)
        r["mean_price_before"] = round(float(impact["mean_before"]), 2)
        r["mean_price_after"] = round(float(impact["mean_after"]), 2)
        r["pct_price_change"] = round(float(impact["pct_change"]), 2)

        matches = match_change_points_to_events([cd], window_days=100)
        matches = matches.dropna(subset=["Event"])
        r["matched_events"] = [
            {
                "event": m["Event"],
                "category": m["Category"],
                "days_offset": m["days_offset"],
            }
            for m in matches.to_dict(orient="records")
        ]

    print("Building PELT change-point summaries...")
    pelt_summary = []
    for d in pelt_dates:
        impact = summarize_regime_shift(prices["Price"], d, window_days=30)
        matches = match_change_points_to_events([d], window_days=21)
        matches = matches.dropna(subset=["Event"])
        pelt_summary.append({
            "date": str(d.date()),
            "mean_price_before": round(float(impact["mean_before"]), 2),
            "mean_price_after": round(float(impact["mean_after"]), 2),
            "pct_price_change": round(float(impact["pct_change"]), 2) if impact["pct_change"] == impact["pct_change"] else None,
            "volatility_ratio": round(float(impact["volatility_ratio"]), 3) if impact["volatility_ratio"] == impact["volatility_ratio"] else None,
            "matched_events": [
                {"event": m["Event"], "category": m["Category"], "days_offset": m["days_offset"]}
                for m in matches.to_dict(orient="records")
            ],
        })

    print("Serializing price series...")
    price_series = [
        {"date": str(d.date()), "price": round(float(p), 2)}
        for d, p in zip(prices.index, prices["Price"])
    ]

    print("Serializing events...")
    events_df = get_events_df()
    events = [
        {
            "date": str(row["Date"].date()),
            "event": row["Event"],
            "category": row["Category"],
            "expected_direction": row["ExpectedDirection"],
        }
        for _, row in events_df.iterrows()
    ]

    output = {
        "metadata": {
            "data_range": [str(prices.index.min().date()), str(prices.index.max().date())],
            "n_observations": len(prices),
            "generated_at": pd.Timestamp.now().isoformat(),
        },
        "price_series": price_series,
        "events": events,
        "pelt_change_points": pelt_summary,
        "bayesian_case_studies": bayesian_results,
    }

    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved results to {OUTPUT_PATH}")
    print(f"  - {len(price_series)} price observations")
    print(f"  - {len(events)} events")
    print(f"  - {len(pelt_summary)} PELT change points")
    print(f"  - {len(bayesian_results)} Bayesian case studies")


if __name__ == "__main__":
    main()
