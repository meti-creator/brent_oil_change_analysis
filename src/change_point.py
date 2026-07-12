"""
Change-point detection on Brent log-return series.

Two complementary approaches:

1. `bayesian_single_change_point` — a PyMC model assuming exactly one
   change point in the mean/volatility of log returns. Returns the full
   posterior over the change-point location (as an index into the series),
   which lets you express uncertainty ("the shift most likely occurred
   between day X and day Y") rather than a single point estimate.

2. `ruptures_change_points` — a fast frequentist baseline (PELT algorithm)
   for detecting multiple change points, useful as a cross-check and for
   an initial scan before committing to Bayesian inference (which is
   slower and, in its simple form here, assumes a single change point).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def bayesian_single_change_point(
    series: pd.Series,
    draws: int = 2000,
    tune: int = 2000,
    chains: int = 4,
    random_seed: int = 42,
):
    """
    Fit a single-change-point Bayesian model to a log-return series.

    Model:
        tau  ~ DiscreteUniform(0, n-1)              # change point location
        mu_1, mu_2       ~ Normal(0, sigma=0.1)      # pre/post mean return
        sigma_1, sigma_2 ~ HalfNormal(sigma=0.1)     # pre/post volatility
        returns[t] ~ Normal(mu_1, sigma_1) if t < tau else Normal(mu_2, sigma_2)

    Parameters
    ----------
    series : pd.Series
        Log-return series (e.g. from data_loader.add_log_returns), indexed
        by date.
    draws, tune, chains : int
        PyMC sampling parameters.
    random_seed : int

    Returns
    -------
    dict with keys:
        'trace'       : the PyMC InferenceData object (full posterior)
        'tau_mode'    : most probable change-point index (int)
        'tau_date'    : corresponding date from series.index
        'mu_1_mean', 'mu_2_mean' : posterior mean returns before/after
        'sigma_1_mean', 'sigma_2_mean' : posterior mean volatility before/after
    """
    import pymc as pm

    y = series.values
    n = len(y)
    idx = np.arange(n)

    with pm.Model() as model:
        tau = pm.DiscreteUniform("tau", lower=0, upper=n - 1)

        mu_1 = pm.Normal("mu_1", mu=0, sigma=0.1)
        mu_2 = pm.Normal("mu_2", mu=0, sigma=0.1)
        sigma_1 = pm.HalfNormal("sigma_1", sigma=0.1)
        sigma_2 = pm.HalfNormal("sigma_2", sigma=0.1)

        mu = pm.math.switch(tau >= idx, mu_1, mu_2)
        sigma = pm.math.switch(tau >= idx, sigma_1, sigma_2)

        pm.Normal("obs", mu=mu, sigma=sigma, observed=y)

        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=random_seed,
            target_accept=0.9,
            progressbar=False,
        )

    tau_samples = trace.posterior["tau"].values.flatten()
    tau_mode = int(np.bincount(tau_samples).argmax())
    tau_date = series.index[tau_mode]

    return {
        "trace": trace,
        "tau_samples": tau_samples,
        "tau_mode": tau_mode,
        "tau_date": tau_date,
        "mu_1_mean": float(trace.posterior["mu_1"].values.mean()),
        "mu_2_mean": float(trace.posterior["mu_2"].values.mean()),
        "sigma_1_mean": float(trace.posterior["sigma_1"].values.mean()),
        "sigma_2_mean": float(trace.posterior["sigma_2"].values.mean()),
    }


def ruptures_change_points(
    series: pd.Series, model: str = "rbf", penalty: float = 10.0
) -> list[pd.Timestamp]:
    """
    Detect multiple change points using the PELT algorithm (ruptures library).

    Parameters
    ----------
    series : pd.Series
        Log-return (or price) series, indexed by date.
    model : str
        Cost model for ruptures ('rbf', 'l2', 'l1', etc.).
    penalty : float
        Penalty term controlling sensitivity — lower values detect more
        change points. Tune based on series length and noise level.

    Returns
    -------
    list of pd.Timestamp
        Dates of detected change points (excludes the final endpoint that
        ruptures always includes).
    """
    import ruptures as rpt

    signal = series.values.reshape(-1, 1)
    algo = rpt.Pelt(model=model).fit(signal)
    breakpoints = algo.predict(pen=penalty)

    # ruptures includes len(signal) as the last "breakpoint"; drop it
    breakpoints = [b for b in breakpoints if b < len(series)]

    return [series.index[b] for b in breakpoints]
