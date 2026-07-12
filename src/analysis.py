"""
Quantify the impact of detected change points and match them to real-world
events from src.events.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .events import events_in_window, get_events_df


def summarize_regime_shift(
    prices: pd.Series, change_date: pd.Timestamp, window_days: int = 30
) -> dict:
    """
    Quantify the price/volatility shift around a detected change point.

    Compares the `window_days` before the change point to the
    `window_days` after, on the raw price series.

    Parameters
    ----------
    prices : pd.Series
        Price series indexed by date.
    change_date : pd.Timestamp
        The detected change-point date.
    window_days : int
        Size of the before/after comparison windows, in calendar days.

    Returns
    -------
    dict with keys: mean_before, mean_after, pct_change,
    volatility_before, volatility_after, volatility_ratio
    """
    before = prices[
        (prices.index >= change_date - pd.Timedelta(days=window_days))
        & (prices.index < change_date)
    ]
    after = prices[
        (prices.index >= change_date)
        & (prices.index < change_date + pd.Timedelta(days=window_days))
    ]

    mean_before = before.mean()
    mean_after = after.mean()
    pct_change = (mean_after - mean_before) / mean_before * 100 if mean_before else np.nan

    vol_before = before.pct_change().std()
    vol_after = after.pct_change().std()
    vol_ratio = vol_after / vol_before if vol_before else np.nan

    return {
        "change_date": change_date,
        "mean_before": mean_before,
        "mean_after": mean_after,
        "pct_change": pct_change,
        "volatility_before": vol_before,
        "volatility_after": vol_after,
        "volatility_ratio": vol_ratio,
    }


def match_change_points_to_events(
    change_dates: list[pd.Timestamp], window_days: int = 14
) -> pd.DataFrame:
    """
    For each detected change point, find nearby curated events.

    Parameters
    ----------
    change_dates : list of pd.Timestamp
    window_days : int
        Tolerance window for matching (a change point rarely lands
        exactly on an event date, due to lags in market reaction/reporting).

    Returns
    -------
    pd.DataFrame
        One row per (change_date, matched event) pair. Change points with
        no nearby event get a row with NaN event fields.
    """
    events_df = get_events_df()
    rows = []

    for cd in change_dates:
        nearby = events_in_window(events_df, cd, window_days=window_days)
        if nearby.empty:
            rows.append(
                {"change_date": cd, "Event": None, "Category": None, "days_offset": None}
            )
        else:
            for _, ev in nearby.iterrows():
                rows.append(
                    {
                        "change_date": cd,
                        "Event": ev["Event"],
                        "Category": ev["Category"],
                        "days_offset": (ev["Date"] - cd).days,
                    }
                )

    return pd.DataFrame(rows)
