"""
Load and clean Brent crude oil price data.

Expects a CSV with at minimum a date column and a price column. Column
names are flexible (case-insensitive matching for common variants like
'Date'/'DATE'/'date' and 'Price'/'PRICE'/'Close').
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATE_CANDIDATES = ["date", "dates", "observation_date"]
PRICE_CANDIDATES = ["price", "close", "brent", "value", "dcoilbrenteu"]


def _find_column(columns: list[str], candidates: list[str]) -> str:
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    raise ValueError(
        f"Could not find a matching column among {columns}. "
        f"Expected one of: {candidates}"
    )


def load_brent_prices(
    filepath: str | Path,
    date_col: str | None = None,
    price_col: str | None = None,
) -> pd.DataFrame:
    """
    Load a Brent oil price CSV into a clean, sorted, indexed DataFrame.

    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file.
    date_col : str, optional
        Name of the date column. Auto-detected if not provided.
    price_col : str, optional
        Name of the price column. Auto-detected if not provided.

    Returns
    -------
    pd.DataFrame
        Indexed by Date (DatetimeIndex), single column 'Price', sorted
        ascending, with missing dates left as gaps (not forward-filled)
        so the caller can decide how to handle them.
    """
    df = pd.read_csv(filepath)

    if date_col is None:
        date_col = _find_column(list(df.columns), DATE_CANDIDATES)
    if price_col is None:
        price_col = _find_column(list(df.columns), PRICE_CANDIDATES)

    out = df[[date_col, price_col]].copy()
    out.columns = ["Date", "Price"]

    out["Date"] = pd.to_datetime(out["Date"], errors="coerce", format="mixed")
    out["Price"] = pd.to_numeric(out["Price"], errors="coerce")

    out = out.dropna(subset=["Date"]).sort_values("Date")
    out = out.set_index("Date")

    return out


def clean_prices(df: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
    """
    Handle missing/invalid price values.

    Parameters
    ----------
    df : pd.DataFrame
        Output of load_brent_prices (indexed by Date, has 'Price').
    method : str
        'ffill' (forward-fill), 'drop', or 'interpolate'.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with no NaN prices.
    """
    out = df.copy()
    out = out[~out.index.duplicated(keep="first")]

    if method == "ffill":
        out["Price"] = out["Price"].ffill()
    elif method == "drop":
        out = out.dropna(subset=["Price"])
    elif method == "interpolate":
        out["Price"] = out["Price"].interpolate(method="linear")
    else:
        raise ValueError(f"Unknown method: {method}")

    return out.dropna(subset=["Price"])


def add_log_returns(df: pd.DataFrame, price_col: str = "Price") -> pd.DataFrame:
    """
    Add a 'LogReturn' column: log(P_t / P_{t-1}).

    Log returns are used instead of raw prices for change-point detection
    because they are closer to stationary, which is an assumption of most
    change-point models.
    """
    out = df.copy()
    out["LogReturn"] = np.log(out[price_col] / out[price_col].shift(1))
    return out.dropna(subset=["LogReturn"])
