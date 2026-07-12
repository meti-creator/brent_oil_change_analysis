"""
Curated dataset of major political and economic events that have plausibly
affected Brent crude oil prices over roughly the past decade.

This is a starting point, not a definitive list — dates are to the day
where a specific decision/event occurred, and the 'category' field lets you
group by shock type (supply / demand / geopolitical / policy) for analysis.

IMPORTANT: verify/extend this list against primary sources (EIA, OPEC
press releases, Reuters/Bloomberg archives) before using it in a client
deliverable, and add any events from 2025-2026 onward, which are outside
this dataset's original compilation window.
"""

from __future__ import annotations

import pandas as pd

EVENTS = [
    # date, event, category, expected_direction
    ("1990-08-02", "Iraq invades Kuwait, triggering the Gulf War and a spike in oil prices", "Conflict", "up"),
    ("1991-01-17", "US-led coalition launches Operation Desert Storm; prices fall as supply fears ease", "Conflict", "down"),
    ("1997-07-02", "Asian Financial Crisis begins (Thai baht devaluation), later dragging down oil demand", "Demand shock", "down"),
    ("1998-12-01", "Oil prices bottom out amid Asian crisis demand collapse and OPEC oversupply", "Demand shock", "down"),
    ("2001-09-11", "September 11 attacks in the US trigger short-term market shock and demand uncertainty", "Conflict", "mixed"),
    ("2003-03-20", "US-led invasion of Iraq begins, raising Middle East supply-disruption risk", "Conflict", "up"),
    ("2008-07-11", "Brent crude peaks near record highs (~$147/bbl) amid strong pre-crisis demand and speculation", "Demand shock", "up"),
    ("2008-09-15", "Lehman Brothers collapses, triggering the Global Financial Crisis and a collapse in oil demand", "Demand shock", "down"),
    ("2008-12-31", "OPEC announces large output cuts in response to the financial-crisis price collapse", "OPEC policy", "up"),
    ("2011-02-15", "Libyan civil war erupts, disrupting Libyan oil exports amid the wider Arab Spring", "Conflict", "up"),
    ("2014-11-27", "OPEC declines to cut production despite oversupply, accelerating the 2014-16 price crash", "OPEC policy", "down"),
    ("2016-11-30", "OPEC agrees to first production cut since 2008 (Vienna agreement)", "OPEC policy", "up"),
    ("2016-12-10", "Non-OPEC producers (incl. Russia) join OPEC in coordinated cuts, forming OPEC+", "OPEC policy", "up"),
    ("2018-05-08", "US withdraws from the Iran nuclear deal; sanctions on Iranian oil exports reinstated", "Sanctions", "up"),
    ("2018-11-05", "US grants waivers to major buyers of Iranian oil, easing supply fears", "Sanctions", "down"),
    ("2019-09-14", "Drone/missile attack on Saudi Aramco's Abqaiq and Khurais facilities knocks out ~5% of global supply", "Conflict", "up"),
    ("2020-03-08", "Saudi-Russia price war begins after OPEC+ talks collapse; Saudi Arabia slashes prices", "OPEC policy", "down"),
    ("2020-04-20", "WTI futures briefly turn negative amid COVID-19 demand collapse and storage shortage", "Demand shock", "down"),
    ("2020-04-12", "OPEC+ agrees record ~9.7 million bpd production cut to stabilize the COVID-crushed market", "OPEC policy", "up"),
    ("2021-07-18", "OPEC+ agrees to gradually increase output through 2022 as demand recovers", "OPEC policy", "down"),
    ("2022-02-24", "Russia invades Ukraine, triggering a sharp spike in oil and gas prices on supply-disruption fears", "Conflict", "up"),
    ("2022-03-08", "US bans imports of Russian oil, gas, and coal", "Sanctions", "up"),
    ("2022-10-05", "OPEC+ announces a large production cut (2 million bpd) despite US pressure to increase supply", "OPEC policy", "up"),
    ("2022-12-05", "EU embargo on Russian seaborne crude and G7 price cap on Russian oil take effect", "Sanctions", "mixed"),
    ("2023-04-02", "Saudi Arabia and other OPEC+ members announce surprise voluntary output cuts", "OPEC policy", "up"),
    ("2023-06-04", "Saudi Arabia announces additional unilateral 1 million bpd cut", "OPEC policy", "up"),
    ("2023-10-07", "Hamas attack on Israel and start of the Israel-Gaza war raise Middle East risk premium", "Conflict", "up"),
    ("2024-01-12", "US and UK begin airstrikes on Houthi targets in Yemen after Red Sea shipping attacks", "Conflict", "up"),
    ("2024-04-13", "Iran launches direct drone/missile attack on Israel, escalating regional tensions", "Conflict", "up"),
]


def get_events_df() -> pd.DataFrame:
    """
    Return the curated events list as a DataFrame sorted by date.

    Returns
    -------
    pd.DataFrame
        Columns: Date (datetime64), Event (str), Category (str),
        ExpectedDirection (str: 'up'/'down'/'mixed').
    """
    df = pd.DataFrame(EVENTS, columns=["Date", "Event", "Category", "ExpectedDirection"])
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def events_in_window(
    events_df: pd.DataFrame, center_date: pd.Timestamp, window_days: int = 14
) -> pd.DataFrame:
    """
    Return events falling within +/- window_days of a given date.

    Useful for matching a detected change point to nearby real-world events.
    """
    lower = center_date - pd.Timedelta(days=window_days)
    upper = center_date + pd.Timedelta(days=window_days)
    mask = (events_df["Date"] >= lower) & (events_df["Date"] <= upper)
    return events_df.loc[mask].copy()
