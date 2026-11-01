"""Score companies with a saved scorecard (model/scorecard.json).

Only needs numpy and pandas, so the Streamlit app can use it without
scikit-learn. The training script uses the same functions, so the numbers in
the app and the numbers in the evaluation always match.
"""

import json
import math
from bisect import bisect_left
from pathlib import Path

import numpy as np
import pandas as pd


def load_scorecard(path="model/scorecard.json"):
    return json.loads(Path(path).read_text())


def _is_missing(value):
    return value is None or (isinstance(value, float) and math.isnan(value))


def find_bin(feature, value):
    """Return the bin dict a value falls into. Bins are (left, right]."""
    if _is_missing(value):
        return feature["missing_bin"]
    return feature["bins"][bisect_left(feature["edges"], value)]


def pd_from_score(card, score):
    """Score -> probability of default, using the scaling the card was built with.

    score = offset + factor * ln(good:bad odds)  =>  PD = 1 / (1 + odds)
    """
    s = card["scaling"]
    odds = np.exp((np.asarray(score, dtype=float) - s["offset"]) / s["factor"])
    return 1.0 / (1.0 + odds)


def band_for_score(card, score):
    for band in card["bands"]:  # ordered from safest to riskiest
        if score >= band["min_score"]:
            return band
    return card["bands"][-1]


def score_company(card, values):
    """Score one company.

    values: dict like {"Attr27": 1.5, "Attr25": None, ...}
    Returns the total score, PD, risk band and a per-ratio breakdown.
    """
    rows = []
    for f in card["features"]:
        value = values.get(f["name"])
        b = find_bin(f, value)
        rows.append(
            {
                "feature": f["name"],
                "name": f["short_name"],
                "value": value,
                "bin": b["label"],
                "points": b["points"],
                "best_points": f["max_points"],
                "points_lost": f["max_points"] - b["points"],
            }
        )
    breakdown = pd.DataFrame(rows)
    score = int(breakdown["points"].sum())
    return {
        "score": score,
        "pd": float(pd_from_score(card, score)),
        "band": band_for_score(card, score),
        "breakdown": breakdown,
    }


def score_frame(card, df):
    """Vectorised scoring for a DataFrame with one column per ratio."""
    total = np.zeros(len(df), dtype=int)
    for f in card["features"]:
        x = df[f["name"]].astype(float).to_numpy()
        idx = np.searchsorted(np.array(f["edges"], dtype=float), x, side="left")
        points = np.array([b["points"] for b in f["bins"]])[np.minimum(idx, len(f["bins"]) - 1)]
        points = np.where(np.isnan(x), f["missing_bin"]["points"], points)
        total += points
    return pd.Series(total, index=df.index, name="score")
