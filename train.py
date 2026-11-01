"""Train the bankruptcy scorecard and the LightGBM benchmark.

Run:  python train.py
Reads:   data/5year.arff
Writes:  model/scorecard.json      the scorecard the app uses
         model/metrics.json        test-set results for both models
         model/iv_table.csv        Information Value of all 64 ratios
         model/test_companies.csv  scored test companies (for the app)
"""

import json
from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from src.ratios import RATIO_DEFINITIONS, short_name
from src.scorecard import pd_from_score, score_frame
from src.woe import MISSING, WoEBinner

SEED = 42
DATA = Path("data/5year.arff")
OUT = Path("model")

# Feature selection settings
MIN_IV = 0.1  # drop ratios that barely separate goods from bads
MAX_CORR = 0.6  # drop ratios that repeat a stronger one
MAX_FEATURES = 10
MIN_GAIN = 0.002  # stop adding ratios when CV AUC improves by less than this

# Scorecard scaling: 600 points = odds of 50 good : 1 bad, and every
# 20 points doubles the odds (PDO = points to double the odds)
BASE_SCORE, BASE_ODDS, PDO = 600, 50, 20

# Risk bands by probability of default (upper PD limit of each band)
BANDS = [("A", "Very low risk", 0.01), ("B", "Low risk", 0.03), ("C", "Medium risk", 0.08), ("D", "High risk", 0.20), ("E", "Very high risk", 1.0)]


# ---------------------------------------------------------------- 1. load data
def load_data():
    raw, _ = arff.loadarff(DATA)
    df = pd.DataFrame(raw)
    df["bankrupt"] = df.pop("class").str.decode("utf-8").astype(int)
    n_before = len(df)
    # 60 rows are exact copies of others. Drop them so the same company
    # can't land in both the training set and the test set.
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Loaded {n_before} rows, {n_before - len(df)} duplicates removed -> {len(df)} companies, "
          f"{df['bankrupt'].sum()} bankrupt ({df['bankrupt'].mean():.1%})")
    return df.drop(columns="bankrupt"), df["bankrupt"]


# ------------------------------------------------------- 2. feature selection
def select_features(binners, W, y):
    """IV filter -> correlation filter -> forward selection with CV AUC."""
    iv = pd.Series({c: b.iv for c, b in binners.items()}).sort_values(ascending=False)
    status = pd.Series("", index=iv.index)
    status[iv < MIN_IV] = f"IV below {MIN_IV}"

    # Correlation filter: walk from the strongest ratio down, keep a ratio only
    # if it is not strongly correlated with one already kept
    candidates = []
    for c in iv[iv >= MIN_IV].index:
        twin = next((k for k in candidates if abs(np.corrcoef(W[c], W[k])[0, 1]) >= MAX_CORR), None)
        if twin:
            status[c] = f"correlated with {twin}"
        else:
            candidates.append(c)

    # Forward selection: add the ratio that improves 5-fold CV AUC the most,
    # but only if every coefficient keeps the expected sign (negative, because
    # higher WoE = safer = lower chance of bankruptcy)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    selected, best, log = [], 0.0, []
    while len(selected) < MAX_FEATURES:
        scores = {}
        for c in candidates:
            if c in selected:
                continue
            cols = selected + [c]
            coef = LogisticRegression(max_iter=1000).fit(W[cols], y).coef_[0]
            if (coef >= 0).any():
                continue
            scores[c] = cross_val_score(LogisticRegression(max_iter=1000), W[cols], y, cv=cv, scoring="roc_auc").mean()
        if not scores:
            break
        c, s = max(scores.items(), key=lambda kv: kv[1])
        if selected and s - best < MIN_GAIN:
            break
        selected.append(c)
        log.append({"step": len(selected), "added": c, "name": short_name(c), "cv_auc": round(s, 4)})
        best = s
        print(f"  step {len(selected)}: + {c:<7} {short_name(c):<35} CV AUC {s:.4f}")

    for c in candidates:
        status[c] = "selected" if c in selected else "did not improve CV AUC enough"
    iv_table = pd.DataFrame({"ratio": iv.index, "name": [short_name(c) for c in iv.index],
                             "definition": [RATIO_DEFINITIONS[c] for c in iv.index],
                             "iv": iv.round(4).values, "status": status[iv.index].values})
    return selected, iv_table, log


# ---------------------------------------------------- 3. build the scorecard
def build_scorecard(binners, selected, model, X_train, y_train):
    factor = PDO / np.log(2)
    offset = BASE_SCORE - factor * np.log(BASE_ODDS)
    b0, n = model.intercept_[0], len(selected)

    def points(coef, woe):
        # Each ratio's share of: score = offset - factor * (b0 + sum(coef * woe))
        return int(round(-factor * (coef * woe + b0 / n) + offset / n))

    features = []
    for c, coef in zip(selected, model.coef_[0]):
        b = binners[c]
        table = b.table_.set_index("bin")
        bins = []
        for label in b.bin_labels():
            woe = b.woe_map.get(label, 0.0)
            row = table.loc[label] if label in table.index else None
            bins.append({
                "label": label,
                "woe": round(float(woe), 4),
                "points": points(coef, woe),
                "count": int(row["count"]) if row is not None else 0,
                "bad_rate": round(float(row["bad_rate"]), 4) if row is not None else None,
            })
        regular, missing_bin = bins[:-1], bins[-1]
        features.append({
            "name": c,
            "short_name": short_name(c),
            "definition": RATIO_DEFINITIONS[c],
            "iv": round(b.iv, 4),
            "coef": round(float(coef), 4),
            "median": round(float(X_train[c].median()), 4),  # a typical value, used as the app's default
            "edges": [float(e) for e in b.edges],
            "bins": regular,
            "missing_bin": missing_bin,
            "max_points": max(x["points"] for x in bins),
            "min_points": min(x["points"] for x in bins),
        })

    def score_at(pd_):
        return offset + factor * np.log((1 - pd_) / pd_)

    bands, prev_pd = [], 0.0
    for code, label, max_pd in BANDS:
        min_score = -10_000 if max_pd >= 1 else int(np.ceil(score_at(max_pd)))
        bands.append({"band": code, "label": label, "pd_from": prev_pd, "pd_to": max_pd, "min_score": min_score})
        prev_pd = max_pd

    return {
        "meta": {
            "dataset": "Polish companies bankruptcy (UCI ML Repository, id 365), file 5year.arff",
            "target": "bankrupt within 1 year (1 = bankrupt)",
            "trained_on": str(date.today()),
            "n_train": int(len(X_train)),
            "train_bad_rate": round(float(y_train.mean()), 4),
        },
        "scaling": {"base_score": BASE_SCORE, "base_odds": BASE_ODDS, "pdo": PDO,
                    "factor": round(factor, 6), "offset": round(offset, 6)},
        "intercept": round(float(b0), 4),
        "score_range": [sum(f["min_points"] for f in features), sum(f["max_points"] for f in features)],
        "features": features,
        "bands": bands,
    }


# ------------------------------------------------------------- 4. evaluation
def metrics(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    auc = roc_auc_score(y, p)
    return {"auc": round(auc, 4), "gini": round(2 * auc - 1, 4), "ks": round(float(np.max(tpr - fpr)), 4)}


def bootstrap_auc(y, p_card, p_gbm, n=1000):
    """95% confidence intervals: the test set has only ~120 bankruptcies."""
    rng = np.random.default_rng(SEED)
    y, p_card, p_gbm = map(np.asarray, (y, p_card, p_gbm))
    card, gbm, diff = [], [], []
    while len(card) < n:
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        a, b = roc_auc_score(y[i], p_card[i]), roc_auc_score(y[i], p_gbm[i])
        card.append(a)
        gbm.append(b)
        diff.append(b - a)

    def ci(v):
        return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]

    return {"scorecard_auc_ci": ci(card), "lightgbm_auc_ci": ci(gbm), "auc_gap_ci": ci(diff)}


def roc_points(y, p, k=60):
    fpr, tpr, _ = roc_curve(y, p)
    idx = np.unique(np.linspace(0, len(fpr) - 1, k).astype(int))
    return {"fpr": fpr[idx].round(4).tolist(), "tpr": tpr[idx].round(4).tolist()}


def main():
    OUT.mkdir(exist_ok=True)
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
    print(f"Train: {len(X_train)} companies ({y_train.sum()} bankrupt) | Test: {len(X_test)} ({y_test.sum()} bankrupt)")

    # WoE binning is learned on the training set only
    binners = {c: WoEBinner().fit(X_train[c], y_train) for c in X.columns}
    W_train = pd.DataFrame({c: b.transform(X_train[c]) for c, b in binners.items()})

    print("Feature selection:")
    selected, iv_table, selection_log = select_features(binners, W_train, y_train)

    model = LogisticRegression(max_iter=1000).fit(W_train[selected], y_train)
    card = build_scorecard(binners, selected, model, X_train, y_train)

    # Evaluate the integer-points scorecard exactly as the app will use it
    test_scores = score_frame(card, X_test)
    train_scores = score_frame(card, X_train)
    pd_test = pd_from_score(card, test_scores)
    pd_train = pd_from_score(card, train_scores)

    # Benchmark: gradient boosting on all 64 raw ratios, no binning
    gbm = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=20,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                             random_state=SEED, n_jobs=1, verbose=-1)
    gbm.fit(X_train, y_train)
    gbm_test = gbm.predict_proba(X_test)[:, 1]

    bands = pd.DataFrame({"score": test_scores, "bankrupt": y_test.values})
    bands["band"] = [next(b["band"] for b in card["bands"] if s >= b["min_score"]) for s in bands["score"]]
    band_table = (bands.groupby("band")["bankrupt"].agg(companies="count", bankrupt="sum")
                  .reindex([b["band"] for b in card["bands"]], fill_value=0).reset_index())
    band_table["bankrupt_rate"] = (band_table["bankrupt"] / band_table["companies"].replace(0, np.nan)).round(4)

    results = {
        "data": {"companies": int(len(X)), "bankrupt": int(y.sum()), "train": int(len(X_train)),
                 "test": int(len(X_test)), "test_bankrupt": int(y_test.sum())},
        "scorecard": {"train": metrics(y_train, pd_train), "test": metrics(y_test, pd_test),
                      "n_features": len(selected)},
        "lightgbm": {"test": metrics(y_test, gbm_test), "n_features": X.shape[1]},
        "bootstrap": bootstrap_auc(y_test, pd_test, gbm_test),
        "roc": {"scorecard": roc_points(y_test, pd_test), "lightgbm": roc_points(y_test, gbm_test)},
        "bands_test": band_table.to_dict(orient="records"),
        "selection_log": selection_log,
    }

    (OUT / "scorecard.json").write_text(json.dumps(card, indent=2))
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2))
    iv_table.to_csv(OUT / "iv_table.csv", index=False)
    test_out = X_test[selected].copy()
    test_out.insert(0, "company_id", X_test.index)
    test_out["bankrupt"] = y_test.values
    test_out["score"] = test_scores.values
    test_out["pd_scorecard"] = np.round(pd_test, 4)
    test_out["pd_lightgbm"] = np.round(gbm_test, 4)
    test_out.to_csv(OUT / "test_companies.csv", index=False)

    s, g, bs = results["scorecard"], results["lightgbm"], results["bootstrap"]
    print(f"\nScorecard ({len(selected)} ratios)  test AUC {s['test']['auc']}  Gini {s['test']['gini']}  KS {s['test']['ks']}"
          f"  (train AUC {s['train']['auc']})  95% CI {bs['scorecard_auc_ci']}")
    print(f"LightGBM  (64 ratios) test AUC {g['test']['auc']}  Gini {g['test']['gini']}  KS {g['test']['ks']}"
          f"  95% CI {bs['lightgbm_auc_ci']}")
    print(f"Score range {card['score_range']}  |  saved to {OUT}/")
    print(band_table.to_string(index=False))


if __name__ == "__main__":
    main()
