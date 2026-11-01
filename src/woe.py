"""Weight of Evidence (WoE) binning for one numeric feature.

How it works, in four steps:
1. Cut the feature into ~20 equal-sized bins (quantiles) using training data.
2. Merge neighbouring bins until every bin is big enough to be trusted.
3. Merge neighbouring bins until the bad rate only moves in one direction
   (monotonic), so the scorecard is easy to explain.
4. Give missing values their own bin.

WoE convention used here:  WoE = ln(share of goods in bin / share of bads in bin)
    WoE > 0  -> the bin is safer than average
    WoE < 0  -> the bin is riskier than average
Information Value (IV) = sum over bins of (share of goods - share of bads) * WoE.
It measures how well the feature separates goods from bads.
"""

import numpy as np
import pandas as pd

MISSING = "Missing"


class WoEBinner:
    def __init__(self, initial_bins=20, max_bins=6, min_bin_share=0.05, min_bads=5, smoothing=0.5):
        self.initial_bins = initial_bins
        self.max_bins = max_bins
        self.min_bin_share = min_bin_share  # each bin needs at least this share of rows
        self.min_bads = min_bads  # ... and at least this many bankrupt companies
        self.smoothing = smoothing  # added to counts so WoE never divides by zero

    # ------------------------------------------------------------------ fitting
    def fit(self, x: pd.Series, y: pd.Series):
        x = pd.Series(x, dtype=float)
        y = pd.Series(y, dtype=int).loc[x.index]
        present = x.notna()
        xs, ys = x[present].to_numpy(), y[present].to_numpy()

        # Step 1: start from quantile edges (interior cut points only)
        qs = np.linspace(0, 1, self.initial_bins + 1)[1:-1]
        edges = np.unique(np.quantile(xs, qs))

        # Step 2 + 3: merge until bins are large, monotonic and few enough
        edges = self._merge(xs, ys, edges)
        self.edges = edges

        # Step 4: statistics for every bin, including the missing bin
        self.total_goods = int((y == 0).sum())
        self.total_bads = int((y == 1).sum())
        labels = self._assign(x)
        table = self._stats(labels, y)
        # A missing bin that is too small to trust gets WoE = 0 (average risk),
        # the same size rule as every other bin. This also stops the model from
        # learning data-collection quirks, e.g. a ratio that is only missing for
        # firms that went bankrupt.
        is_small_missing = (table["bin"] == MISSING) & (table["count"] < self.min_bin_share * len(x))
        table.loc[is_small_missing, ["woe", "iv"]] = 0.0
        self.table_ = table
        self.woe_map = dict(zip(table["bin"], table["woe"]))
        # No missing values in training data -> treat missing as average risk
        self.woe_map.setdefault(MISSING, 0.0)
        self.iv = float(table["iv"].sum())
        return self

    def _counts(self, xs, ys, edges):
        idx = np.searchsorted(edges, xs, side="left")  # bins are (left, right]
        n_bins = len(edges) + 1
        bads = np.bincount(idx, weights=ys, minlength=n_bins)
        total = np.bincount(idx, minlength=n_bins).astype(float)
        return total, bads

    def _merge(self, xs, ys, edges):
        n = len(xs)
        edges = list(edges)

        def bad_rates(total, bads):
            return bads / np.maximum(total, 1)

        # Step 2: merge bins that are too small (too few rows or too few bads)
        while edges:
            total, bads = self._counts(xs, ys, np.array(edges))
            small = np.where((total < self.min_bin_share * n) | (bads < self.min_bads))[0]
            if len(small) == 0:
                break
            i = small[np.argmin(total[small])]
            rates = bad_rates(total, bads)
            # merge with the neighbour whose bad rate is closest
            if i == 0:
                drop = 0
            elif i == len(total) - 1:
                drop = i - 1
            else:
                drop = i - 1 if abs(rates[i] - rates[i - 1]) <= abs(rates[i] - rates[i + 1]) else i
            edges.pop(drop)

        # Direction of the trend: does risk rise or fall as the ratio grows?
        if edges:
            corr = np.corrcoef(xs.argsort().argsort(), ys)[0, 1]  # rank correlation
            increasing = corr > 0

        # Step 3: merge until the bad rate is monotonic in that direction
        while edges:
            total, bads = self._counts(xs, ys, np.array(edges))
            diffs = np.diff(bad_rates(total, bads))
            bad_steps = np.where(diffs < 0)[0] if increasing else np.where(diffs > 0)[0]
            if len(bad_steps) == 0:
                break
            edges.pop(bad_steps[np.argmin(np.abs(diffs[bad_steps]))])

        # Keep at most max_bins: merge the two neighbours with the most similar bad rate
        while len(edges) + 1 > self.max_bins:
            total, bads = self._counts(xs, ys, np.array(edges))
            edges.pop(int(np.argmin(np.abs(np.diff(bad_rates(total, bads))))))

        return np.array(edges)

    # ---------------------------------------------------------------- applying
    def bin_labels(self):
        """Readable labels like '(-inf, 0.12]' in bin order, then Missing."""
        bounds = [-np.inf, *self.edges, np.inf]
        return [f"({_fmt(lo)}, {_fmt(hi)}]" for lo, hi in zip(bounds[:-1], bounds[1:])] + [MISSING]

    def _assign(self, x):
        x = pd.Series(x, dtype=float)
        labels = np.array(self.bin_labels(), dtype=object)
        idx = np.searchsorted(self.edges, x.fillna(0).to_numpy(), side="left")
        out = labels[idx]
        out[x.isna().to_numpy()] = MISSING
        return pd.Series(out, index=x.index)

    def _stats(self, labels, y):
        df = pd.DataFrame({"bin": labels, "bad": y.to_numpy()})
        g = df.groupby("bin")["bad"].agg(["count", "sum"]).reindex(self.bin_labels()).fillna(0)
        g.columns = ["count", "bads"]
        g = g[g["count"] > 0].copy()
        g["goods"] = g["count"] - g["bads"]
        g["bad_rate"] = g["bads"] / g["count"]
        dist_good = (g["goods"] + self.smoothing) / (self.total_goods + self.smoothing * len(g))
        dist_bad = (g["bads"] + self.smoothing) / (self.total_bads + self.smoothing * len(g))
        g["woe"] = np.log(dist_good / dist_bad)
        g["iv"] = (dist_good - dist_bad) * g["woe"]
        return g.reset_index()

    def transform(self, x):
        return self._assign(x).map(self.woe_map).astype(float)


def _fmt(v):
    if np.isinf(v):
        return "-inf" if v < 0 else "inf"
    return f"{v:.4g}"
