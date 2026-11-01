"""Streamlit app: score a company's bankruptcy risk with the trained scorecard.

Run locally:  streamlit run streamlit_app.py
"""

import json
import math
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.scorecard import load_scorecard, score_company

ROOT = Path(__file__).parent
BAND_COLORS = {"A": "#1D9E75", "B": "#639922", "C": "#BA7517", "D": "#D85A30", "E": "#A32D2D"}

st.set_page_config(page_title="Company bankruptcy scorecard", page_icon=":material/monitoring:", layout="wide")


@st.cache_data
def load_assets():
    card = load_scorecard(ROOT / "model" / "scorecard.json")
    metrics = json.loads((ROOT / "model" / "metrics.json").read_text())
    iv = pd.read_csv(ROOT / "model" / "iv_table.csv")
    test = pd.read_csv(ROOT / "model" / "test_companies.csv")
    return card, metrics, iv, test


card, metrics, iv_table, test_companies = load_assets()
features = card["features"]


def input_key(name):
    return f"input_{name}"


def set_inputs(values, example=None):
    for f in features:
        v = values.get(f["name"])
        st.session_state[input_key(f["name"])] = None if v is None or pd.isna(v) else float(v)
    st.session_state["example"] = example


def load_typical():
    set_inputs({f["name"]: f["median"] for f in features})


def load_example(bankrupt):
    pool = test_companies[test_companies["bankrupt"] == bankrupt]
    row = pool.sample(1).iloc[0]
    set_inputs(row.to_dict(), example={"id": int(row["company_id"]), "bankrupt": int(row["bankrupt"])})


if input_key(features[0]["name"]) not in st.session_state:
    load_typical()


# ------------------------------------------------------------------- header
s_test = metrics["scorecard"]["test"]
st.title("Company bankruptcy scorecard")
st.caption(
    f"Predicts whether a company goes bankrupt within a year from {len(features)} financial ratios. "
    f"Logistic regression scorecard on WoE-binned ratios, trained on {metrics['data']['train']:,} Polish companies. "
    f"Test AUC {s_test['auc']:.3f}, Gini {s_test['gini']:.3f}."
)

tab_score, tab_perf, tab_card, tab_about = st.tabs(["Score a company", "Model performance", "Scorecard", "Method"])

# ---------------------------------------------------------- score a company
with tab_score:
    left, right = st.columns([2, 3], gap="large")

    with left:
        st.subheader("Financial ratios")
        st.caption("Type in a company's ratios, or load an example:")
        b1, b2, b3 = st.columns(3)
        b1.button("Typical", on_click=load_typical, width="stretch",
                  help="The median value of each ratio in the training data")
        b2.button("Healthy", on_click=load_example, args=(0,), width="stretch",
                  help="A random real company from the test set that did not go bankrupt")
        b3.button("Bankrupt", on_click=load_example, args=(1,), width="stretch",
                  help="A random real company from the test set that went bankrupt")
        for f in features:
            st.number_input(
                f["short_name"],
                value=None,  # allows an empty box, which means "not reported"
                key=input_key(f["name"]),
                step=0.01,
                format="%.4f",
                placeholder="Not reported",
                help=f"{f['definition']}. Leave empty if the company did not report it.",
            )

    values = {f["name"]: st.session_state[input_key(f["name"])] for f in features}
    result = score_company(card, values)
    band = result["band"]

    with right:
        st.subheader("Result")
        m1, m2, m3 = st.columns(3)
        m1.metric("Score", result["score"], help=f"Possible range {card['score_range'][0]} to {card['score_range'][1]}. Higher is safer.", border=True)
        m2.metric("Probability of bankruptcy", f"{result['pd']:.1%}", help="Chance of bankruptcy within 1 year, at the training sample's 7% base rate.", border=True)
        m3.metric(f"Risk band: {band['label'].lower()}", band["band"], border=True,
                  help=f"Band {band['band']} covers a probability of bankruptcy from {band['pd_from']:.0%} to {band['pd_to']:.0%}.")

        lo, hi = card["score_range"]
        st.progress((result["score"] - lo) / (hi - lo), text=f"{result['score']} on a scale of {lo} (riskiest) to {hi} (safest)")

        example = st.session_state.get("example")
        if example:
            outcome = "went bankrupt" if example["bankrupt"] else "did not go bankrupt"
            st.info(f"Test-set company #{example['id']}: this company actually **{outcome}**. "
                    "The model never saw it during training.", icon=":material/info:")

        st.markdown("**Main reasons for the score**")
        breakdown = result["breakdown"]
        reasons = breakdown[breakdown["points_lost"] > 0].sort_values("points_lost", ascending=False).head(3)
        if reasons.empty:
            st.write("Every ratio is in its safest range.")
        for _, r in reasons.iterrows():
            missing = r["value"] is None or (isinstance(r["value"], float) and math.isnan(r["value"]))
            shown = "is not reported" if missing else f"is {r['value']:.4g} (range {r['bin']})"
            st.markdown(f"- **{r['name']}** {shown}: **{r['points_lost']} points** below the safest range")

        st.markdown("**Points by ratio**")
        st.dataframe(
            breakdown[["name", "value", "bin", "points", "best_points"]],
            hide_index=True,
            column_config={
                "name": "Ratio",
                "value": st.column_config.NumberColumn("Value", format="%.4g"),
                "bin": "Range",
                "points": st.column_config.ProgressColumn("Points", min_value=0, max_value=int(breakdown["best_points"].max()),
                                                          format="%d", color="#534AB7"),
                "best_points": st.column_config.NumberColumn("Best possible", format="%d"),
            },
        )

# --------------------------------------------------------- model performance
with tab_perf:
    g_test = metrics["lightgbm"]["test"]
    boot = metrics["bootstrap"]
    st.subheader("Scorecard vs. black-box benchmark")
    st.write(
        f"Both models were tested on the same {metrics['data']['test']:,} companies "
        f"({metrics['data']['test_bankrupt']} bankruptcies) that neither model saw in training."
    )
    comparison = pd.DataFrame({
        "Model": ["Scorecard (logistic regression)", "LightGBM benchmark"],
        "Ratios used": [metrics["scorecard"]["n_features"], metrics["lightgbm"]["n_features"]],
        "Test AUC": [s_test["auc"], g_test["auc"]],
        "AUC 95% CI": [f"{boot['scorecard_auc_ci'][0]:.3f} – {boot['scorecard_auc_ci'][1]:.3f}",
                       f"{boot['lightgbm_auc_ci'][0]:.3f} – {boot['lightgbm_auc_ci'][1]:.3f}"],
        "Gini": [s_test["gini"], g_test["gini"]],
        "KS": [s_test["ks"], g_test["ks"]],
        "Explains each decision": ["Yes, points per ratio", "No, needs extra tools like SHAP"],
    })
    st.dataframe(comparison, hide_index=True)
    gap = g_test["auc"] - s_test["auc"]
    st.write(
        f"LightGBM ranks companies better by {gap:.3f} AUC (95% CI {boot['auc_gap_ci'][0]:.3f} to "
        f"{boot['auc_gap_ci'][1]:.3f}). The scorecard gives up that accuracy in exchange for using "
        f"{metrics['scorecard']['n_features']} ratios instead of 64 and for a score anyone can trace back to its inputs."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**ROC curve (test set)**")
        roc = pd.concat([
            pd.DataFrame({**metrics["roc"]["scorecard"], "Model": "Scorecard"}),
            pd.DataFrame({**metrics["roc"]["lightgbm"], "Model": "LightGBM"}),
        ])
        diagonal = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(strokeDash=[4, 4], color="gray").encode(x="x", y="y")
        curves = alt.Chart(roc).mark_line().encode(
            x=alt.X("fpr", title="Healthy firms flagged (FPR)"),
            y=alt.Y("tpr", title="Bankruptcies caught (TPR)"),
            color=alt.Color("Model", scale=alt.Scale(domain=["Scorecard", "LightGBM"], range=["#534AB7", "#888780"]),
                            legend=alt.Legend(orient="bottom", title=None)),
        )
        st.altair_chart((diagonal + curves).properties(height=340), width="stretch")
    with c2:
        st.markdown("**Bankruptcy rate by risk band (test set)**")
        bands = pd.DataFrame(metrics["bands_test"])
        bands["label"] = bands["band"] + " · " + bands["companies"].astype(str) + " firms"
        base = alt.Chart(bands).encode(
            x=alt.X("label", sort=None, title="Risk band", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("bankrupt_rate", title="Went bankrupt", axis=alt.Axis(format="%")),
            tooltip=["band", "companies", "bankrupt", alt.Tooltip("bankrupt_rate", format=".1%")],
        )
        bars = base.mark_bar().encode(
            color=alt.Color("band", scale=alt.Scale(domain=list(BAND_COLORS), range=list(BAND_COLORS.values())), legend=None))
        labels = base.mark_text(dy=-8).encode(text=alt.Text("bankrupt_rate", format=".1%"))
        st.altair_chart((bars + labels).properties(height=340), width="stretch")

    st.markdown("**Score distribution (test set)**")
    st.caption("Each group adds up to 100%, so the much smaller bankrupt group stays visible.")
    dist = test_companies.assign(
        Outcome=np.where(test_companies["bankrupt"] == 1, "Went bankrupt", "Did not go bankrupt"),
        bucket=(test_companies["score"] // 10) * 10,
    )
    shares = dist.groupby(["Outcome", "bucket"]).size().rename("companies").reset_index()
    shares["share"] = shares["companies"] / shares.groupby("Outcome")["companies"].transform("sum")
    hist = alt.Chart(shares).mark_bar().encode(
        x=alt.X("bucket:O", title="Score band start (higher is safer)", axis=alt.Axis(labelAngle=0)),
        xOffset="Outcome",
        y=alt.Y("share", title="Share of group", axis=alt.Axis(format="%")),
        color=alt.Color("Outcome", scale=alt.Scale(domain=["Did not go bankrupt", "Went bankrupt"], range=["#378ADD", "#D85A30"]),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["Outcome", alt.Tooltip("bucket", title="Score from"), "companies", alt.Tooltip("share", format=".1%")],
    )
    st.altair_chart(hist.properties(height=280), width="stretch")

# ---------------------------------------------------------------- scorecard
with tab_card:
    st.subheader("The scorecard")
    sc = card["scaling"]
    st.write(
        f"Add up the points for each ratio to get the score. {sc['base_score']} points means odds of "
        f"{sc['base_odds']} healthy to 1 bankrupt, and every {sc['pdo']} extra points doubles those odds."
    )
    for f in features:
        with st.expander(f"{f['short_name']}  ·  {f['min_points']}–{f['max_points']} points  ·  IV {f['iv']:.2f}"):
            st.caption(f"{f['name']}: {f['definition']}")
            rows = pd.DataFrame(f["bins"] + [f["missing_bin"]])
            st.dataframe(
                rows[["label", "count", "bad_rate", "woe", "points"]],
                hide_index=True,
                column_config={
                    "label": "Range",
                    "count": st.column_config.NumberColumn("Training firms", format="%d"),
                    "bad_rate": st.column_config.NumberColumn("Went bankrupt", format="percent"),
                    "woe": st.column_config.NumberColumn("WoE", format="%.3f"),
                    "points": st.column_config.NumberColumn("Points", format="%d"),
                },
            )

    st.subheader("How the ratios were chosen")
    st.write("All 64 ratios ranked by Information Value (IV), with the reason each one was kept or dropped.")
    st.dataframe(iv_table, hide_index=True, height=360,
                 column_config={"iv": st.column_config.NumberColumn("IV", format="%.3f")})

# ------------------------------------------------------------------- method
with tab_about:
    st.subheader("How it works")
    st.markdown(
        f"""
1. **Data.** {metrics['data']['companies']:,} Polish companies (after removing 60 duplicate rows), each with 64
   financial ratios and a label for whether it went bankrupt within a year. {metrics['data']['bankrupt']} went bankrupt.
2. **Split.** 70% for training, 30% held back for testing, with the same bankruptcy rate in both.
3. **WoE binning.** Each ratio is cut into ranges learned from the training set only. Every range gets a
   Weight of Evidence: positive means safer than average, negative means riskier. Missing values get their own range.
4. **Feature selection.** Drop ratios with IV below 0.1, then drop ratios that repeat a stronger one
   (correlation above 0.6), then add ratios one at a time while 5-fold cross-validated AUC keeps improving.
5. **Logistic regression** on the WoE values, keeping only ratios whose effect points the expected way.
6. **Points.** The model is rescaled so {sc['base_score']} points = 50:1 odds and {sc['pdo']} points doubles the odds.
7. **Test.** Both the scorecard and a LightGBM benchmark are scored on the untouched test set.
"""
    )
    st.subheader("Limitations")
    st.markdown(
        """
- Probabilities reflect this dataset's 7% bankruptcy rate, which is far higher than in a typical loan book.
  A real deployment would recalibrate them to the portfolio's actual default rate.
- The data covers Polish companies from 2000–2013 and has no industry or macroeconomic information.
- Some ratios are missing far more often for bankrupt firms (for example, sales growth is missing for 24% of
  bankrupt firms but 0.07% of healthy ones). Small missing groups are treated as average risk so the model
  doesn't learn data-collection quirks.
"""
    )
    st.caption(
        "Data: Tomczak, S. (2016). Polish Companies Bankruptcy [Dataset]. UCI Machine Learning Repository. "
        "https://doi.org/10.24432/C5F600 (CC BY 4.0)"
    )
