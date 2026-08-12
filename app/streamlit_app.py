from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="NFL Spread Prediction",
    page_icon="🏈",
    layout="wide",
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "nfl_spread_model.joblib"
DATA_DIR = ROOT / "app" / "data"
HOLDOUT_PATH = DATA_DIR / "streamlit_holdout_games.csv"
WALK_FORWARD_PATH = DATA_DIR / "walk_forward_thresholds.csv"
SCENARIO_PATH = DATA_DIR / "scenario_performance.csv"


@st.cache_resource
def load_model_bundle():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_holdout():
    return pd.read_csv(HOLDOUT_PATH)


@st.cache_data
def load_walk_forward():
    if WALK_FORWARD_PATH.exists():
        return pd.read_csv(WALK_FORWARD_PATH)
    return pd.DataFrame()


@st.cache_data
def load_scenarios():
    if SCENARIO_PATH.exists():
        return pd.read_csv(SCENARIO_PATH)
    return pd.DataFrame()


def margin_label(value):
    value = float(value)
    if value > 0:
        return f"Home by {value:.1f}"
    if value < 0:
        return f"Away by {abs(value):.1f}"
    return "Pick'em"


def betting_signal(edge, threshold):
    if edge >= threshold:
        return "BET HOME ATS"
    if edge <= -threshold:
        return "BET AWAY ATS"
    return "NO BET"


def historical_bet_result(signal, actual_margin, market_margin):
    if signal == "NO BET":
        return "No bet"

    if np.isclose(actual_margin, market_margin):
        return "Push"

    if signal == "BET HOME ATS":
        return "Win" if actual_margin > market_margin else "Loss"

    if signal == "BET AWAY ATS":
        return "Win" if actual_margin < market_margin else "Loss"

    return ""


try:
    bundle = load_model_bundle()
    model = bundle["model"]
    features = bundle["features"]
    holdout_df = load_holdout()
except Exception as exc:
    st.error(
        "The production model or app data could not be loaded. "
        "Make sure the exported files are in the repository."
    )
    st.exception(exc)
    st.stop()


st.title("NFL Spread Prediction")
st.caption(
    "Top 20 Random Forest-selected features + Ridge Regression | "
    "nflverse data, 2010–2025"
)

threshold = st.sidebar.slider(
    "Minimum model edge",
    min_value=0.5,
    max_value=4.5,
    value=float(bundle.get("default_edge_threshold", 3.5)),
    step=0.5,
)

st.sidebar.caption(
    "A bet is shown only when the model differs from the sportsbook "
    "by at least this many points."
)

explorer_tab, results_tab, model_tab = st.tabs(
    [
        "Historical Holdout Explorer",
        "Walk-Forward Results",
        "About the Model",
    ]
)


with explorer_tab:
    st.subheader("2024–2025 Holdout Game Explorer")
    st.write(
        "Choose a game from the unseen holdout period. The app reloads "
        "the saved model, uses that game's 20 pregame features, and "
        "recreates the prediction."
    )

    seasons = sorted(
        holdout_df["season"].dropna().astype(int).unique()
    )
    season = st.selectbox(
        "Season",
        seasons,
        index=len(seasons) - 1,
    )

    season_df = holdout_df[
        holdout_df["season"].astype(int) == season
    ].copy()

    weeks = sorted(
        season_df["week"].dropna().astype(int).unique()
    )
    week = st.selectbox("Week", weeks)

    week_df = season_df[
        season_df["week"].astype(int) == week
    ].copy()

    week_df["matchup"] = (
        week_df["away_team"].astype(str)
        + " at "
        + week_df["home_team"].astype(str)
    )

    matchup = st.selectbox(
        "Game",
        week_df["matchup"].tolist(),
    )

    selected = week_df[
        week_df["matchup"] == matchup
    ].iloc[[0]].copy()

    input_row = selected[features]
    prediction = float(model.predict(input_row)[0])

    market_margin = float(
        selected.iloc[0]["market_home_margin"]
    )
    actual_margin = float(
        selected.iloc[0]["home_margin"]
    )
    edge = prediction - market_margin

    signal = betting_signal(edge, threshold)
    result = historical_bet_result(
        signal,
        actual_margin,
        market_margin,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Sportsbook Expectation",
        margin_label(market_margin),
    )
    c2.metric(
        "Model Prediction",
        margin_label(prediction),
    )
    c3.metric(
        "Model Edge",
        f"{edge:+.2f} pts",
    )
    c4.metric(
        "Actual Final Margin",
        margin_label(actual_margin),
    )

    if signal == "NO BET":
        st.info(
            f"{signal} at a {threshold:.1f}-point threshold."
        )
    else:
        st.success(
            f"{signal} at a {threshold:.1f}-point threshold — "
            f"historical result: {result}."
        )

    with st.expander("Show the 20 model inputs"):
        feature_table = pd.DataFrame(
            {
                "Feature": features,
                "Value": [
                    selected.iloc[0][feature]
                    for feature in features
                ],
            }
        )
        st.dataframe(
            feature_table,
            hide_index=True,
            use_container_width=True,
        )


with results_tab:
    st.subheader("Walk-Forward Threshold Results")

    walk_forward = load_walk_forward()

    if walk_forward.empty:
        st.warning(
            "walk_forward_thresholds.csv was not found."
        )
    else:
        table = walk_forward.copy()

        if "win_rate_excluding_pushes" in table.columns:
            table["win_rate_excluding_pushes"] = (
                table["win_rate_excluding_pushes"] * 100
            ).round(1)

        if "roi" in table.columns:
            table["roi"] = (
                table["roi"] * 100
            ).round(1)

        display_cols = [
            col for col in [
                "threshold",
                "total_bets",
                "wins",
                "losses",
                "pushes",
                "win_rate_excluding_pushes",
                "total_profit",
                "roi",
            ]
            if col in table.columns
        ]

        st.dataframe(
            table[display_cols],
            hide_index=True,
            use_container_width=True,
        )

        if {"threshold", "roi"}.issubset(
            walk_forward.columns
        ):
            chart_df = (
                walk_forward[
                    ["threshold", "roi"]
                ]
                .set_index("threshold")
            )
            st.bar_chart(chart_df)

        st.write(
            "**Main result:** the historical walk-forward backtest was "
            "positive at the larger model edges, with the 3.5-point "
            "threshold producing the strongest aggregate ROI."
        )

    scenarios = load_scenarios()

    if not scenarios.empty:
        st.subheader("Selected Situational Results")

        scenario_view = scenarios.copy()

        if "total_bets" in scenario_view.columns:
            scenario_view = scenario_view[
                scenario_view["total_bets"] >= 25
            ]

        if "roi" in scenario_view.columns:
            scenario_view = scenario_view.sort_values(
                "roi",
                ascending=False,
            ).head(12)
            scenario_view["roi"] = (
                scenario_view["roi"] * 100
            ).round(1)

        scenario_cols = [
            col for col in [
                "scenario",
                "situation",
                "total_bets",
                "win_rate_excluding_pushes",
                "roi",
                "seasons_with_bets",
                "profitable_seasons",
            ]
            if col in scenario_view.columns
        ]

        st.dataframe(
            scenario_view[scenario_cols],
            hide_index=True,
            use_container_width=True,
        )


with model_tab:
    st.subheader("Model Information")

    holdout = bundle.get(
        "holdout_metrics",
        {},
    )

    st.markdown(
        f"""
        **Model:** {bundle.get("model_name", "Ridge Regression")}  
        **Training seasons:** {bundle.get("training_period", "2010–2023")}  
        **Final holdout:** {bundle.get("holdout_period", "2024–2025")}  
        **Target:** `{bundle.get("target", "home_margin")}`  
        **Feature count:** {len(features)}
        """
    )

    if holdout:
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Holdout MAE",
            f'{holdout.get("mae", float("nan")):.3f}',
        )
        c2.metric(
            "Sportsbook MAE",
            f'{holdout.get("sportsbook_mae", float("nan")):.3f}',
        )
        c3.metric(
            "Holdout R²",
            f'{holdout.get("r2", float("nan")):.3f}',
        )

    st.markdown("#### Final 20 Features")
    feature_df = pd.DataFrame(
        {
            "Number": range(1, len(features) + 1),
            "Feature": features,
        }
    )
    st.dataframe(
        feature_df,
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "This application is a capstone demonstration of historical "
        "model predictions and simulated betting thresholds. It is not "
        "a guarantee of future betting performance."
    )
