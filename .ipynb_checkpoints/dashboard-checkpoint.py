import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


def load_utterances(results_folder: Path) -> pd.DataFrame:
    utterances_path = results_folder / "all_utterances.json"
    if not utterances_path.exists():
        st.error(f"File not found: {utterances_path}")
        st.stop()

    with open(utterances_path) as f:
        data = json.load(f)

    rows = []
    for entry in data:
        features = entry.get("features_dict", {})
        question = entry["utterance"]["question"].strip()
        if not question:
            continue
        row = {
            "question": question[:200],
            "answer": (entry["utterance"].get("answer") or "")[:300],
            "is_critical": entry.get("is_critical", False),
        }
        row.update(features)
        for k, v in entry.get("fitness", {}).items():
            if k != "distance":
                row[f"fitness_{k}"] = v
        rows.append(row)

    return pd.DataFrame(rows)


def get_discrete_features(df: pd.DataFrame) -> list[str]:
    exclude = {"question", "is_critical"}
    candidates = []
    for col in df.columns:
        if col in exclude or col.startswith("fitness_"):
            continue
        if df[col].nunique() <= 30:
            candidates.append(col)
    return candidates


def build_heatmap(df: pd.DataFrame, feat_x: str, feat_y: str):
    df_clean = df.dropna(subset=[feat_x, feat_y]).copy()
    df_clean[feat_x] = df_clean[feat_x].astype(str)
    df_clean[feat_y] = df_clean[feat_y].astype(str)

    grouped = df_clean.groupby([feat_y, feat_x]).agg(
        total=("is_critical", "count"),
        failures=("is_critical", "sum"),
    ).reset_index()
    grouped["failure_rate"] = grouped["failures"] / grouped["total"]

    pivot = grouped.pivot(index=feat_y, columns=feat_x, values="failure_rate")

    pivot.index = [s[:30] for s in pivot.index]
    pivot.columns = [s[:30] for s in pivot.columns]

    fig = px.imshow(
        pivot.values,
        x=list(pivot.columns),
        y=list(pivot.index),
        color_continuous_scale="RdYlGn_r",
        zmin=0,
        zmax=1,
        labels=dict(x=feat_x, y=feat_y, color="Failure Rate"),
        aspect="auto",
    )
    fig.update_traces(
        text=np.round(pivot.values, 2),
        texttemplate="%{text}",
        textfont_size=12,
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        height=max(400, len(pivot.index) * 40),
    )
    return fig


def build_count_heatmap(df: pd.DataFrame, feat_x: str, feat_y: str):
    df_clean = df.dropna(subset=[feat_x, feat_y]).copy()
    df_clean[feat_x] = df_clean[feat_x].astype(str)
    df_clean[feat_y] = df_clean[feat_y].astype(str)

    pivot = pd.crosstab(df_clean[feat_y], df_clean[feat_x])
    pivot.index = [s[:30] for s in pivot.index]
    pivot.columns = [s[:30] for s in pivot.columns]

    fig = px.imshow(
        pivot.values,
        x=list(pivot.columns),
        y=list(pivot.index),
        color_continuous_scale="Blues",
        labels=dict(x=feat_x, y=feat_y, color="Count"),
        aspect="auto",
    )
    fig.update_traces(
        text=pivot.values,
        texttemplate="%{text}",
        textfont_size=12,
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        height=max(400, len(pivot.index) * 40),
    )
    return fig


def find_run_dirs(root: Path) -> list[Path]:
    return sorted(
        [p.parent for p in root.rglob("all_utterances.json")],
        key=os.path.getmtime,
        reverse=True,
    )


def main():
    st.set_page_config(page_title="STELLAR Results Dashboard", layout="wide")
    st.title("STELLAR Results Dashboard")

    root_folder = st.text_input(
        "Results root folder",
        value="",
        help="Path to a results directory. The latest subfolder containing all_utterances.json will be used.",
    )
    if not root_folder:
        st.info("Enter a results folder path to begin.")
        st.stop()

    root_path = Path(root_folder)
    if not root_path.is_dir():
        st.error(f"Not a valid directory: {root_path}")
        st.stop()

    run_dirs = find_run_dirs(root_path)
    if not run_dirs:
        st.error(f"No folders with all_utterances.json found under {root_path}")
        st.stop()

    options = [str(d) for d in run_dirs]
    selected = st.selectbox(
        f"Found {len(run_dirs)} run(s) — select one",
        options,
        index=0,
    )
    results_folder = Path(selected)

    df = load_utterances(results_folder)

    st.header("Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total test cases", len(df))
    col2.metric("Failures (critical)", int(df["is_critical"].sum()))
    col3.metric("Failure rate", f"{df['is_critical'].mean():.1%}")

    calc_path = results_folder / "calculation_properties.csv"
    if calc_path.exists():
        with st.expander("Calculation Properties"):
            calc_df = pd.read_csv(calc_path)
            st.dataframe(calc_df, use_container_width=True, hide_index=True)

    discrete_features = get_discrete_features(df)
    if len(discrete_features) < 2:
        st.warning("Not enough discrete features found for heatmap.")
        st.stop()

    st.header("Failure Distribution Heatmap")

    col_left, col_right = st.columns(2)
    with col_left:
        feat_x = st.selectbox("Feature (X-axis)", discrete_features, index=0)
    with col_right:
        default_y = 1 if len(discrete_features) > 1 else 0
        feat_y = st.selectbox("Feature (Y-axis)", discrete_features, index=default_y)

    if feat_x == feat_y:
        st.warning("Select two different features.")
        st.stop()

    tab1, tab2 = st.tabs(["Failure Rate", "Test Count"])

    with tab1:
        fig_rate = build_heatmap(df, feat_x, feat_y)
        st.plotly_chart(fig_rate, use_container_width=True)

    with tab2:
        fig_count = build_count_heatmap(df, feat_x, feat_y)
        st.plotly_chart(fig_count, use_container_width=True)

    st.header("Utterances")

    fitness_cols = [c for c in df.columns if c.startswith("fitness_")]
    display_cols = ["question", "answer", "is_critical"] + fitness_cols

    view_mode = st.radio("View", ["All failures", "Filter by heatmap cell", "All test cases"], horizontal=True)

    if view_mode == "All failures":
        display_df = df[df["is_critical"]].reset_index(drop=True)
        st.markdown(f"**{len(display_df)}** failure(s)")
        if len(display_df) > 0:
            st.dataframe(display_df[display_cols], use_container_width=True, hide_index=True)

    elif view_mode == "Filter by heatmap cell":
        df_for_filter = df.dropna(subset=[feat_x, feat_y]).copy()
        df_for_filter[feat_x] = df_for_filter[feat_x].astype(str)
        df_for_filter[feat_y] = df_for_filter[feat_y].astype(str)

        x_vals = sorted(df_for_filter[feat_x].unique())
        y_vals = sorted(df_for_filter[feat_y].unique())

        col_fx, col_fy = st.columns(2)
        with col_fx:
            chosen_x = st.selectbox(f"Filter by {feat_x}", x_vals, key="filter_x")
        with col_fy:
            chosen_y = st.selectbox(f"Filter by {feat_y}", y_vals, key="filter_y")

        cell_df = df_for_filter[(df_for_filter[feat_x] == chosen_x) & (df_for_filter[feat_y] == chosen_y)]
        n_total = len(cell_df)
        n_fail = int(cell_df["is_critical"].sum())
        if n_total > 0:
            st.markdown(f"**{n_total}** test case(s), **{n_fail}** failure(s) ({n_fail/n_total:.0%})")
            st.dataframe(cell_df[display_cols].reset_index(drop=True), use_container_width=True, hide_index=True)
        else:
            st.info("No test cases for this combination.")

    else:
        st.markdown(f"**{len(df)}** test case(s)")
        st.dataframe(df[display_cols].reset_index(drop=True), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
