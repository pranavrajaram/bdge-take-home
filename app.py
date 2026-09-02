"""Optional Streamlit explorer; run with `streamlit run app.py -- --projections file.csv`."""
from __future__ import annotations

import argparse

import pandas as pd


def main() -> None:
    import streamlit as st
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--projections", default="outputs/week1_2026/projections.csv")
    args, _ = parser.parse_known_args()
    st.set_page_config(page_title="Fantasy ranges", layout="wide")
    st.title("Week 1 fantasy range projections")
    try:
        data = pd.read_csv(args.projections)
    except FileNotFoundError:
        st.info("Run `fantasy-ranges project-week1` or a demo first, then point `--projections` at its CSV.")
        return
    player = st.selectbox("Player", data["player_name"].tolist())
    row = data.loc[data["player_name"] == player].iloc[0]
    cols = st.columns(5)
    for col, key, label in zip(cols, ["p10", "p25", "p50", "p75", "p90"], ["P10", "P25", "Median", "P75", "P90"]):
        col.metric(label, f"{row[key]:.1f}")
    st.caption(f"Role uncertainty: {row['role_uncertainty']:.0%}. This width comes from workload uncertainty propagated through the component model.")
    st.dataframe(data[["player_name", "position", "team", "p10", "p25", "p50", "p75", "p90", "p_10_plus", "p_15_plus", "p_20_plus", "role_uncertainty"]], hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()
