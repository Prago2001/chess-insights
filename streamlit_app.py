"""
ChessInsight Streamlit Dashboard
Team 029 - CSE6242 Spring 2026

Run locally with:
    streamlit run streamlit_app.py

This app visualizes the outputs of the analysis pipeline in `run_analysis.py` using
precomputed artifacts in `data/processed/` and `models/`.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans

from config import PROCESSED_DATA_DIR, MODELS_DIR

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ChessInsight Streamlit Dashboard",
    page_icon="♟",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SKILL_TIERS = ["Beginner", "Intermediate", "Advanced", "Expert"]
TIER_COLORS = {
    "Beginner": "#e74c3c",
    "Intermediate": "#f39c12",
    "Advanced": "#3498db",
    "Expert": "#2ecc71",
}
CLUSTER_COLORS = px.colors.qualitative.Set2


# ---------------------------------------------------------------------------
# Data loading helpers (cached for fast local dev)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_player_data() -> pd.DataFrame:
    """Load player-level features and clustering embeddings."""
    pf = pd.read_parquet(PROCESSED_DATA_DIR / "player_features.parquet")
    emb = pd.read_parquet(PROCESSED_DATA_DIR / "player_clustering_embeddings.parquet")
    pf = pd.concat([pf.reset_index(drop=True), emb.reset_index(drop=True)], axis=1)
    return pf


@st.cache_data(show_spinner=False)
def load_game_features() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DATA_DIR / "game_features.parquet")


@st.cache_data(show_spinner=False)
def load_games_processed() -> pd.DataFrame:
    """Load game-level data used for opening pattern analysis."""
    return pd.read_parquet(PROCESSED_DATA_DIR / "games_processed.parquet")


@st.cache_data(show_spinner=False)
def load_json(name: str) -> dict:
    with open(MODELS_DIR / name) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(MODELS_DIR / name)


# ---------------------------------------------------------------------------
# Load all core artifacts once per session with basic error handling
# ---------------------------------------------------------------------------
try:
    player_df = load_player_data()
    game_df = load_game_features()
    analysis = load_json("analysis_summary.json")
    clustering_results = load_json("player_clustering_results.json")
    classifier_metrics = load_json("skill_classifier_metrics.json")
    feature_importance = load_csv("skill_classifier_feature_importance.csv")
    confusion_csv = load_csv("skill_classifier_confusion_matrix.csv")
    cluster_stats = load_csv("player_clustering_statistics.csv")
    method_comparison = load_csv("clustering_method_comparison.csv")
except Exception as e:  # pragma: no cover - runtime safeguard
    st.error(
        "Failed to load processed data and model artifacts. "
        "Please run `python run_analysis.py` first so that `data/processed/` "
        "and `models/` are populated.\n\nDetails: " + str(e)
    )
    st.stop()

# Build cluster name map and attach to player_df (pipeline baseline)
cluster_name_map = {int(k): v["name"] for k, v in clustering_results["cluster_names"].items()}
player_df["cluster_name"] = player_df["cluster"].map(cluster_name_map)

# Default k from pipeline results
DEFAULT_K = clustering_results.get("n_clusters", len(cluster_name_map))

# Confusion matrix as numpy array
cm_labels = list(confusion_csv.columns[1:])
cm_array = confusion_csv.iloc[:, 1:].values.astype(int)

# Precompute aggregates for time heatmap (game-level)
time_phase_cols = {
    "Opening": "white_avg_time_opening",
    "Middlegame": "white_avg_time_middlegame",
    "Endgame": "white_avg_time_endgame",
}

time_heatmap_data = []
for tier in SKILL_TIERS:
    tier_games = game_df[game_df["white_skill_tier"] == tier]
    row = {"Skill Tier": tier}
    for phase, col in time_phase_cols.items():
        row[phase] = round(tier_games[col].mean(), 2) if col in tier_games.columns else 0.0
    time_heatmap_data.append(row)

time_heatmap_df = pd.DataFrame(time_heatmap_data).set_index("Skill Tier")

# Precompute accuracy-related aggregates (game-level)
accuracy_by_tier = []
for tier in SKILL_TIERS:
    t = game_df[game_df["white_skill_tier"] == tier]
    accuracy_by_tier.append(
        {
            "Skill Tier": tier,
            "Blunder Rate (%)": round(t["white_blunder_rate"].mean() * 100, 2)
            if "white_blunder_rate" in t.columns
            else 0.0,
            "Mistake Rate (%)": round(t["white_mistake_rate"].mean() * 100, 2)
            if "white_mistake_rate" in t.columns
            else 0.0,
            "Avg CPL": round(t["white_avg_centipawn_loss"].mean(), 1)
            if "white_avg_centipawn_loss" in t.columns
            else 0.0,
            "Accuracy (%)": round(t["white_accuracy_percentage"].mean(), 1)
            if "white_accuracy_percentage" in t.columns
            else 0.0,
        }
    )
accuracy_df = pd.DataFrame(accuracy_by_tier)


# ---------------------------------------------------------------------------
# Helper to render a numeric KPI-style block
# ---------------------------------------------------------------------------
def render_kpi(label: str, value: str, help_text: Optional[str] = None):
    st.metric(label=label, value=value, help=help_text)


# ---------------------------------------------------------------------------
# Overview Tab
# ---------------------------------------------------------------------------
def render_overview_tab(k_selected: int):
    st.subheader("Overview")

    ds = analysis["dataset"]
    cl = analysis["classification"]
    clu = analysis["clustering"]

    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        render_kpi("Total Games", f"{ds['total_games']:,}", ds["data_source"])
    with kpi_cols[1]:
        render_kpi("Unique Players", f"{ds['unique_players']:,}", f"Rating {ds['rating_range']}")
    with kpi_cols[2]:
        render_kpi("Test Accuracy", cl["test_accuracy"], "Exact tier classification accuracy")
    with kpi_cols[3]:
        render_kpi(
            "Clusters (k)",
            str(k_selected),
            f"Pipeline baseline used k = {clu['n_clusters']}",
        )

    # Skill tier distribution
    tier_counts = pd.DataFrame(
        [
            {"Tier": t, "Count": int(classifier_metrics["class_distribution"].get(t, 0))}
            for t in SKILL_TIERS
        ]
    )
    fig_dist = px.bar(
        tier_counts,
        x="Tier",
        y="Count",
        color="Tier",
        color_discrete_map=TIER_COLORS,
        title="Skill Tier Distribution (Classification Dataset)",
    )
    fig_dist.update_layout(showlegend=False, height=350)

    # Rating histogram from player data
    fig_rating = px.histogram(
        player_df,
        x="avg_elo",
        nbins=60,
        title="Player Rating Distribution",
        labels={"avg_elo": "Average Elo"},
        color_discrete_sequence=["#3498db"],
    )
    fig_rating.update_layout(height=350)

    # Accuracy metrics by tier
    acc_melted = accuracy_df.melt(id_vars="Skill Tier", var_name="Metric", value_name="Value")
    fig_acc = px.bar(
        acc_melted,
        x="Skill Tier",
        y="Value",
        color="Metric",
        barmode="group",
        title="Accuracy Metrics by Skill Tier",
    )
    fig_acc.update_layout(height=380)

    col_top = st.columns(2)
    with col_top[0]:
        st.plotly_chart(fig_dist, use_container_width=True)
    with col_top[1]:
        st.plotly_chart(fig_rating, use_container_width=True)

    st.plotly_chart(fig_acc, use_container_width=True)


# ---------------------------------------------------------------------------
# Player Cluster Map Tab (with per-player drill-down and local k slider)
# ---------------------------------------------------------------------------
def render_cluster_tab(k_selected: int):
    st.subheader("Player Cluster Map")

    # Local k slider for this tab (defaults to global k)
    k_tab = st.slider(
        "Clusters for this view (k)",
        min_value=3,
        max_value=8,
        value=k_selected,
        step=1,
        key="cluster_tab_k",
        help="Adjusts the number of clusters used in this Player Cluster Map view.",
    )

    # Compute k-means on 2D embedding for interactive visualization (shared across controls/plot)
    coords = player_df[["x", "y"]].values
    kmeans = KMeans(n_clusters=k_tab, random_state=42, n_init=10)
    viz_labels_all = kmeans.fit_predict(coords)

    player_viz = player_df.copy()
    player_viz["viz_cluster"] = viz_labels_all
    player_viz["viz_cluster_name"] = [
        f"Cluster {c + 1}" for c in player_viz["viz_cluster"]
    ]

    col_controls, col_plot = st.columns([1, 3])

    with col_controls:
        color_by = st.radio(
            "Color By",
            options=["Cluster (k)", "Skill Tier"],
            index=0,
        )

        tiers_selected = st.multiselect(
            "Filter Skill Tiers",
            options=SKILL_TIERS,
            default=SKILL_TIERS,
            key="cluster_tiers_multiselect",
        )

        min_elo = int(player_viz["avg_elo"].min())
        max_elo = int(player_viz["avg_elo"].max())
        rating_min, rating_max = st.slider(
            "Rating Range",
            min_value=min_elo,
            max_value=max_elo,
            value=(min_elo, max_elo),
            step=50,
        )

        st.markdown(f"**Cluster Summary (k = {k_tab})**")
        summary_df = (
            player_viz.groupby("viz_cluster_name")
            .agg(Size=("player", "count"), AvgElo=("avg_elo", "mean"))
            .reset_index()
        )
        summary_df["AvgElo"] = summary_df["AvgElo"].round(0).astype(int)
        summary_df.rename(columns={"viz_cluster_name": "Cluster", "AvgElo": "Avg Elo"}, inplace=True)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        st.markdown("**Player Drill-Down**")
        player_query = st.text_input(
            "Enter player handle (exact match)",
            "",
            help="Type a specific Lichess player handle present in the dataset.",
        )
        selected_player = None
        if player_query:
            matches = player_viz[player_viz["player"] == player_query]
            if not matches.empty:
                selected_player = matches.iloc[0]
                st.markdown(
                    f"**Selected player:** `{player_query}`  |  Elo: {selected_player['avg_elo']:.0f}  "
                    f"|  Tier: {selected_player['skill_tier']}  |  Cluster (k={k_tab}): {selected_player['viz_cluster_name']}"
                )
                st.markdown(
                    f"Games analyzed: **{int(selected_player.get('num_games', 0))}**"
                )
            else:
                st.info("No player found with that handle in the processed dataset.")
        else:
            st.caption("Tip: paste a handle from the scatterplot tooltip to inspect a player.")

    with col_plot:
        df = player_viz[
            (player_viz["skill_tier"].isin(tiers_selected))
            & (player_viz["avg_elo"] >= rating_min)
            & (player_viz["avg_elo"] <= rating_max)
        ].copy()

        if color_by == "Skill Tier":
            fig = px.scatter(
                df,
                x="x",
                y="y",
                color="skill_tier",
                color_discrete_map=TIER_COLORS,
                hover_data=["player", "avg_elo", "num_games", "viz_cluster_name"],
                title=f"Player Embedding Map — {len(df):,} players",
                labels={
                    "x": "Embedding Dim 1",
                    "y": "Embedding Dim 2",
                    "skill_tier": "Skill Tier",
                },
                category_orders={"skill_tier": SKILL_TIERS},
            )
        else:
            fig = px.scatter(
                df,
                x="x",
                y="y",
                color="viz_cluster_name",
                color_discrete_sequence=CLUSTER_COLORS,
                hover_data=["player", "avg_elo", "num_games", "skill_tier"],
                title=f"Player Embedding Map — k = {k_tab}",
                labels={
                    "x": "Embedding Dim 1",
                    "y": "Embedding Dim 2",
                    "viz_cluster_name": "Cluster",
                },
            )

        # Highlight selected player if present
        if "selected_player" in locals() and selected_player is not None:
            fig.add_trace(
                go.Scatter(
                    x=[selected_player["x"]],
                    y=[selected_player["y"]],
                    mode="markers",
                    marker=dict(
                        size=12,
                        color="black",
                        symbol="x",
                        line=dict(width=2, color="white"),
                    ),
                    name="Selected Player",
                    hovertext=[f"Selected: {selected_player['player']}`"],
                    hoverinfo="text",
                )
            )

        fig.update_traces(
            marker=dict(size=4, opacity=0.6, line=dict(width=0)),
            selector=dict(mode="markers"),
        )
        fig.update_layout(
            margin=dict(l=10, r=10, t=60, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Time Analysis Tab
# ---------------------------------------------------------------------------
... (rest of file unchanged) ...
