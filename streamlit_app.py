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
# Tab 1 – Overview
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
# Tab 2 – Player Cluster Map (with local k slider and per-player drill-down)
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

    # Compute k-means on 2D embedding for visualization
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
        summary_df.rename(
            columns={"viz_cluster_name": "Cluster", "AvgElo": "Avg Elo"}, inplace=True
        )
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
                    f"**Selected:** `{player_query}`  |  Elo: {selected_player['avg_elo']:.0f}"
                    f"  |  Tier: {selected_player['skill_tier']}"
                    f"  |  {selected_player['viz_cluster_name']}"
                )
                st.markdown(
                    f"Games analyzed: **{int(selected_player.get('num_games', 0))}**"
                )
            else:
                st.info("No player found with that handle in the processed dataset.")
        else:
            st.caption("Tip: paste a handle from the scatterplot tooltip to inspect a player.")

    with col_plot:
        df_filtered = player_viz[
            (player_viz["skill_tier"].isin(tiers_selected))
            & (player_viz["avg_elo"] >= rating_min)
            & (player_viz["avg_elo"] <= rating_max)
        ].copy()

        if color_by == "Skill Tier":
            fig = px.scatter(
                df_filtered,
                x="x",
                y="y",
                color="skill_tier",
                color_discrete_map=TIER_COLORS,
                hover_data=["player", "avg_elo", "num_games", "viz_cluster_name"],
                title=f"Player Embedding Map — {len(df_filtered):,} players",
                labels={
                    "x": "Embedding Dim 1",
                    "y": "Embedding Dim 2",
                    "skill_tier": "Skill Tier",
                },
                category_orders={"skill_tier": SKILL_TIERS},
            )
        else:
            fig = px.scatter(
                df_filtered,
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

        # Highlight selected player if found
        if selected_player is not None:
            fig.add_trace(
                go.Scatter(
                    x=[selected_player["x"]],
                    y=[selected_player["y"]],
                    mode="markers",
                    marker=dict(
                        size=14,
                        color="black",
                        symbol="x",
                        line=dict(width=2, color="white"),
                    ),
                    name="Selected Player",
                    hovertext=[f"Selected: {selected_player['player']}"],
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
# Tab 3 – Time Analysis
# ---------------------------------------------------------------------------
def render_time_tab():
    st.subheader("Time Analysis")

    # Heatmap: avg time per move by tier and phase
    fig_heatmap = go.Figure(
        data=go.Heatmap(
            z=time_heatmap_df.values,
            x=time_heatmap_df.columns.tolist(),
            y=time_heatmap_df.index.tolist(),
            colorscale="YlOrRd",
            text=time_heatmap_df.values.round(2),
            texttemplate="%{text}s",
            hovertemplate="Tier: %{y}<br>Phase: %{x}<br>Avg Time: %{z:.2f}s<extra></extra>",
        )
    )
    fig_heatmap.update_layout(
        title="Average Time per Move (seconds) by Skill Tier & Game Phase",
        xaxis_title="Game Phase",
        yaxis_title="Skill Tier",
        height=380,
    )

    # Time variance by tier and phase (player-level aggregates)
    tv_data = []
    for phase, col in [
        ("Opening", "time_variance_opening_mean"),
        ("Middlegame", "time_variance_middlegame_mean"),
        ("Endgame", "time_variance_endgame_mean"),
    ]:
        if col in player_df.columns:
            for tier in SKILL_TIERS:
                vals = player_df[player_df["skill_tier"] == tier][col]
                tv_data.append({"Phase": phase, "Skill Tier": tier, "Time Variance": vals.mean()})
    tv_df = pd.DataFrame(tv_data)
    fig_variance = px.bar(
        tv_df,
        x="Phase",
        y="Time Variance",
        color="Skill Tier",
        barmode="group",
        color_discrete_map=TIER_COLORS,
        title="Average Time Variance by Skill Tier & Game Phase",
        category_orders={"Skill Tier": SKILL_TIERS},
    )
    fig_variance.update_layout(height=380)

    # Time trouble frequency by tier
    if "time_trouble_frequency_mean" in player_df.columns:
        tt_data = []
        for tier in SKILL_TIERS:
            vals = player_df[player_df["skill_tier"] == tier]["time_trouble_frequency_mean"]
            tt_data.append({"Skill Tier": tier, "Time Trouble Freq": vals.mean()})
        tt_df = pd.DataFrame(tt_data)
        fig_tt = px.bar(
            tt_df,
            x="Skill Tier",
            y="Time Trouble Freq",
            color="Skill Tier",
            color_discrete_map=TIER_COLORS,
            title="Average Time Trouble Frequency by Skill Tier",
        )
        fig_tt.update_layout(showlegend=False, height=370)
    else:
        fig_tt = go.Figure()

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_heatmap, use_container_width=True)
    with col2:
        st.plotly_chart(fig_variance, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(fig_tt, use_container_width=True)
    with col4:
        st.markdown("### Key Time Insights")
        st.markdown(
            """
- Beginners spend more time per move on average than higher-rated players.
- Time variance in the opening is highest for beginners, suggesting inconsistent opening preparation.
- Advanced and Expert players show more consistent time management across all phases.
- Time trouble frequency decreases with increasing skill level.
- Stronger players allocate time more deliberately in the middlegame, which is the most predictive phase for skill tier.
"""
        )


# ---------------------------------------------------------------------------
# Tab 4 – Classification Results
# ---------------------------------------------------------------------------
def render_classification_tab():
    st.subheader("Skill-Tier Classification")

    # Normalized confusion matrix heatmap
    cm_norm = cm_array.astype(float) / cm_array.sum(axis=1, keepdims=True)
    fig_cm = go.Figure(
        data=go.Heatmap(
            z=cm_norm,
            x=cm_labels,
            y=cm_labels,
            colorscale="Blues",
            text=cm_array,
            texttemplate="%{text}",
            hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{text}<br>Rate: %{z:.2%}<extra></extra>",
        )
    )
    fig_cm.update_layout(
        title="Confusion Matrix (normalized by row)",
        xaxis_title="Predicted Tier",
        yaxis_title="Actual Tier",
        height=450,
    )

    # Feature importance (top 15)
    fi = feature_importance.head(15).copy()
    fi["feature_clean"] = (
        fi["feature"].str.replace("white_", "").str.replace("_", " ").str.title()
    )
    fig_fi = px.bar(
        fi,
        x="importance",
        y="feature_clean",
        orientation="h",
        color="importance",
        color_continuous_scale="Viridis",
        title="Top 15 Feature Importances (Random Forest)",
        labels={"importance": "Importance", "feature_clean": "Feature"},
    )
    fig_fi.update_layout(
        yaxis=dict(autorange="reversed"), height=450, coloraxis_showscale=False
    )

    # Metrics summary card
    m = classifier_metrics["metrics"]
    metrics_data = {
        "Metric": [
            "Test Accuracy",
            "Adjacent Accuracy (±1 tier)",
            "Macro Precision",
            "Macro Recall",
            "Macro F1",
            "Train / Val / Test",
        ],
        "Value": [
            f"{m['test_accuracy']:.1%}",
            f"{m['adjacent_accuracy']:.1%}",
            f"{m['macro_precision']:.3f}",
            f"{m['macro_recall']:.3f}",
            f"{m['macro_f1']:.3f}",
            f"{classifier_metrics['train_size']:,} / {classifier_metrics['val_size']:,} / {classifier_metrics['test_size']:,}",
        ],
    }
    metrics_df = pd.DataFrame(metrics_data)

    col1, col2 = st.columns([7, 5])
    with col1:
        st.plotly_chart(fig_cm, use_container_width=True)
    with col2:
        st.markdown("#### Classification Metrics")
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        st.caption(
            "Model: Random Forest (18 behavioral features). "
            "Adjacent accuracy counts predictions within ±1 tier as correct."
        )

    st.plotly_chart(fig_fi, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 5 – Cluster Analysis
# ---------------------------------------------------------------------------
def render_cluster_analysis_tab():
    st.subheader("Cluster Analysis")

    # Cluster sizes and avg elo
    clu_sizes = []
    for cid, info in clustering_results["cluster_names"].items():
        clu_sizes.append(
            {"Archetype": info["name"], "Size": info["size"], "Avg Elo": round(info["avg_elo"])}
        )
    clu_df = pd.DataFrame(clu_sizes)

    # Pie chart: cluster size distribution
    fig_pie = px.pie(
        clu_df,
        values="Size",
        names="Archetype",
        title="Cluster Size Distribution",
        color_discrete_sequence=CLUSTER_COLORS,
        hole=0.35,
    )
    fig_pie.update_layout(height=400)

    # Stacked bar: skill tier composition per cluster
    comp_data = []
    for _, row in cluster_stats.iterrows():
        cid = int(row["cluster"])
        name = cluster_name_map.get(cid, f"Cluster {cid}")
        for tier in SKILL_TIERS:
            col = f"pct_{tier.lower()}"
            if col in row:
                comp_data.append(
                    {"Cluster": name, "Skill Tier": tier, "Percentage": row[col]}
                )
    comp_df = pd.DataFrame(comp_data)
    fig_comp = px.bar(
        comp_df,
        x="Cluster",
        y="Percentage",
        color="Skill Tier",
        color_discrete_map=TIER_COLORS,
        barmode="stack",
        title="Skill Tier Composition per Cluster",
        category_orders={"Skill Tier": SKILL_TIERS},
    )
    fig_comp.update_layout(height=400)

    # Bar chart: avg Elo by cluster
    fig_elo = px.bar(
        clu_df,
        x="Archetype",
        y="Avg Elo",
        color="Archetype",
        color_discrete_sequence=CLUSTER_COLORS,
        title="Average Elo by Cluster",
    )
    fig_elo.update_layout(showlegend=False, height=370)

    # Clustering method comparison table
    mc = method_comparison.copy()
    for col in ["silhouette_score", "calinski_harabasz_index", "davies_bouldin_index"]:
        if col in mc.columns:
            mc[col] = mc[col].round(3)
    display_cols = [c for c in ["method", "n_clusters", "silhouette_score", "calinski_harabasz_index", "davies_bouldin_index"] if c in mc.columns]
    mc_display = mc[display_cols].rename(
        columns={
            "method": "Method",
            "n_clusters": "k",
            "silhouette_score": "Silhouette ↑",
            "calinski_harabasz_index": "CH ↑",
            "davies_bouldin_index": "DB ↓",
        }
    )

    col1, col2 = st.columns([5, 7])
    with col1:
        st.plotly_chart(fig_pie, use_container_width=True)
    with col2:
        st.plotly_chart(fig_comp, use_container_width=True)

    col3, col4 = st.columns([7, 5])
    with col3:
        st.plotly_chart(fig_elo, use_container_width=True)
    with col4:
        st.markdown("#### Clustering Method Comparison")
        st.dataframe(mc_display, use_container_width=True, hide_index=True)
        st.caption(
            "Silhouette and CH: higher is better. "
            "Davies-Bouldin: lower is better. "
            "K-Means (k=5) is used as the primary pipeline method."
        )


# ---------------------------------------------------------------------------
# Sidebar: global controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://lichess1.org/assets/logo/lichess-favicon-32.png", width=32)
    st.title("ChessInsight")
    st.caption("Team 029 · CSE 6242 · Spring 2026")
    st.divider()

    k_global = st.slider(
        "Global cluster count (k)",
        min_value=3,
        max_value=8,
        value=DEFAULT_K,
        step=1,
        key="global_k",
        help="Sets the default k for the cluster map tab. You can override it in the tab itself.",
    )

    st.divider()
    st.markdown(
        "**Data:** [Lichess dataset](https://database.lichess.org)  \n"
        "**Source:** [GitHub](https://github.com/Prago2001/chess-insights)"
    )


# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "Player Cluster Map", "Time Analysis", "Classification", "Cluster Analysis"]
)

with tab1:
    render_overview_tab(k_global)

with tab2:
    render_cluster_tab(k_global)

with tab3:
    render_time_tab()

with tab4:
    render_classification_tab()

with tab5:
    render_cluster_analysis_tab()
