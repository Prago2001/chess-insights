"""
ChessInsight Streamlit Dashboard
Team 029 - CSE6242 Spring 2026

Run locally with:
    streamlit run streamlit_app.py

This app visualizes the outputs of the analysis pipeline in `run_analysis.py` using
precomputed artifacts in `data/processed/` and `models/`.
"""

import json
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

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
SKILL_TIERS = ["Beginner", "Intermediate", "Advanced"]
TIER_COLORS = {
    "Beginner": "#e74c3c",
    "Intermediate": "#f39c12",
    "Advanced": "#2ecc71",
}

# Player archetypes from clustering (K=3, PCA=2)
ARCHETYPE_DESCRIPTIONS = {
    "Time Scramblers": "Fast play style, comfortable in time pressure, highest average Elo",
    "Tactical Battlers": "Seek complex positions with material imbalances, lowest average Elo",
    "Positional Grinders": "Keep pieces active in simpler positions, medium average Elo",
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

# Note: Accuracy features (blunder rate, CPL, etc.) require Stockfish engine evaluation
# which is not available in our PGN dataset. These are excluded from analysis.


# ---------------------------------------------------------------------------
# Helper to render a numeric KPI-style block
# ---------------------------------------------------------------------------
def render_kpi(label: str, value: str, help_text: Optional[str] = None):
    st.metric(label=label, value=value, help=help_text)


# ---------------------------------------------------------------------------
# Tab 1 – Overview
# ---------------------------------------------------------------------------
def render_overview_tab():
    st.subheader("Overview")

    # Project introduction
    st.markdown(
        "**ChessInsight** answers two questions: (1) Can we predict a player's skill tier from their behavior? "
        "(2) What distinct playing styles exist among chess players? Using 350K Lichess games, we built a "
        "classifier achieving **65.8% accuracy** and identified **3 behavioral archetypes**."
    )
    st.markdown("")

    ds = analysis["dataset"]
    cl = analysis["classification"]
    clu = analysis["clustering"]

    # KPI row with 5 metrics
    kpi_cols = st.columns(5)
    with kpi_cols[0]:
        render_kpi("Total Games", f"{ds['total_games']:,}",
                   "Number of chess games analyzed from the Lichess database")
    with kpi_cols[1]:
        render_kpi("Unique Players", f"{len(player_df):,}",
                   "Number of distinct players in our dataset, rated between 606 and 3253 Elo")
    with kpi_cols[2]:
        render_kpi("Test Accuracy", cl["test_accuracy"],
                   "How often the model predicts the exact skill tier correctly")
    with kpi_cols[3]:
        render_kpi("Adjacent Accuracy", cl["adjacent_accuracy"],
                   "How often the prediction is correct or within one tier (e.g., predicting Intermediate for a Beginner)")
    with kpi_cols[4]:
        render_kpi(
            "Archetypes",
            str(clu['n_clusters']),
            "Number of distinct playing style clusters discovered through behavioral analysis",
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
        x="elo",
        nbins=60,
        title="Player Rating Distribution",
        labels={"elo": "Average Elo"},
        color_discrete_sequence=["#3498db"],
    )
    fig_rating.update_layout(height=350)

    col_top = st.columns(2)
    with col_top[0]:
        st.plotly_chart(fig_dist, use_container_width=True)
    with col_top[1]:
        st.plotly_chart(fig_rating, use_container_width=True)

    # Archetype summary with percentages
    st.markdown("### Player Archetypes (from Clustering)")

    # Calculate archetype percentages
    archetype_counts = player_df["cluster_name"].value_counts()
    total_players = len(player_df)

    arch_cols = st.columns(3)
    for i, (name, desc) in enumerate(ARCHETYPE_DESCRIPTIONS.items()):
        count = archetype_counts.get(name, 0)
        pct = count / total_players * 100 if total_players > 0 else 0
        with arch_cols[i]:
            st.info(f"**{name}** ({pct:.0f}%)\n\n{desc}\n\n*{count:,} players*")

    # Key finding callout
    st.markdown("### Key Finding: Multiple Paths to Mastery")
    st.success(
        "**Can we predict skill from behavior? Yes — but not from playing style.**\n\n"
        "Our classifier achieves **65.8% accuracy** predicting skill tier from behavioral metrics like "
        "time management, decision consistency, and complexity handling. However, playing *style* "
        "(Time Scrambler vs. Positional Grinder vs. Tactical Battler) does **not** determine skill — "
        "all three archetypes reach Advanced ratings.\n\n"
        "**What this means:** *Which* style you play doesn't matter. *How efficiently* you execute it does. "
        "Players should focus on improving their time management and decision consistency, not changing their natural approach."
    )


# ---------------------------------------------------------------------------
# Tab 2 – Player Cluster Map (using pre-computed K=3 archetypes)
# ---------------------------------------------------------------------------
def render_cluster_tab():
    st.subheader("Player Cluster Map")

    # Use pre-computed clusters from the pipeline (K=3 archetypes)
    player_viz = player_df.dropna(subset=["x", "y"]).copy()

    col_controls, col_plot = st.columns([1, 3])

    with col_controls:
        color_by = st.radio(
            "Color By",
            options=["Archetype", "Skill Tier"],
            index=0,
        )

        tiers_selected = st.multiselect(
            "Filter Skill Tiers",
            options=SKILL_TIERS,
            default=SKILL_TIERS,
            key="cluster_tiers_multiselect",
        )

        min_elo = int(player_viz["elo"].min())
        max_elo = int(player_viz["elo"].max())
        rating_min, rating_max = st.slider(
            "Rating Range",
            min_value=min_elo,
            max_value=max_elo,
            value=(min_elo, max_elo),
            step=50,
        )

        st.markdown("**Archetype Summary**")
        summary_df = (
            player_viz.groupby("cluster_name")
            .agg(Size=("player", "count"), AvgElo=("elo", "mean"))
            .reset_index()
        )
        total = summary_df["Size"].sum()
        summary_df["Pct"] = (summary_df["Size"] / total * 100).round(0).astype(int).astype(str) + "%"
        summary_df["AvgElo"] = summary_df["AvgElo"].round(0).astype(int)
        summary_df.rename(
            columns={"cluster_name": "Archetype", "AvgElo": "Avg Elo"}, inplace=True
        )
        summary_df = summary_df[["Archetype", "Size", "Pct", "Avg Elo"]]
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        st.markdown("**Player Drill-Down**")

        # Get sample players from each archetype for users to try
        sample_players = []
        for archetype in ARCHETYPE_DESCRIPTIONS.keys():
            archetype_players = player_viz[player_viz["cluster_name"] == archetype]
            if not archetype_players.empty:
                sample = archetype_players.sample(1).iloc[0]
                sample_players.append(sample["player"])

        player_query = st.text_input(
            "Enter player handle (exact match)",
            "",
            help="Type a specific Lichess player handle present in the dataset.",
        )

        if sample_players:
            st.caption(f"Try these: `{sample_players[0]}`, `{sample_players[1] if len(sample_players) > 1 else ''}`, `{sample_players[2] if len(sample_players) > 2 else ''}`")

        selected_player = None
        if player_query:
            matches = player_viz[player_viz["player"] == player_query]
            if not matches.empty:
                selected_player = matches.iloc[0]
                st.markdown(
                    f"**Selected:** `{player_query}`  |  Elo: {selected_player['elo']:.0f}"
                    f"  |  Tier: {selected_player['skill_tier']}"
                    f"  |  {selected_player['cluster_name']}"
                )
                st.markdown(
                    f"Games analyzed: **{int(selected_player.get('game_count', 0))}**"
                )
            else:
                st.info("No player found with that handle in the processed dataset.")

    with col_plot:
        df_filtered = player_viz[
            (player_viz["skill_tier"].isin(tiers_selected))
            & (player_viz["elo"] >= rating_min)
            & (player_viz["elo"] <= rating_max)
        ].copy()

        # Common labels for human-readable hover tooltips
        friendly_labels = {
            "x": "PCA Component 1",
            "y": "PCA Component 2",
            "skill_tier": "Skill Tier",
            "cluster_name": "Archetype",
            "player": "Player",
            "elo": "Rating",
            "game_count": "Games Played",
        }

        if color_by == "Skill Tier":
            fig = px.scatter(
                df_filtered,
                x="x",
                y="y",
                color="skill_tier",
                color_discrete_map=TIER_COLORS,
                hover_data=["player", "elo", "game_count", "cluster_name"],
                title=f"Player Embedding Map — {len(df_filtered):,} players",
                labels=friendly_labels,
                category_orders={"skill_tier": SKILL_TIERS},
            )
        else:
            fig = px.scatter(
                df_filtered,
                x="x",
                y="y",
                color="cluster_name",
                color_discrete_sequence=CLUSTER_COLORS,
                hover_data=["player", "elo", "game_count", "skill_tier"],
                title=f"Player Behavioral Map — 3 Archetypes",
                labels=friendly_labels,
            )

        # Highlight selected player if found
        if selected_player is not None:
            # Add pulsing ring effect (outer glow)
            fig.add_trace(
                go.Scatter(
                    x=[selected_player["x"]],
                    y=[selected_player["y"]],
                    mode="markers",
                    marker=dict(
                        size=35,
                        color="rgba(255, 0, 0, 0.3)",  # Red glow
                        symbol="circle",
                        line=dict(width=3, color="red"),
                    ),
                    name="",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            # Main star marker
            fig.add_trace(
                go.Scatter(
                    x=[selected_player["x"]],
                    y=[selected_player["y"]],
                    mode="markers+text",
                    marker=dict(
                        size=22,
                        color="#FF4444",  # Bright red
                        symbol="star",
                        line=dict(width=2, color="white"),
                    ),
                    text=[selected_player["player"]],
                    textposition="top center",
                    textfont=dict(size=14, color="white", family="Arial Black"),
                    name="Selected Player",
                    hovertext=[f"<b>{selected_player['player']}</b><br>Elo: {selected_player['elo']:.0f}<br>{selected_player['cluster_name']}"],
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

        # Explanatory note about skill tier distribution
        if color_by == "Skill Tier":
            st.caption(
                "💡 **Why do skill tiers appear mixed?** The scatter plot shows *playing style*, not skill. "
                "Beginners, Intermediates, and Advanced players all share similar styles — proving that "
                "*which* style you play doesn't determine your rating. However, our classifier still achieves "
                "65.8% accuracy using *efficiency metrics* (time management, consistency). "
                "Switch to 'Color by Archetype' to see the three behavioral clusters."
            )
        else:
            st.caption(
                "💡 Three behavioral archetypes based on playing patterns. Each contains players from all skill levels — "
                "style doesn't determine rating, but execution efficiency does."
            )


# ---------------------------------------------------------------------------
# Tab 3 – Time Analysis
# ---------------------------------------------------------------------------
def render_time_tab():
    st.subheader("Time Analysis")

    st.markdown(
        "How players manage their clock reveals skill differences. This tab explores "
        "**time per move**, **time variance** (consistency), and **time trouble frequency** "
        "(moves made under pressure) across skill tiers and game phases."
    )
    st.markdown("")

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
        ("Opening", "time_variance_opening"),
        ("Middlegame", "time_variance_middlegame"),
        ("Endgame", "time_variance_endgame"),
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
    if "time_trouble_frequency" in player_df.columns:
        tt_data = []
        for tier in SKILL_TIERS:
            vals = player_df[player_df["skill_tier"] == tier]["time_trouble_frequency"]
            tt_data.append({"Skill Tier": tier, "Time Trouble (%)": vals.mean() * 100})
        tt_df = pd.DataFrame(tt_data)
        fig_tt = px.bar(
            tt_df,
            x="Skill Tier",
            y="Time Trouble (%)",
            color="Skill Tier",
            color_discrete_map=TIER_COLORS,
            title="Time Trouble Frequency by Skill Tier",
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
        st.caption(
            "💡 **Time Trouble** = percentage of moves made with less than 10% of starting time remaining. "
            "Higher-rated players enter time trouble more often, likely due to deeper calculation."
        )
    with col4:
        st.markdown("### Key Time Insights")
        st.markdown(
            """
- Beginners spend more time per move (2.3s opening) vs Advanced (1.4s opening).
- Advanced players have highest opening time variance, suggesting diverse opening repertoires.
- Beginners have most consistent opening times but higher middlegame/endgame variance.
- Time trouble frequency increases with skill: Beginner 3.8% → Advanced 6.3%.
- Top predictive features: game length, middlegame time usage, position complexity.
- Time Scramblers make 28% of moves in low time vs Tactical Battlers at 4%.
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
        st.caption(
            "💡 **How to read this:** Each row shows how players of a true skill tier were classified. "
            "Diagonal cells (top-left to bottom-right) are correct predictions. "
            "Off-diagonal cells show misclassifications — most errors are to adjacent tiers."
        )
    with col2:
        st.markdown("#### Classification Metrics")
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        st.caption(
            "**Test Accuracy:** Exact match rate. "
            "**Adjacent Accuracy:** Correct or within ±1 tier. "
            "**Precision:** Of predicted tier X, how many were actually X. "
            "**Recall:** Of actual tier X, how many were predicted X. "
            "**F1:** Harmonic mean of precision and recall."
        )
        st.markdown("---")
        st.caption(
            "Model: Random Forest · 3 skill tiers (Beginner <1400, Intermediate 1400-1899, Advanced 1900+) · "
            "Features: time usage, complexity, opening patterns."
        )

    st.plotly_chart(fig_fi, use_container_width=True)
    st.caption(
        "💡 **Feature Importance:** Shows which player behaviors most influence skill tier predictions. "
        "Higher importance = stronger signal for distinguishing skill levels."
    )


# ---------------------------------------------------------------------------
# Tab 5 – Cluster Analysis (with Archetype Comparison Panel)
# ---------------------------------------------------------------------------
def render_cluster_analysis_tab():
    st.subheader("Cluster Analysis")

    # Load cluster centers for archetype comparison
    try:
        cluster_centers = load_csv("cluster_centers_final.csv")
    except Exception:
        cluster_centers = None

    # Archetype comparison panel (as promised in progress report)
    st.markdown("### Archetype Comparison Panel")
    if cluster_centers is not None and not cluster_centers.empty:
        arch_cols = st.columns(3)
        for i, row in cluster_centers.iterrows():
            name = row["archetype"]
            with arch_cols[i % 3]:
                st.markdown(f"**{name}**")
                st.caption(ARCHETYPE_DESCRIPTIONS.get(name, ""))
                st.markdown(f"- Moves in Time Pressure: **{row['low_time_move_ratio']:.1%}**")
                st.markdown(f"- Position Complexity: **{row['avg_position_complexity']:.1f}**")
                st.markdown(f"- Material Imbalance: **{row['material_imbalance_freq']:.1%}**")
                st.markdown(f"- Piece Activity: **{row['piece_activity_score']:.1f}**")

        # Radar chart for archetype comparison
        categories = ["Time Pressure", "Complexity", "Material Imbalance", "Piece Activity", "Aggression"]
        fig_radar = go.Figure()
        for _, row in cluster_centers.iterrows():
            # Normalize values for radar chart (scale to 0-100)
            values = [
                row["low_time_move_ratio"] * 100 / 0.3,  # Scale low_time to ~100
                row["avg_position_complexity"] / 40 * 100,  # Scale complexity
                row["material_imbalance_freq"] * 100,  # Already 0-1
                row["piece_activity_score"] / 35 * 100,  # Scale activity
                row["opening_aggression_score"] / 70 * 100,  # Scale aggression
            ]
            values.append(values[0])  # Close the radar
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill='toself',
                name=row["archetype"],
                opacity=0.6,
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            title="Archetype Behavioral Profiles (Normalized)",
            height=400,
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    else:
        for i, (name, desc) in enumerate(ARCHETYPE_DESCRIPTIONS.items()):
            with st.columns(3)[i]:
                st.info(f"**{name}**\n\n{desc}")

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
            "n_clusters": "Clusters",
            "silhouette_score": "Silhouette",
            "calinski_harabasz_index": "CH Index",
            "davies_bouldin_index": "DB Index",
        }
    )

    col1, col2 = st.columns([5, 7])
    with col1:
        st.plotly_chart(fig_pie, use_container_width=True)
        st.caption("💡 Distribution of players across the 3 behavioral archetypes.")
    with col2:
        st.plotly_chart(fig_comp, use_container_width=True)
        st.caption(
            "💡 **Skill composition within each archetype.** All three archetypes contain similar proportions "
            "of Beginners, Intermediates, and Advanced players. This confirms that *which* style you play "
            "doesn't determine your rating — but *how efficiently* you execute it does (65.8% accuracy)."
        )

    col3, col4 = st.columns([7, 5])
    with col3:
        st.plotly_chart(fig_elo, use_container_width=True)
    with col4:
        st.markdown("#### Clustering Method Comparison")
        st.dataframe(mc_display, use_container_width=True, hide_index=True)
        st.caption(
            "**Silhouette** (↑ better): Measures how similar points are to their own cluster vs others. "
            "**CH Index** (↑ better): Ratio of between-cluster to within-cluster variance. "
            "**DB Index** (↓ better): Average similarity between clusters. "
            "K-Means with K=3 was selected."
        )


# ---------------------------------------------------------------------------
# Sidebar: global controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://lichess1.org/assets/logo/lichess-favicon-32.png", width=32)
    st.title("ChessInsight")
    st.caption("Team 029 · CSE 6242 · Spring 2026")
    st.divider()

    st.markdown("**Clustering:** K-Means (K=3)")
    st.markdown("**Classification:** 3-tier (Random Forest)")
    st.markdown(f"**Players:** {len(player_df):,}")

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
    render_overview_tab()

with tab2:
    render_cluster_tab()

with tab3:
    render_time_tab()

with tab4:
    render_classification_tab()

with tab5:
    render_cluster_analysis_tab()
