"""
ChessInsight Interactive Dashboard
Team 029 - CSE6242 Spring 2026

Run with: python3 dashboard.py
Then open http://127.0.0.1:8050 in your browser.
"""

import json
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

SKILL_TIERS = ["Beginner", "Intermediate", "Advanced", "Expert"]
TIER_COLORS = {
    "Beginner": "#e74c3c",
    "Intermediate": "#f39c12",
    "Advanced": "#3498db",
    "Expert": "#2ecc71",
}
CLUSTER_COLORS = px.colors.qualitative.Set2

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_player_data() -> pd.DataFrame:
    """Load player features and merge with clustering embeddings."""
    pf = pd.read_parquet(DATA_DIR / "player_features.parquet")
    emb = pd.read_parquet(DATA_DIR / "player_clustering_embeddings.parquet")
    pf = pd.concat([pf.reset_index(drop=True), emb.reset_index(drop=True)], axis=1)
    return pf


def load_game_features() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "game_features.parquet")


def load_json(name: str) -> dict:
    with open(MODELS_DIR / name) as f:
        return json.load(f)


def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(MODELS_DIR / name)


# ---------------------------------------------------------------------------
# Load all data once at startup
# ---------------------------------------------------------------------------
player_df = load_player_data()
game_df = load_game_features()
analysis = load_json("analysis_summary.json")
clustering_results = load_json("player_clustering_results.json")
classifier_metrics = load_json("skill_classifier_metrics.json")
feature_importance = load_csv("skill_classifier_feature_importance.csv")
confusion_csv = load_csv("skill_classifier_confusion_matrix.csv")
cluster_stats = load_csv("player_clustering_statistics.csv")
method_comparison = load_csv("clustering_method_comparison.csv")

# Build cluster name map: cluster_id -> name
cluster_name_map = {
    int(k): v["name"] for k, v in clustering_results["cluster_names"].items()
}
player_df["cluster_name"] = player_df["cluster"].map(cluster_name_map)

# Confusion matrix as numpy
cm_labels = list(confusion_csv.columns[1:])
cm_array = confusion_csv.iloc[:, 1:].values.astype(int)

# ---------------------------------------------------------------------------
# Precompute aggregates for the time heatmap
# ---------------------------------------------------------------------------
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
        row[phase] = round(tier_games[col].mean(), 2) if col in tier_games.columns else 0
    time_heatmap_data.append(row)
time_heatmap_df = pd.DataFrame(time_heatmap_data).set_index("Skill Tier")

# Precompute accuracy aggregates
accuracy_by_tier = []
for tier in SKILL_TIERS:
    t = game_df[game_df["white_skill_tier"] == tier]
    accuracy_by_tier.append({
        "Skill Tier": tier,
        "Blunder Rate (%)": round(t["white_blunder_rate"].mean() * 100, 2) if "white_blunder_rate" in t.columns else 0,
        "Mistake Rate (%)": round(t["white_mistake_rate"].mean() * 100, 2) if "white_mistake_rate" in t.columns else 0,
        "Avg CPL": round(t["white_avg_centipawn_loss"].mean(), 1) if "white_avg_centipawn_loss" in t.columns else 0,
        "Accuracy (%)": round(t["white_accuracy_percentage"].mean(), 1) if "white_accuracy_percentage" in t.columns else 0,
    })
accuracy_df = pd.DataFrame(accuracy_by_tier)

# Free memory - we don't need the full game_df after precomputing
del game_df

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    title="ChessInsight Dashboard",
    suppress_callback_exceptions=True,
)

# ---------------------------------------------------------------------------
# Reusable components
# ---------------------------------------------------------------------------

def metric_card(title, value, subtitle="", color="primary"):
    return dbc.Card(
        dbc.CardBody([
            html.H6(title, className="card-subtitle mb-1 text-muted"),
            html.H3(str(value), className=f"card-title text-{color} mb-0"),
            html.Small(subtitle, className="text-muted") if subtitle else html.Span(),
        ]),
        className="shadow-sm h-100",
    )


# ---------------------------------------------------------------------------
# TAB 1 – Overview
# ---------------------------------------------------------------------------
def build_overview_tab():
    ds = analysis["dataset"]
    cl = analysis["classification"]
    clu = analysis["clustering"]

    cards = dbc.Row([
        dbc.Col(metric_card("Total Games", f"{ds['total_games']:,}", "Lichess dataset", "primary"), md=3),
        dbc.Col(metric_card("Unique Players", f"{ds['unique_players']:,}", f"Rating {ds['rating_range']}", "info"), md=3),
        dbc.Col(metric_card("Test Accuracy", cl["test_accuracy"], f"Adjacent: {cl['adjacent_accuracy']}", "success"), md=3),
        dbc.Col(metric_card("Clusters Found", clu["n_clusters"], f"Silhouette: {clu['silhouette_score']}", "warning"), md=3),
    ], className="mb-4 g-3")

    # Skill tier distribution
    tier_counts = pd.DataFrame([
        {"Tier": t, "Count": int(classifier_metrics["class_distribution"].get(t, 0))}
        for t in SKILL_TIERS
    ])
    fig_dist = px.bar(
        tier_counts, x="Tier", y="Count",
        color="Tier", color_discrete_map=TIER_COLORS,
        title="Skill Tier Distribution (Classification Dataset)",
    )
    fig_dist.update_layout(showlegend=False, height=350)

    # Rating histogram from player data
    fig_rating = px.histogram(
        player_df, x="avg_elo", nbins=60,
        title="Player Rating Distribution",
        labels={"avg_elo": "Average Elo"},
        color_discrete_sequence=["#3498db"],
    )
    fig_rating.update_layout(height=350)

    # Accuracy by tier grouped bar
    acc_melted = accuracy_df.melt(id_vars="Skill Tier", var_name="Metric", value_name="Value")
    fig_acc = px.bar(
        acc_melted, x="Skill Tier", y="Value", color="Metric",
        barmode="group", title="Accuracy Metrics by Skill Tier",
    )
    fig_acc.update_layout(height=370)

    return html.Div([
        cards,
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_dist), md=6),
            dbc.Col(dcc.Graph(figure=fig_rating), md=6),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_acc), md=12),
        ]),
    ])


# ---------------------------------------------------------------------------
# TAB 2 – Player Cluster Map
# ---------------------------------------------------------------------------
def build_cluster_tab():
    controls = dbc.Card(dbc.CardBody([
        html.H6("Color By", className="fw-bold"),
        dbc.RadioItems(
            id="cluster-color-by",
            options=[
                {"label": "Cluster Archetype", "value": "cluster_name"},
                {"label": "Skill Tier", "value": "skill_tier"},
            ],
            value="cluster_name",
            inline=True,
            className="mb-3",
        ),
        html.H6("Filter Skill Tier", className="fw-bold"),
        dbc.Checklist(
            id="cluster-tier-filter",
            options=[{"label": t, "value": t} for t in SKILL_TIERS],
            value=SKILL_TIERS,
            inline=True,
            className="mb-3",
        ),
        html.H6("Rating Range", className="fw-bold"),
        dcc.RangeSlider(
            id="cluster-rating-slider",
            min=int(player_df["avg_elo"].min()),
            max=int(player_df["avg_elo"].max()),
            step=50,
            value=[int(player_df["avg_elo"].min()), int(player_df["avg_elo"].max())],
            marks={v: str(v) for v in range(600, 3400, 400)},
            tooltip={"placement": "bottom", "always_visible": False},
        ),
    ]), className="shadow-sm mb-3")

    cluster_info = dbc.Card(dbc.CardBody([
        html.H6("Cluster Summary", className="fw-bold mb-2"),
        html.Div(id="cluster-info-table"),
    ]), className="shadow-sm")

    return html.Div([
        dbc.Row([
            dbc.Col([controls, cluster_info], md=3),
            dbc.Col(dcc.Graph(id="cluster-scatter", style={"height": "650px"}), md=9),
        ]),
    ])


@callback(
    Output("cluster-scatter", "figure"),
    Output("cluster-info-table", "children"),
    Input("cluster-color-by", "value"),
    Input("cluster-tier-filter", "value"),
    Input("cluster-rating-slider", "value"),
)
def update_cluster_scatter(color_by, tiers, rating_range):
    df = player_df[
        (player_df["skill_tier"].isin(tiers))
        & (player_df["avg_elo"] >= rating_range[0])
        & (player_df["avg_elo"] <= rating_range[1])
    ].copy()

    if color_by == "skill_tier":
        fig = px.scatter(
            df, x="x", y="y", color="skill_tier",
            color_discrete_map=TIER_COLORS,
            hover_data=["player", "avg_elo", "num_games", "cluster_name"],
            title=f"Player Embedding Map — {len(df):,} players",
            labels={"x": "t-SNE Dim 1", "y": "t-SNE Dim 2", "skill_tier": "Skill Tier"},
            category_orders={"skill_tier": SKILL_TIERS},
        )
    else:
        fig = px.scatter(
            df, x="x", y="y", color="cluster_name",
            color_discrete_sequence=CLUSTER_COLORS,
            hover_data=["player", "avg_elo", "num_games", "skill_tier"],
            title=f"Player Embedding Map — {len(df):,} players",
            labels={"x": "t-SNE Dim 1", "y": "t-SNE Dim 2", "cluster_name": "Archetype"},
        )

    fig.update_traces(marker=dict(size=4, opacity=0.6, line=dict(width=0)))
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40),
    )

    # Build cluster summary table
    summary_rows = []
    for cid in sorted(cluster_name_map.keys()):
        info = clustering_results["cluster_names"][str(cid)]
        summary_rows.append(
            html.Tr([
                html.Td(info["name"], style={"fontSize": "0.85rem"}),
                html.Td(f"{info['size']:,}", style={"fontSize": "0.85rem"}),
                html.Td(f"{info['avg_elo']:.0f}", style={"fontSize": "0.85rem"}),
            ])
        )
    table = dbc.Table(
        [html.Thead(html.Tr([html.Th("Archetype"), html.Th("Size"), html.Th("Avg Elo")]))]
        + [html.Tbody(summary_rows)],
        bordered=True, size="sm", striped=True,
    )

    return fig, table


# ---------------------------------------------------------------------------
# TAB 3 – Time Analysis
# ---------------------------------------------------------------------------
def build_time_tab():
    # Heatmap
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=time_heatmap_df.values,
        x=time_heatmap_df.columns.tolist(),
        y=time_heatmap_df.index.tolist(),
        colorscale="YlOrRd",
        text=time_heatmap_df.values.round(2),
        texttemplate="%{text}s",
        hovertemplate="Tier: %{y}<br>Phase: %{x}<br>Avg Time: %{z:.2f}s<extra></extra>",
    ))
    fig_heatmap.update_layout(
        title="Average Time per Move (seconds) by Skill Tier & Game Phase",
        xaxis_title="Game Phase", yaxis_title="Skill Tier",
        height=380,
    )

    # Time variance by tier using player-level data
    tv_data = []
    for phase, col in [("Opening", "time_variance_opening_mean"), ("Middlegame", "time_variance_middlegame_mean"), ("Endgame", "time_variance_endgame_mean")]:
        if col in player_df.columns:
            for tier in SKILL_TIERS:
                vals = player_df[player_df["skill_tier"] == tier][col]
                tv_data.append({"Phase": phase, "Skill Tier": tier, "Time Variance": vals.mean()})
    tv_df = pd.DataFrame(tv_data)
    fig_variance = px.bar(
        tv_df, x="Phase", y="Time Variance", color="Skill Tier",
        barmode="group", color_discrete_map=TIER_COLORS,
        title="Average Time Variance by Skill Tier & Game Phase",
        category_orders={"Skill Tier": SKILL_TIERS},
    )
    fig_variance.update_layout(height=380)

    # Time trouble frequency
    if "time_trouble_frequency_mean" in player_df.columns:
        tt_data = []
        for tier in SKILL_TIERS:
            vals = player_df[player_df["skill_tier"] == tier]["time_trouble_frequency_mean"]
            tt_data.append({"Skill Tier": tier, "Time Trouble Freq": vals.mean()})
        tt_df = pd.DataFrame(tt_data)
        fig_tt = px.bar(
            tt_df, x="Skill Tier", y="Time Trouble Freq",
            color="Skill Tier", color_discrete_map=TIER_COLORS,
            title="Average Time Trouble Frequency by Skill Tier",
        )
        fig_tt.update_layout(showlegend=False, height=370)
    else:
        fig_tt = go.Figure()

    return html.Div([
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_heatmap), md=6),
            dbc.Col(dcc.Graph(figure=fig_variance), md=6),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_tt), md=6),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H5("Key Time Insights", className="fw-bold"),
                html.Ul([
                    html.Li("Beginners spend significantly more time per move than higher-rated players."),
                    html.Li("Time variance in the opening is highest for beginners, suggesting inconsistent opening preparation."),
                    html.Li("Advanced and expert players show more consistent time management across all phases."),
                    html.Li("Time trouble frequency decreases with increasing skill level."),
                ], className="mb-0"),
            ]), className="shadow-sm h-100"), md=6),
        ]),
    ])


# ---------------------------------------------------------------------------
# TAB 4 – Classification Results
# ---------------------------------------------------------------------------
def build_classification_tab():
    # Confusion matrix
    cm_norm = cm_array.astype(float) / cm_array.sum(axis=1, keepdims=True)
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm_norm,
        x=cm_labels,
        y=cm_labels,
        colorscale="Blues",
        text=cm_array,
        texttemplate="%{text}",
        hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{text}<br>Rate: %{z:.2%}<extra></extra>",
    ))
    fig_cm.update_layout(
        title="Skill Tier Classification — Confusion Matrix",
        xaxis_title="Predicted Tier", yaxis_title="Actual Tier",
        height=450,
    )

    # Feature importance
    fi = feature_importance.head(15).copy()
    fi["feature_clean"] = fi["feature"].str.replace("white_", "").str.replace("_", " ").str.title()
    fig_fi = px.bar(
        fi, x="importance", y="feature_clean", orientation="h",
        color="importance", color_continuous_scale="Viridis",
        title="Top 15 Feature Importances (Random Forest)",
        labels={"importance": "Importance", "feature_clean": "Feature"},
    )
    fig_fi.update_layout(yaxis=dict(autorange="reversed"), height=450, coloraxis_showscale=False)

    # Metrics summary
    m = classifier_metrics["metrics"]
    metrics_table = dbc.Table(
        [
            html.Thead(html.Tr([html.Th("Metric"), html.Th("Value")])),
            html.Tbody([
                html.Tr([html.Td("Test Accuracy"), html.Td(f"{m['test_accuracy']:.1%}")]),
                html.Tr([html.Td("Adjacent Accuracy (±1 tier)"), html.Td(f"{m['adjacent_accuracy']:.1%}")]),
                html.Tr([html.Td("Macro Precision"), html.Td(f"{m['macro_precision']:.3f}")]),
                html.Tr([html.Td("Macro Recall"), html.Td(f"{m['macro_recall']:.3f}")]),
                html.Tr([html.Td("Macro F1"), html.Td(f"{m['macro_f1']:.3f}")]),
                html.Tr([html.Td("Train / Val / Test Split"),
                         html.Td(f"{classifier_metrics['train_size']:,} / {classifier_metrics['val_size']:,} / {classifier_metrics['test_size']:,}")]),
            ]),
        ],
        bordered=True, striped=True, hover=True, size="sm",
    )

    return html.Div([
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_cm), md=7),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H5("Classification Metrics", className="fw-bold mb-3"),
                metrics_table,
                html.Hr(),
                html.P([
                    html.Strong("Model: "), "Random Forest (18 behavioral features)",
                ], className="mb-1"),
                html.P([
                    html.Strong("Note: "), "Adjacent accuracy counts predictions within ±1 skill tier as correct.",
                ], className="mb-0 text-muted", style={"fontSize": "0.85rem"}),
            ]), className="shadow-sm"), md=5),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_fi), md=12),
        ]),
    ])


# ---------------------------------------------------------------------------
# TAB 5 – Cluster Analysis
# ---------------------------------------------------------------------------
def build_cluster_analysis_tab():
    # Cluster size pie
    clu_sizes = []
    for cid, info in clustering_results["cluster_names"].items():
        clu_sizes.append({"Archetype": info["name"], "Size": info["size"], "Avg Elo": round(info["avg_elo"])})
    clu_df = pd.DataFrame(clu_sizes)
    fig_pie = px.pie(
        clu_df, values="Size", names="Archetype",
        title="Cluster Size Distribution",
        color_discrete_sequence=CLUSTER_COLORS,
        hole=0.35,
    )
    fig_pie.update_layout(height=400)

    # Skill composition per cluster (stacked bar)
    comp_data = []
    for _, row in cluster_stats.iterrows():
        cid = int(row["cluster"])
        name = cluster_name_map.get(cid, f"Cluster {cid}")
        for tier in SKILL_TIERS:
            col = f"pct_{tier.lower()}"
            if col in row:
                comp_data.append({"Cluster": name, "Skill Tier": tier, "Percentage": row[col]})
    comp_df = pd.DataFrame(comp_data)
    fig_comp = px.bar(
        comp_df, x="Cluster", y="Percentage", color="Skill Tier",
        color_discrete_map=TIER_COLORS,
        title="Skill Tier Composition per Cluster",
        category_orders={"Skill Tier": SKILL_TIERS},
    )
    fig_comp.update_layout(height=400, barmode="stack")

    # Avg Elo by cluster
    fig_elo = px.bar(
        clu_df, x="Archetype", y="Avg Elo",
        color="Archetype", color_discrete_sequence=CLUSTER_COLORS,
        title="Average Elo by Cluster",
    )
    fig_elo.update_layout(showlegend=False, height=370)

    # Clustering methods comparison table
    mc = method_comparison.copy()
    mc["silhouette_score"] = mc["silhouette_score"].round(3)
    mc["calinski_harabasz_index"] = mc["calinski_harabasz_index"].round(1)
    mc["davies_bouldin_index"] = mc["davies_bouldin_index"].round(3)
    method_table = dbc.Table.from_dataframe(
        mc[["method", "n_clusters", "silhouette_score", "calinski_harabasz_index", "davies_bouldin_index"]].rename(
            columns={
                "method": "Method",
                "n_clusters": "k",
                "silhouette_score": "Silhouette",
                "calinski_harabasz_index": "Calinski-Harabasz",
                "davies_bouldin_index": "Davies-Bouldin",
            }
        ),
        bordered=True, striped=True, hover=True, size="sm",
    )

    return html.Div([
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_pie), md=5),
            dbc.Col(dcc.Graph(figure=fig_comp), md=7),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_elo), md=7),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H5("Clustering Method Comparison", className="fw-bold mb-3"),
                method_table,
                html.Hr(),
                html.P([
                    html.Strong("Selected: "), "K-Means (k=5) — best balance of silhouette score and interpretability.",
                ], className="mb-0 text-muted", style={"fontSize": "0.85rem"}),
            ]), className="shadow-sm"), md=5),
        ]),
    ])


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
navbar = dbc.Navbar(
    dbc.Container([
        dbc.Row([
            dbc.Col(html.Span("♟", style={"fontSize": "1.8rem"}), width="auto"),
            dbc.Col(dbc.NavbarBrand("ChessInsight Dashboard", className="ms-2 fw-bold")),
        ], align="center", className="g-0"),
        dbc.NavbarToggler(id="navbar-toggler"),
        html.Small("Team 029 · CSE 6242 · Spring 2026", className="text-light ms-auto"),
    ], fluid=True),
    color="dark",
    dark=True,
    className="mb-3",
)

tabs = dbc.Tabs([
    dbc.Tab(label="Overview", tab_id="overview", className="p-3"),
    dbc.Tab(label="Player Cluster Map", tab_id="clusters", className="p-3"),
    dbc.Tab(label="Time Analysis", tab_id="time", className="p-3"),
    dbc.Tab(label="Classification", tab_id="classification", className="p-3"),
    dbc.Tab(label="Cluster Analysis", tab_id="cluster-analysis", className="p-3"),
], id="tabs", active_tab="overview", className="mb-3")

app.layout = dbc.Container([
    navbar,
    tabs,
    html.Div(id="tab-content"),
], fluid=True)


@callback(Output("tab-content", "children"), Input("tabs", "active_tab"))
def render_tab(active_tab):
    if active_tab == "overview":
        return build_overview_tab()
    elif active_tab == "clusters":
        return build_cluster_tab()
    elif active_tab == "time":
        return build_time_tab()
    elif active_tab == "classification":
        return build_classification_tab()
    elif active_tab == "cluster-analysis":
        return build_cluster_analysis_tab()
    return html.P("Select a tab.")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n  ChessInsight Dashboard")
    print("  " + "=" * 40)
    print("  Open http://127.0.0.1:8050 in your browser\n")
    app.run(debug=True, port=8050)
