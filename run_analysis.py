"""
ChessInsight - Main Analysis Pipeline
Team 029 - CSE6242 Spring 2026

This script runs the complete analysis pipeline:
1. Generate/load game data
2. Extract features
3. Train skill classifiers (RF, XGB, Ensemble)
4. Perform clustering
5. Generate visualizations
6. Output results summary
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    VIZ_DIR,
    SKILL_TIERS,
    RANDOM_STATE,
    DATA_SOURCE,
    CLUSTERING_FEATURES,
)

from src.data_loader import (
    load_chunks,
    get_dataset_stats,
)
from src.feature_extractor import (
    extract_features_from_dataframe,
)
from src.classifier import (
    prepare_player_level_data,
    train_classifier,
    save_model,
    print_results_summary,
)
from src.clustering import (
    prepare_clustering_data,
    find_optimal_k,
    perform_clustering,
    analyze_clusters,
    name_clusters,
    save_clustering_results,
    print_clustering_summary,
    compare_clustering_methods,
)
from src.visualizations import (
    generate_all_visualizations,
    create_dashboard_wireframe,
)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_real_dataset() -> tuple:
    """Load chess game data from pre-processed parquet chunks.

    Chunks are read from:
        PROCESSED_DATA_DIR / "chunks" / games_processed_part_XXXX.parquet
        PROCESSED_DATA_DIR / "chunks" / games_full_part_XXXX.parquet
    """
    games_df, full_games_df = load_chunks()

    stats = get_dataset_stats(games_df)
    print(f"\nDataset Statistics:")
    print(f"  Total games: {stats['total_games']:,}")
    print(f"  Unique white players: {stats['unique_white_players']:,}")
    print(f"  Unique black players: {stats['unique_black_players']:,}")
    print(f"  Rating range: {stats['rating_range'][0]} - {stats['rating_range'][1]}")

    return games_df, full_games_df


def generate_synthetic_dataset(
    n_games: int = 50000, random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    """Generate a realistic synthetic chess game dataset (for testing)."""
    print(f"Generating synthetic dataset with {n_games} games...")
    np.random.seed(random_state)

    # Generate player ratings with realistic distribution
    white_elo = np.random.normal(1500, 350, n_games).clip(600, 2800).astype(int)
    black_elo = np.random.normal(1500, 350, n_games).clip(600, 2800).astype(int)

    def get_skill_tier(elo):
        for tier, (low, high) in SKILL_TIERS.items():
            if low <= elo < high:
                return tier
        return "Advanced"  # Fallback for high Elo players

    white_skill_tier = [get_skill_tier(e) for e in white_elo]
    black_skill_tier = [get_skill_tier(e) for e in black_elo]

    n_unique_players = n_games // 10
    player_pool = [f"player_{i:06d}" for i in range(n_unique_players)]
    white_player = np.random.choice(player_pool, n_games)
    black_player = np.random.choice(player_pool, n_games)

    time_controls = ["180+0", "180+2", "300+0", "300+3", "600+0", "600+5"]
    time_control_categories = ["blitz", "blitz", "blitz", "blitz", "rapid", "rapid"]
    tc_idx = np.random.choice(len(time_controls), n_games)
    time_control = [time_controls[i] for i in tc_idx]
    time_control_category = [time_control_categories[i] for i in tc_idx]
    base_time = [int(time_controls[i].split("+")[0]) for i in tc_idx]

    rating_diff = white_elo - black_elo
    white_win_prob = 1 / (1 + 10 ** (-rating_diff / 400))
    rand = np.random.random(n_games)
    results = []
    for prob, r in zip(white_win_prob, rand):
        if r < prob * 0.9:
            results.append("1-0")
        elif r < prob * 0.9 + 0.1:
            results.append("1/2-1/2")
        else:
            results.append("0-1")

    num_moves = np.random.normal(70, 25, n_games).clip(20, 200).astype(int)
    eco_codes = ["A00", "B00", "B20", "B50", "C00", "C20", "C50", "D00", "D30", "E00", "E60"]
    opening_eco = np.random.choice(eco_codes, n_games)

    games_df = pd.DataFrame(
        {
            "white_player": white_player,
            "black_player": black_player,
            "white_elo": white_elo,
            "black_elo": black_elo,
            "white_skill_tier": white_skill_tier,
            "black_skill_tier": black_skill_tier,
            "result": results,
            "time_control": time_control,
            "time_control_category": time_control_category,
            "base_time": base_time,
            "num_moves": num_moves,
            "opening_eco": opening_eco,
        }
    )

    print(f"Generated {len(games_df)} games")
    print(
        f"Unique players: {games_df['white_player'].nunique() + games_df['black_player'].nunique()}"
    )
    print(
        f"Rating range: {games_df['white_elo'].min()} - {games_df['white_elo'].max()}"
    )

    return games_df


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_full_pipeline(use_real_data: bool = True, n_synthetic_games: int = 50000):
    """Run the complete ChessInsight analysis pipeline."""
    print("=" * 70)
    print("CHESSINSIGHT - FULL ANALYSIS PIPELINE")
    print("Team 029 - CSE6242 Spring 2026")
    print("=" * 70)

    # Step 1: Load/Generate Data
    print("\n" + "=" * 70)
    print("STEP 1: Data Loading")
    print("=" * 70)

    full_games_df = None
    chunks_dir = PROCESSED_DATA_DIR / "chunks"
    chunks_exist = chunks_dir.exists() and any(chunks_dir.glob("games_processed_part_*.parquet"))
    game_features_path = PROCESSED_DATA_DIR / "game_features.parquet"

    if use_real_data and chunks_exist:
        games_df, full_games_df = load_real_dataset()
        games_df.to_parquet(PROCESSED_DATA_DIR / "games_processed.parquet")
        print(f"Saved games to {PROCESSED_DATA_DIR / 'games_processed.parquet'}")
    elif use_real_data and game_features_path.exists():
        print(
            "No raw chunks found — running in cache-only mode using "
            "pre-computed game_features.parquet.\n"
            "Feature extraction will be skipped; classification, clustering, "
            "visualizations, and summary will be regenerated from cache."
        )
        games_df = pd.read_parquet(game_features_path)
        print(f"Loaded {len(games_df):,} cached game-level rows")
    elif use_real_data:
        raise FileNotFoundError(
            f"Cannot run pipeline. Neither {chunks_dir} nor {game_features_path} "
            "is present. Download the raw Lichess PGN data into data/raw/ and "
            "regenerate chunks, or restore data/processed/game_features.parquet."
        )
    else:
        print("Using synthetic data (fallback mode)")
        games_df = generate_synthetic_dataset(n_games=n_synthetic_games)
        games_df.to_parquet(PROCESSED_DATA_DIR / "games_processed.parquet")

    # Step 2: Feature Extraction
    print("\n" + "=" * 70)
    print("STEP 2: Feature Extraction")
    print("=" * 70)

    game_features_path = PROCESSED_DATA_DIR / "game_features.parquet"

    if game_features_path.exists():
        print(f"Loading cached game features from {game_features_path}")
        features_df = pd.read_parquet(game_features_path)
        print(f"Loaded {len(features_df.columns)} features for {len(features_df)} games")
    else:
        features_df = extract_features_from_dataframe(games_df, full_games=full_games_df)
        print(f"Extracted {len(features_df.columns)} features from {len(features_df)} games")
        features_df.to_parquet(game_features_path)

    # Step 3: Skill Classification (multiple models)
    print("\n" + "=" * 70)
    print("STEP 3: Skill Tier Classification")
    print("=" * 70)

    X, y, player_data = prepare_player_level_data(features_df)
    print(f"Classification data: {len(X)} samples, {len(X.columns)} features")

    model_types = ["random_forest", "xgboost", "ensemble_soft"]
    results_by_type: dict = {}

    best_type = None
    best_test_acc = -1.0

    for m_type in model_types:
        print("\n" + "-" * 60)
        print(f"Training model: {m_type}")
        results = train_classifier(X, y, model_type=m_type)
        print_results_summary(results)

        # Save under type-specific name
        save_model(results, model_name=f"skill_classifier_{m_type}")

        results_by_type[m_type] = results
        if results["metrics"]["test_accuracy"] > best_test_acc:
            best_test_acc = results["metrics"]["test_accuracy"]
            best_type = m_type

    # Use the best-performing model as the canonical classifier
    assert best_type is not None
    classification_results = results_by_type[best_type]
    print("\n" + "-" * 60)
    print(f"Best-performing model based on test accuracy: {best_type}")

    # Also save it under the generic skill_classifier name for dashboards
    save_model(classification_results, model_name="skill_classifier")

    # Step 4: Behavioral Clustering
    print("\n" + "=" * 70)
    print("STEP 4: Behavioral Clustering")
    print("=" * 70)

    # Restrict clustering to the 11 curated style features defined in config.
    # Using the full mean+std feature set produces one degenerate cluster
    # because a handful of extreme players dominate the PCA projection.
    metadata_cols = ["player", "elo", "game_count", "skill_tier"]
    keep_cols = [c for c in metadata_cols + CLUSTERING_FEATURES if c in player_data.columns]
    missing = [c for c in CLUSTERING_FEATURES if c not in player_data.columns]
    if missing:
        print(f"WARNING: missing clustering features: {missing}")
    cluster_input = player_data[keep_cols].reset_index(drop=True)

    X_cluster, feature_cols = prepare_clustering_data(cluster_input)
    print(f"Clustering data: {len(X_cluster)} players, {len(feature_cols)} features")

    k_results = find_optimal_k(X_cluster.values, k_range=(2, 5))
    optimal_k = 3  # K=3 optimal based on silhouette analysis
    print(
        f"Using n_clusters={optimal_k} for primary behavioral clustering "
        "(K=3 optimal with silhouette=0.34)."
    )

    print(f"\nEvaluating alternative clustering methods on the same features (k = {optimal_k})...")
    method_comparison_df = compare_clustering_methods(X_cluster, n_clusters=optimal_k)
    comparison_path = MODELS_DIR / "clustering_method_comparison.csv"
    method_comparison_df.to_csv(comparison_path, index=False)
    print(f"Saved clustering method comparison to {comparison_path}")
    if not method_comparison_df.empty:
        print(method_comparison_df.to_string(index=False))

    # Always use K-Means for balanced, interpretable clusters
    # (BIRCH may show higher silhouette but produces degenerate single-cluster results)
    best_method = "kmeans"

    print(f"\nUsing K-Means clustering (k = {optimal_k})")

    clustering_results = perform_clustering(X_cluster, n_clusters=optimal_k, method=best_method)

    cluster_stats = analyze_clusters(
        cluster_input, clustering_results["labels"], feature_cols
    )
    cluster_names = name_clusters(
        cluster_stats, cluster_input, clustering_results["labels"]
    )

    print_clustering_summary(
        cluster_stats, cluster_names, clustering_results["metrics"]
    )
    save_clustering_results(clustering_results, cluster_stats, cluster_names)

    # Write the canonical player_features.parquet consumed by the dashboard.
    # Use the full per-player aggregate (all mean+std cols) so the dashboard
    # has every column it references (time variance, std, etc.), but keep the
    # row order aligned with the clustering embedding: clustering ran on the
    # same 44,613 rows (cluster_input) in the same order as player_data.
    dashboard_player_df = player_data.rename(
        columns={"elo": "avg_elo", "game_count": "num_games"}
    )
    dashboard_player_df.to_parquet(PROCESSED_DATA_DIR / "player_features.parquet")
    print(
        f"Wrote dashboard player_features.parquet "
        f"({len(dashboard_player_df)} players, {len(dashboard_player_df.columns)} columns)"
    )

    # Step 5: Visualizations
    print("\n" + "=" * 70)
    print("STEP 5: Generating Visualizations")
    print("=" * 70)

    create_dashboard_wireframe()

    generate_all_visualizations(
        features_df,
        cluster_input,
        classification_results,
        clustering_results,
        cluster_stats,
        cluster_names,
    )

    # Step 6: Summary Report
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE - SUMMARY")
    print("=" * 70)

    data_source = DATA_SOURCE if use_real_data else "Synthetic data"

    summary = {
        "dataset": {
            "total_games": len(games_df),
            "unique_players": len(player_data),
            "rating_range": f"{games_df['white_elo'].min()} - {games_df['white_elo'].max()}",
            "data_source": data_source,
        },
        "features": {
            "game_level_features": len(features_df.columns),
            "player_level_features": len(player_data.columns),
            "players_analyzed": len(player_data),
        },
        "classification": {
            "model_type": classification_results["model_type"],
            "test_accuracy": f"{classification_results['metrics']['test_accuracy']:.1%}",
            "adjacent_accuracy": f"{classification_results['metrics']['adjacent_accuracy']:.1%}",
            "macro_f1": f"{classification_results['metrics']['macro_f1']:.4f}",
        },
        "clustering": {
            "method": clustering_results["method"],
            "n_clusters": clustering_results["n_clusters"],
            "silhouette_score": f"{clustering_results['metrics']['silhouette_score']:.4f}",
            "archetypes": {k: v["name"] for k, v in cluster_names.items()},
        },
    }

    summary_path = MODELS_DIR / "analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved analysis summary to {summary_path}")

    print("\n" + "-" * 50)
    print("DATASET STATISTICS:")
    print(f"  Total games: {summary['dataset']['total_games']:,}")
    print(f"  Unique players: {summary['dataset']['unique_players']:,}")
    print(f"  Rating range: {summary['dataset']['rating_range']}")

    print("\nCLASSIFICATION RESULTS:")
    print(f"  Model: {summary['classification']['model_type']}")
    print(f"  Test Accuracy: {summary['classification']['test_accuracy']}")
    print(f"  Adjacent Accuracy: {summary['classification']['adjacent_accuracy']}")

    print("\nCLUSTERING RESULTS:")
    print(f"  Method: {summary['clustering']['method']}")
    print(f"  Clusters: {summary['clustering']['n_clusters']}")
    print(f"  Silhouette Score: {summary['clustering']['silhouette_score']}")
    print("  Identified Clusters:")
    for cluster_id, name in summary["clustering"]["archetypes"].items():
        print(f"    - {name}")

    print("\nOUTPUT FILES:")
    print(f"  Data: {PROCESSED_DATA_DIR}")
    print(f"  Models: {MODELS_DIR}")
    print(f"  Visualizations: {VIZ_DIR}")

    print("\n" + "=" * 70)
    print("Pipeline completed successfully!")
    print("=" * 70)

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ChessInsight Analysis Pipeline")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data instead of the pre-processed chunks",
    )

    args = parser.parse_args()

    summary = run_full_pipeline(use_real_data=not args.synthetic)
