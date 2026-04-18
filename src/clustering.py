"""
Behavioral Clustering Module - Identify player archetypes
Team 029 - CSE6242 Spring 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle
import json

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, Birch
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
import umap

import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import PROCESSED_DATA_DIR, MODELS_DIR, RANDOM_STATE, N_CLUSTERS_RANGE


# ---------------------------------------------------------------------------
# Data preparation and k search
# ---------------------------------------------------------------------------


def prepare_clustering_data(
    player_features: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str]]:
    """Prepare player-level features for clustering.

    Drops identifiers, fills missing values, and removes constant columns.
    """
    exclude_cols = ["player", "num_games", "skill_tier", "avg_elo"]
    feature_cols = [
        c
        for c in player_features.columns
        if c not in exclude_cols
        and player_features[c].dtype in [np.float64, np.int64, np.float32, np.int32]
    ]

    X = player_features[feature_cols].copy()
    X = X.fillna(X.mean())

    constant_cols = X.columns[X.std() == 0]
    if len(constant_cols) > 0:
        print(f"Removing constant columns: {list(constant_cols)}")
        X = X.drop(columns=constant_cols)

    return X, list(X.columns)


def find_optimal_k(X: np.ndarray, k_range: Tuple[int, int] = N_CLUSTERS_RANGE) -> Dict:
    """Find optimal number of clusters using k-means and internal metrics."""
    results = {
        "k_values": [],
        "inertia": [],
        "silhouette": [],
        "calinski_harabasz": [],
        "davies_bouldin": [],
    }

    print("Evaluating cluster counts...")
    for k in range(k_range[0], k_range[1] + 1):
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = kmeans.fit_predict(X)

        results["k_values"].append(k)
        results["inertia"].append(kmeans.inertia_)
        results["silhouette"].append(silhouette_score(X, labels))
        results["calinski_harabasz"].append(calinski_harabasz_score(X, labels))
        results["davies_bouldin"].append(davies_bouldin_score(X, labels))

        print(
            f"  k={k}: Silhouette={results['silhouette'][-1]:.4f}, "
            f"CH={results['calinski_harabasz'][-1]:.1f}"
        )

    best_idx = np.argmax(results["silhouette"])
    results["optimal_k"] = int(results["k_values"][best_idx])
    print(f"\nOptimal k based on silhouette score: {results['optimal_k']}")

    return results


# ---------------------------------------------------------------------------
# Core clustering
# ---------------------------------------------------------------------------


def _cluster_with_method(
    X_scaled: np.ndarray,
    n_clusters: int,
    method: str,
) -> Tuple[object, np.ndarray, Optional[np.ndarray]]:
    """Run a single clustering method on standardized features."""
    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
        labels = model.fit_predict(X_scaled)
        centers = getattr(model, "cluster_centers_", None)
    elif method == "hierarchical":
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(X_scaled)
        centers = None
    elif method == "dbscan":
        model = DBSCAN(eps=0.5, min_samples=5)
        labels = model.fit_predict(X_scaled)
        centers = None
    elif method == "gmm":
        model = GaussianMixture(n_components=n_clusters, random_state=RANDOM_STATE)
        labels = model.fit_predict(X_scaled)
        centers = getattr(model, "means_", None)
    elif method == "birch":
        model = Birch(n_clusters=n_clusters)
        labels = model.fit_predict(X_scaled)
        centers = getattr(model, "subcluster_centers_", None)
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    return model, labels, centers


def perform_clustering(
    X: pd.DataFrame,
    n_clusters: int = 5,
    method: str = "kmeans",
    compute_embedding: bool = True,
) -> Dict:
    """Cluster in standardized feature space then compute an embedding for viz."""
    print(f"Performing {method} clustering with k={n_clusters}...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model, labels, centers = _cluster_with_method(X_scaled, n_clusters, method)

    unique_labels = set(labels)
    n_found = len(unique_labels) - (1 if -1 in unique_labels else 0)
    print(f"Method {method} produced {n_found} non-noise clusters")

    if n_found < 3:
        print("Warning: very few clusters found; metrics may be unreliable.")

    if len(unique_labels) > 1:
        silhouette = silhouette_score(X_scaled, labels)
        calinski = calinski_harabasz_score(X_scaled, labels)
        davies = davies_bouldin_score(X_scaled, labels)
    else:
        silhouette = calinski = davies = 0.0

    print("Clustering metrics:")
    print(f"  Silhouette Score: {silhouette:.4f}")
    print(f"  Calinski-Harabasz Index: {calinski:.1f}")
    print(f"  Davies-Bouldin Index: {davies:.4f}")

    embedding_2d = None
    embedding_method: Optional[str] = None
    if compute_embedding:
        try:
            print("Computing UMAP embedding for visualization...")
            reducer = umap.UMAP(
                n_components=2,
                random_state=RANDOM_STATE,
                n_neighbors=30,
                min_dist=0.1,
            )
            embedding_2d = reducer.fit_transform(X_scaled)
            embedding_method = "umap"
            print("UMAP embedding complete")
        except Exception as e:
            print(f"UMAP failed ({e}); falling back to t-SNE")
            tsne = TSNE(
                n_components=2,
                random_state=RANDOM_STATE,
                perplexity=min(30, len(X_scaled) - 1),
            )
            embedding_2d = tsne.fit_transform(X_scaled)
            embedding_method = "tsne"
            print("t-SNE embedding complete")

    return {
        "model": model,
        "scaler": scaler,
        "pca": None,
        "method": method,
        "n_clusters": n_clusters,
        "labels": labels,
        "cluster_centers": centers,
        "feature_columns": list(X.columns),
        "metrics": {
            "silhouette_score": silhouette,
            "calinski_harabasz_index": calinski,
            "davies_bouldin_index": davies,
            "pca_explained_variance": 0.0,
        },
        "embedding_2d": embedding_2d,
        "embedding_method": embedding_method,
        "X_scaled": X_scaled,
        "X_pca": X_scaled,
    }


def compare_clustering_methods(
    X: pd.DataFrame, n_clusters: int, methods: Optional[List[str]] = None
) -> pd.DataFrame:
    """Run multiple clustering algorithms on the same feature matrix."""
    if methods is None:
        methods = ["kmeans", "birch"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rows: List[Dict] = []
    for method in methods:
        try:
            model, labels, _ = _cluster_with_method(X_scaled, n_clusters, method)
            if len(set(labels)) > 1:
                silhouette = silhouette_score(X_scaled, labels)
                calinski = calinski_harabasz_score(X_scaled, labels)
                davies = davies_bouldin_score(X_scaled, labels)
            else:
                silhouette = calinski = davies = 0.0

            rows.append(
                {
                    "method": method,
                    "n_clusters": n_clusters,
                    "silhouette_score": silhouette,
                    "calinski_harabasz_index": calinski,
                    "davies_bouldin_index": davies,
                    "pca_explained_variance": 0.0,
                }
            )
        except Exception as e:
            print(f"Skipping method {method} due to error: {e}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cluster analysis and reporting
# ---------------------------------------------------------------------------


def analyze_clusters(
    player_features: pd.DataFrame, labels: np.ndarray, feature_columns: List[str]
) -> pd.DataFrame:
    """Analyze cluster characteristics and compute per-feature means."""
    df = player_features.copy()
    df["cluster"] = labels

    cluster_stats: List[Dict] = []

    for cluster_id in sorted(df["cluster"].unique()):
        if cluster_id == -1:
            continue

        cluster_data = df[df["cluster"] == cluster_id]

        stats: Dict = {
            "cluster": int(cluster_id),
            "size": int(len(cluster_data)),
            "pct_of_total": len(cluster_data) / len(df) * 100,
            "avg_elo": (
                float(cluster_data["avg_elo"].mean())
                if "avg_elo" in cluster_data.columns
                else np.nan
            ),
            "avg_games": (
                float(cluster_data["num_games"].mean())
                if "num_games" in cluster_data.columns
                else np.nan
            ),
        }

        if "skill_tier" in cluster_data.columns:
            tier_dist = cluster_data["skill_tier"].value_counts(normalize=True)
            for tier in ["Beginner", "Intermediate", "Advanced", "Expert"]:
                stats[f"pct_{tier.lower()}"] = tier_dist.get(tier, 0) * 100

        for col in feature_columns:
            if col in cluster_data.columns:
                stats[f"{col}_mean"] = float(cluster_data[col].mean())

        cluster_stats.append(stats)

    return pd.DataFrame(cluster_stats)


def name_clusters(
    cluster_stats: pd.DataFrame, player_features: pd.DataFrame, labels: np.ndarray
) -> Dict[int, Dict]:
    """Name clusters numerically with a concise description."""
    df = player_features.copy()
    df["cluster"] = labels

    cluster_names: Dict[int, Dict] = {}

    for _, row in cluster_stats.iterrows():
        cid = int(row["cluster"])
        cluster_data = df[df["cluster"] == cid]

        size = int(row["size"])
        avg_elo = float(row["avg_elo"]) if pd.notna(row["avg_elo"]) else None

        dominant_tier = None
        if "skill_tier" in cluster_data.columns and not cluster_data.empty:
            dominant = cluster_data["skill_tier"].mode()
            if not dominant.empty:
                dominant_tier = dominant.iat[0]

        name = f"Cluster {cid + 1}"

        desc_bits = [f"{size} players"]
        if avg_elo is not None:
            desc_bits.append(f"avg Elo ≈ {avg_elo:.0f}")
        if dominant_tier is not None:
            desc_bits.append(f"dominant tier {dominant_tier}")

        description = "Cluster of players with " + ", ".join(desc_bits) + "."

        cluster_names[cid] = {
            "name": name,
            "description": description,
            "characteristics": [],
            "size": size,
            "avg_elo": avg_elo,
        }

    return cluster_names


def save_clustering_results(
    results: Dict,
    cluster_stats: pd.DataFrame,
    cluster_names: Dict,
    model_name: str = "player_clustering",
):
    """Save clustering model, metrics, stats, and embeddings to disk."""
    model_path = MODELS_DIR / f"{model_name}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "model": results["model"],
                "scaler": results["scaler"],
                "pca": results["pca"],
                "feature_columns": results["feature_columns"],
            },
            f,
        )
    print(f"Saved clustering model to {model_path}")

    metrics_path = MODELS_DIR / f"{model_name}_results.json"
    results_to_save = {
        "method": results["method"],
        "n_clusters": results["n_clusters"],
        "metrics": results["metrics"],
        "cluster_names": cluster_names,
    }
    with open(metrics_path, "w") as f:
        json.dump(results_to_save, f, indent=2)
    print(f"Saved clustering results to {metrics_path}")

    stats_path = MODELS_DIR / f"{model_name}_statistics.csv"
    cluster_stats.to_csv(stats_path, index=False)
    print(f"Saved cluster statistics to {stats_path}")

    embedding_path = PROCESSED_DATA_DIR / f"{model_name}_embeddings.parquet"
    embedding_df = pd.DataFrame(
        {
            "x": results["embedding_2d"][:, 0],
            "y": results["embedding_2d"][:, 1],
            "cluster": results["labels"],
        }
    )
    embedding_df.to_parquet(embedding_path)
    print(f"Saved embeddings to {embedding_path}")


def print_clustering_summary(
    cluster_stats: pd.DataFrame, cluster_names: Dict, metrics: Dict
):
    """Print a formatted summary of clustering results."""
    print("\n" + "=" * 60)
    print("CLUSTERING RESULTS SUMMARY")
    print("=" * 60)

    print("\nClustering Metrics:")
    print(f"  Silhouette Score: {metrics['silhouette_score']:.4f}")
    print(f"  Calinski-Harabasz Index: {metrics['calinski_harabasz_index']:.1f}")
    print(f"  Davies-Bouldin Index: {metrics['davies_bouldin_index']:.4f}")

    print(f"\nIdentified {len(cluster_names)} Clusters:")
    print("-" * 60)

    for cluster_id, info in sorted(cluster_names.items()):
        print(f"\n  Cluster {cluster_id + 1}: {info['name']}")
        print(f"    Size: {info['size']} players")
        if info["avg_elo"]:
            print(f"    Avg Elo: {info['avg_elo']:.0f}")
        print(f"    Description: {info['description']}")
        if info["characteristics"]:
            print(f"    Characteristics: {', '.join(info['characteristics'])}")


if __name__ == "__main__":
    print("ChessInsight Behavioral Clustering")
    print("=" * 50)

    player_path = PROCESSED_DATA_DIR / "player_features.parquet"

    if player_path.exists():
        player_features = pd.read_parquet(player_path)
        print(f"Loaded features for {len(player_features)} players")

        X, feature_cols = prepare_clustering_data(player_features)
        print(f"Using {len(feature_cols)} features for clustering")

        k_results = find_optimal_k(X.values, k_range=(3, 5))
        optimal_k = int(k_results.get("optimal_k", 4))
        print(
            f"Using n_clusters={optimal_k} for behavioral clustering "
            "based on internal metrics."
        )

        print(
            f"\nEvaluating alternative clustering methods on the same features (k = {optimal_k})..."
        )
        method_comparison_df = compare_clustering_methods(X, n_clusters=optimal_k)
        comparison_path = MODELS_DIR / "clustering_method_comparison.csv"
        method_comparison_df.to_csv(comparison_path, index=False)
        print(f"Saved clustering method comparison to {comparison_path}")
        if not method_comparison_df.empty:
            print(method_comparison_df.to_string(index=False))

        if not method_comparison_df.empty:
            method_sorted = method_comparison_df.sort_values(
                ["silhouette_score", "davies_bouldin_index"],
                ascending=[False, True],
            )
            best_method = method_sorted.iloc[0]["method"]
        else:
            best_method = "kmeans"

        print(f"\nSelected primary clustering method: {best_method} (k = {optimal_k})")

        results = perform_clustering(X, n_clusters=optimal_k, method=best_method)

        cluster_stats = analyze_clusters(
            player_features, results["labels"], feature_cols
        )

        cluster_names = name_clusters(cluster_stats, player_features, results["labels"])

        print_clustering_summary(cluster_stats, cluster_names, results["metrics"])

        save_clustering_results(results, cluster_stats, cluster_names)
    else:
        print(f"No player features found at {player_path}")
        print("Please run feature_extractor.py first")
