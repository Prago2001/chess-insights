"""
Skill Tier Classification Module
Team 029 - CSE6242 Spring 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle
import json

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    SKILL_TIERS,
    MIN_GAMES_PER_PLAYER,
    RANDOM_STATE,
    TEST_SIZE,
    VAL_SIZE,
)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def _add_engineered_features(features_df: pd.DataFrame, target_color: str) -> pd.DataFrame:
    """Add engineered behavioral features derived from base features.

    These are lightweight ratios and interaction terms that can be computed
    directly from the existing clock-based features without re-running
    feature extraction.
    """
    df = features_df.copy()
    p = f"{target_color}_"

    # Guard: only add features if base columns exist
    time_cols = [
        f"{p}avg_time_opening",
        f"{p}avg_time_middlegame",
        f"{p}avg_time_endgame",
    ]
    if all(c in df.columns for c in time_cols):
        total_time = (
            df[f"{p}avg_time_opening"]
            + df[f"{p}avg_time_middlegame"]
            + df[f"{p}avg_time_endgame"]
        )
        total_time = total_time.replace(0, np.nan)

        df[f"{p}time_opening_frac"] = df[f"{p}avg_time_opening"] / total_time
        df[f"{p}time_middlegame_frac"] = df[f"{p}avg_time_middlegame"] / total_time
        df[f"{p}time_endgame_frac"] = df[f"{p}avg_time_endgame"] / total_time

        df[f"{p}opening_vs_endgame_time_ratio"] = df[f"{p}avg_time_opening"] / (
            df[f"{p}avg_time_endgame"] + 1e-3
        )

    # Aggregate variance and volatility indicators
    var_cols = [
        f"{p}time_variance_opening",
        f"{p}time_variance_middlegame",
        f"{p}time_variance_endgame",
    ]
    if all(c in df.columns for c in var_cols):
        df[f"{p}time_variance_total"] = (
            df[f"{p}time_variance_opening"]
            + df[f"{p}time_variance_middlegame"]
            + df[f"{p}time_variance_endgame"]
        )

    # Time pressure behaviour
    if (
        f"{p}low_time_move_ratio" in df.columns
        and f"{p}time_trouble_frequency" in df.columns
    ):
        df[f"{p}low_time_to_trouble_ratio"] = df[f"{p}low_time_move_ratio"] / (
            df[f"{p}time_trouble_frequency"] + 1e-3
        )

    # Error rates normalized by game length
    if "num_moves" in df.columns and f"{p}blunder_rate" in df.columns:
        df[f"{p}blunders_per_40_moves"] = (
            df[f"{p}blunder_rate"] * df["num_moves"] / 40.0
        )
    if "num_moves" in df.columns and f"{p}mistake_rate" in df.columns:
        df[f"{p}mistakes_per_40_moves"] = (
            df[f"{p}mistake_rate"] * df["num_moves"] / 40.0
        )

    # Complexity–time interaction feature
    if "avg_position_complexity" in df.columns and f"{p}avg_time_middlegame" in df.columns:
        df[f"{p}complexity_time_product"] = (
            df["avg_position_complexity"] * df[f"{p}avg_time_middlegame"]
        )

    return df


def prepare_classification_data(
    features_df: pd.DataFrame, target_color: str = "white"
) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare data for skill tier classification.

    This function also adds several engineered behavioral features
    (time-distribution ratios, volatility aggregates, normalized error rates).
    """
    # Add engineered features first so they can be selected below
    df = _add_engineered_features(features_df, target_color=target_color)

    # Base feature columns: per-color metrics plus global complexity/opening
    feature_cols: List[str] = []

    per_color_prefix = f"{target_color}_"
    for c in df.columns:
        if c.startswith(per_color_prefix) and c not in [
            f"{per_color_prefix}player",
            f"{per_color_prefix}skill_tier",
            f"{per_color_prefix}elo",
        ]:
            feature_cols.append(c)

    # Global features (same for both colors)
    global_features = [
        "avg_position_complexity",
        "material_imbalance_freq",
        "piece_activity_score",
        "opening_aggression_score",
        "book_deviation_move",
        "num_moves",
        # Time control one-hot features (fixes time variance confounding)
        "is_bullet",
        "is_blitz",
        "is_rapid",
    ]
    for col in global_features:
        if col in df.columns:
            feature_cols.append(col)

    # Remove non-numeric columns and construct X
    X = df[feature_cols].select_dtypes(include=[np.number])

    # Target variable
    y = df[f"{target_color}_skill_tier"]

    # Remove rows with missing values
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]

    return X, y


def _assign_skill_tier(elo: float) -> str:
    """Assign a skill tier based on Elo rating using configured thresholds."""
    for tier, (low, high) in SKILL_TIERS.items():
        if low <= elo < high:
            return tier
    return list(SKILL_TIERS.keys())[-1]  # Default to highest tier


def prepare_player_level_data(
    features_df: pd.DataFrame,
    min_games: int = None,
    add_std_features: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Prepare player-level aggregated data for skill tier classification.

    This aggregates game-level features to player-level by computing mean
    (and optionally std) of features across all games for each player.

    Args:
        features_df: Game-level features DataFrame
        min_games: Minimum games per player (defaults to config MIN_GAMES_PER_PLAYER)
        add_std_features: Whether to add std features for consistency measurement

    Returns:
        X: Feature matrix (player-level)
        y: Skill tier labels
        player_data: Full player DataFrame with metadata
    """
    if min_games is None:
        min_games = MIN_GAMES_PER_PLAYER

    print(f"Preparing player-level data (min {min_games} games per player)...")

    # Get feature columns (excluding metadata)
    white_cols = [c for c in features_df.columns
                  if c.startswith("white_") and c not in ["white_player", "white_skill_tier", "white_elo"]]
    black_cols = [c for c in features_df.columns
                  if c.startswith("black_") and c not in ["black_player", "black_skill_tier", "black_elo"]]
    global_cols = ["avg_position_complexity", "material_imbalance_freq", "piece_activity_score",
                   "opening_aggression_score", "book_deviation_move", "num_moves"]
    global_cols = [c for c in global_cols if c in features_df.columns]

    # Prepare white player data
    df_white = features_df[["white_player", "white_elo"] + white_cols + global_cols].copy()
    df_white = df_white.rename(columns={"white_player": "player", "white_elo": "elo"})
    df_white.columns = [c.replace("white_", "") if c.startswith("white_") else c for c in df_white.columns]

    # Prepare black player data
    df_black = features_df[["black_player", "black_elo"] + black_cols + global_cols].copy()
    df_black = df_black.rename(columns={"black_player": "player", "black_elo": "elo"})
    df_black.columns = [c.replace("black_", "") if c.startswith("black_") else c for c in df_black.columns]

    # Combine all games
    df_all = pd.concat([df_white, df_black], ignore_index=True)
    print(f"  Total player-game records: {len(df_all):,}")

    # Get numeric columns for aggregation
    numeric_cols = [c for c in df_all.select_dtypes(include=[np.number]).columns if c != "elo"]

    # Aggregate by player
    if add_std_features:
        agg_dict = {col: ["mean", "std"] for col in numeric_cols}
        agg_dict["elo"] = "mean"
        player_agg = df_all.groupby("player").agg(agg_dict)
        # Flatten column names
        player_agg.columns = [f"{col}_{stat}" if stat != "mean" else col
                              for col, stat in player_agg.columns]
        player_agg = player_agg.reset_index()
        # Fill NaN std values (single game players) with 0
        std_cols = [c for c in player_agg.columns if c.endswith("_std")]
        player_agg[std_cols] = player_agg[std_cols].fillna(0)
    else:
        agg_dict = {col: "mean" for col in numeric_cols}
        agg_dict["elo"] = "mean"
        player_agg = df_all.groupby("player").agg(agg_dict).reset_index()

    # Add game count
    games_per_player = df_all.groupby("player").size().reset_index(name="game_count")
    player_agg = player_agg.merge(games_per_player, on="player")

    # Filter by minimum games
    player_agg = player_agg[player_agg["game_count"] >= min_games]
    print(f"  Players with {min_games}+ games: {len(player_agg):,}")

    # Assign skill tier based on average Elo
    player_agg["skill_tier"] = player_agg["elo"].apply(_assign_skill_tier)

    # Prepare feature matrix
    exclude_cols = ["player", "elo", "game_count", "skill_tier"]
    feature_cols = [c for c in player_agg.columns if c not in exclude_cols]

    X = player_agg[feature_cols].select_dtypes(include=[np.number])
    y = player_agg["skill_tier"]

    # Remove rows with missing values
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]
    player_data = player_agg[mask]

    print(f"  Final dataset: {len(X):,} players, {len(X.columns)} features")
    print(f"  Skill tier distribution:\n{y.value_counts()}")

    return X, y, player_data


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def _init_base_model(model_type: str):
    """Create an untrained base classifier for the given type."""
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=18,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    if model_type == "xgboost":
        return XGBClassifier(
            n_estimators=250,
            max_depth=8,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown model type: {model_type}")


def train_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "random_forest",
    use_smote: bool = True,
) -> Dict:
    """Train a skill tier classifier.

    Supported model types:
    - "random_forest"
    - "xgboost"
    - "ensemble_soft" (soft-voting ensemble of RF + XGB)
    """
    print(f"Training {model_type} classifier...")
    print(f"Dataset size: {len(X)} samples")
    print(f"Class distribution:\n{y.value_counts()}")

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y_encoded,
        test_size=(TEST_SIZE + VAL_SIZE),
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Apply SMOTE for class balancing
    if use_smote:
        try:
            smote = SMOTE(random_state=RANDOM_STATE)
            X_train_balanced, y_train_balanced = smote.fit_resample(
                X_train_scaled, y_train
            )
            print(f"After SMOTE: {len(X_train_balanced)} training samples")
        except ValueError:
            print("SMOTE failed (likely too few samples), using original data")
            X_train_balanced, y_train_balanced = X_train_scaled, y_train
    else:
        X_train_balanced, y_train_balanced = X_train_scaled, y_train

    # Initialize model(s)
    if model_type in {"random_forest", "xgboost"}:
        model = _init_base_model(model_type)
    elif model_type == "ensemble_soft":
        estimators = [
            ("rf", _init_base_model("random_forest")),
            ("xgb", _init_base_model("xgboost")),
        ]
        model = VotingClassifier(estimators=estimators, voting="soft")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Train model
    model.fit(X_train_balanced, y_train_balanced)

    # Predict on validation and test sets
    y_val_pred = model.predict(X_val_scaled)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    y_test_pred = model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Adjacent accuracy (prediction within ±1 tier)
    adjacent_correct = sum(abs(p - t) <= 1 for p, t in zip(y_test_pred, y_test))
    adjacent_accuracy = adjacent_correct / len(y_test)
    print(f"Adjacent Accuracy (±1 tier): {adjacent_accuracy:.4f}")

    # Detailed metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_test_pred, average="macro"
    )

    # Feature importance (average across ensemble members when applicable)
    feature_importance: Optional[pd.DataFrame] = None
    if model_type == "ensemble_soft":
        importances = []
        for est in model.estimators_:
            if hasattr(est, "feature_importances_"):
                importances.append(est.feature_importances_)
        if importances:
            avg_importance = np.mean(importances, axis=0)
            feature_importance = pd.DataFrame(
                {"feature": X.columns, "importance": avg_importance}
            ).sort_values("importance", ascending=False)
    elif hasattr(model, "feature_importances_"):
        feature_importance = pd.DataFrame(
            {"feature": X.columns, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_test_pred)

    # Classification report
    class_report = classification_report(
        y_test,
        y_test_pred,
        target_names=label_encoder.classes_,
        output_dict=True,
    )

    results = {
        "model": model,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "model_type": model_type,
        "feature_columns": list(X.columns),
        "metrics": {
            "val_accuracy": val_accuracy,
            "test_accuracy": test_accuracy,
            "adjacent_accuracy": adjacent_accuracy,
            "macro_precision": precision,
            "macro_recall": recall,
            "macro_f1": f1,
        },
        "confusion_matrix": conf_matrix,
        "classification_report": class_report,
        "feature_importance": feature_importance,
        "class_distribution": y.value_counts().to_dict(),
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
    }

    return results


def hyperparameter_tuning(
    X: pd.DataFrame, y: pd.Series, model_type: str = "random_forest"
) -> Dict:
    """Perform hyperparameter tuning using GridSearchCV for RF/XGB."""
    print("Performing hyperparameter tuning...")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if model_type == "random_forest":
        model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 15, 20],
            "min_samples_split": [2, 5, 10],
        }
    elif model_type == "xgboost":
        model = XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1)
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [5, 8, 12],
            "learning_rate": [0.05, 0.1, 0.2],
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=1
    )
    grid_search.fit(X_scaled, y_encoded)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")

    return {
        "best_model": grid_search.best_estimator_,
        "best_params": grid_search.best_params_,
        "best_score": grid_search.best_score_,
        "cv_results": grid_search.cv_results_,
    }


# ---------------------------------------------------------------------------
# Persistence and inference
# ---------------------------------------------------------------------------
def save_model(results: Dict, model_name: str = "skill_classifier"):
    """Save trained model and associated objects."""
    model_path = MODELS_DIR / f"{model_name}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "model": results["model"],
                "scaler": results["scaler"],
                "label_encoder": results["label_encoder"],
                "feature_columns": results["feature_columns"],
            },
            f,
        )
    print(f"Saved model to {model_path}")

    metrics_path = MODELS_DIR / f"{model_name}_metrics.json"
    metrics_to_save = {
        "model_type": results["model_type"],
        "metrics": results["metrics"],
        "class_distribution": results["class_distribution"],
        "train_size": results["train_size"],
        "val_size": results["val_size"],
        "test_size": results["test_size"],
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_to_save, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    if results["feature_importance"] is not None:
        importance_path = MODELS_DIR / f"{model_name}_feature_importance.csv"
        results["feature_importance"].to_csv(importance_path, index=False)
        print(f"Saved feature importance to {importance_path}")

    conf_matrix_path = MODELS_DIR / f"{model_name}_confusion_matrix.csv"
    pd.DataFrame(
        results["confusion_matrix"],
        columns=results["label_encoder"].classes_,
        index=results["label_encoder"].classes_,
    ).to_csv(conf_matrix_path)
    print(f"Saved confusion matrix to {conf_matrix_path}")


def load_model(model_name: str = "skill_classifier") -> Dict:
    """Load trained model from disk."""
    model_path = MODELS_DIR / f"{model_name}.pkl"
    with open(model_path, "rb") as f:
        return pickle.load(f)


def predict_skill_tier(features: pd.DataFrame, model_dict: Dict) -> np.ndarray:
    """Predict skill tier for new games."""
    X = features[model_dict["feature_columns"]]
    X_scaled = model_dict["scaler"].transform(X)
    y_pred = model_dict["model"].predict(X_scaled)
    return model_dict["label_encoder"].inverse_transform(y_pred)


def print_results_summary(results: Dict):
    """Print a formatted summary of classification results."""
    print("\n" + "=" * 60)
    print("CLASSIFICATION RESULTS SUMMARY")
    print("=" * 60)

    print(f"\nModel Type: {results['model_type']}")
    print(f"Training samples: {results['train_size']}")
    print(f"Validation samples: {results['val_size']}")
    print(f"Test samples: {results['test_size']}")

    print("\nPerformance Metrics:")
    print(
        f"  Test Accuracy:     {results['metrics']['test_accuracy']:.4f} "
        f"({results['metrics']['test_accuracy']*100:.1f}%)"
    )
    print(
        f"  Adjacent Accuracy: {results['metrics']['adjacent_accuracy']:.4f} "
        f"({results['metrics']['adjacent_accuracy']*100:.1f}%)"
    )
    print(f"  Macro F1-Score:    {results['metrics']['macro_f1']:.4f}")

    print("\nClass Distribution:")
    for tier, count in results["class_distribution"].items():
        print(f"  {tier}: {count}")

    if results["feature_importance"] is not None:
        print("\nTop 10 Most Important Features:")
        for i, row in results["feature_importance"].head(10).iterrows():
            print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")

    print("\nConfusion Matrix:")
    print(
        pd.DataFrame(
            results["confusion_matrix"],
            columns=results["label_encoder"].classes_,
            index=results["label_encoder"].classes_,
        )
    )


if __name__ == "__main__":
    print("ChessInsight Skill Classifier")
    print("=" * 50)
    print(f"Using {len(SKILL_TIERS)}-tier classification: {list(SKILL_TIERS.keys())}")
    print(f"Minimum games per player: {MIN_GAMES_PER_PLAYER}")
    print()

    features_path = PROCESSED_DATA_DIR / "game_features.parquet"

    if features_path.exists():
        features_df = pd.read_parquet(features_path)
        print(f"Loaded features for {len(features_df):,} games")

        # Use player-level classification (aggregated across games)
        X, y, player_data = prepare_player_level_data(
            features_df,
            min_games=MIN_GAMES_PER_PLAYER,
            add_std_features=True,
        )

        # Train ensemble by default for standalone runs
        results = train_classifier(X, y, model_type="ensemble_soft")
        print_results_summary(results)
        save_model(results)

        # Save player data for downstream use (e.g., clustering, visualization)
        player_data_path = PROCESSED_DATA_DIR / "player_features.parquet"
        player_data.to_parquet(player_data_path, index=False)
        print(f"\nSaved player-level features to {player_data_path}")
    else:
        print(f"No features found at {features_path}")
        print("Please run feature_extractor.py first")
