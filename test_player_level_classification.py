"""
Test: Player-Level vs Game-Level Classification with 3 Tiers
Team 029 - CSE6242 Spring 2026

This script compares:
1. Game-level classification (current approach) - 328K samples
2. Player-level classification (aggregated) - 22K samples

Using 3 skill tiers:
- Beginner: Elo < 1400
- Intermediate: Elo 1400-1900
- Advanced: Elo 1900+
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Config
PROCESSED_DATA_DIR = Path(__file__).parent / "data" / "processed"
RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# 3-tier skill definitions
SKILL_TIERS_3 = {
    "Beginner": (0, 1400),
    "Intermediate": (1400, 1900),
    "Advanced": (1900, 4000),
}


def assign_skill_tier_3(elo: float) -> str:
    """Assign player to one of 3 skill tiers based on Elo."""
    for tier, (low, high) in SKILL_TIERS_3.items():
        if low <= elo < high:
            return tier
    return "Advanced"  # Default for very high Elo


def get_feature_columns(df: pd.DataFrame, color: str) -> list:
    """Get feature columns for a given color."""
    prefix = f"{color}_"
    feature_cols = []

    # Per-color features
    for c in df.columns:
        if c.startswith(prefix) and c not in [
            f"{prefix}player", f"{prefix}skill_tier", f"{prefix}elo"
        ]:
            feature_cols.append(c)

    # Global features
    global_features = [
        "avg_position_complexity",
        "material_imbalance_freq",
        "piece_activity_score",
        "opening_aggression_score",
        "book_deviation_move",
        "num_moves",
        "is_bullet",
        "is_blitz",
        "is_rapid",
    ]
    for col in global_features:
        if col in df.columns:
            feature_cols.append(col)

    return feature_cols


def create_model(model_type: str):
    """Create a model based on type."""
    if model_type == "rf":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=18,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    elif model_type == "xgboost":
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
    elif model_type == "ensemble":
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=18,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        xgb = XGBClassifier(
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
        return VotingClassifier(
            estimators=[("rf", rf), ("xgb", xgb)],
            voting="soft"
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_and_evaluate(X, y, description: str, model_type: str = "rf"):
    """Train a classifier and return metrics."""
    print(f"\n{'='*60}")
    print(f"Training: {description}")
    print(f"Model: {model_type.upper()}")
    print(f"{'='*60}")
    print(f"Samples: {len(X)}, Features: {X.shape[1]}")
    print(f"Class distribution:\n{y.value_counts()}")

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_encoded, test_size=(TEST_SIZE + VAL_SIZE),
        random_state=RANDOM_STATE, stratify=y_encoded
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5,
        random_state=RANDOM_STATE, stratify=y_temp
    )

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Apply SMOTE
    try:
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
        print(f"After SMOTE: {len(X_train_balanced)} training samples")
    except ValueError:
        X_train_balanced, y_train_balanced = X_train_scaled, y_train

    # Train model
    model = create_model(model_type)
    model.fit(X_train_balanced, y_train_balanced)

    # Evaluate
    y_test_pred = model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    # Adjacent accuracy
    adjacent_correct = sum(abs(p - t) <= 1 for p, t in zip(y_test_pred, y_test))
    adjacent_accuracy = adjacent_correct / len(y_test)

    print(f"\nResults:")
    print(f"  Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"  Adjacent Accuracy: {adjacent_accuracy:.4f} ({adjacent_accuracy*100:.2f}%)")

    return {
        "description": description,
        "model_type": model_type,
        "samples": len(X),
        "test_accuracy": test_accuracy,
        "adjacent_accuracy": adjacent_accuracy,
    }


def main():
    print("=" * 70)
    print("Player-Level vs Game-Level Classification Comparison")
    print("Using 3 Skill Tiers: Beginner (<1400), Intermediate (1400-1900), Advanced (1900+)")
    print("=" * 70)

    # Load data
    features_path = PROCESSED_DATA_DIR / "game_features.parquet"
    if not features_path.exists():
        print(f"Error: {features_path} not found")
        return

    df = pd.read_parquet(features_path)
    print(f"\nLoaded {len(df)} games")

    # Get feature columns (using white as reference)
    feature_cols = get_feature_columns(df, "white")
    print(f"Using {len(feature_cols)} features")

    results = []

    # =========================================================================
    # Test 1: Game-level classification (3 tiers, White only)
    # =========================================================================
    df_test1 = df.copy()
    df_test1["skill_tier_3"] = df_test1["white_elo"].apply(assign_skill_tier_3)

    X1 = df_test1[feature_cols].select_dtypes(include=[np.number])
    y1 = df_test1["skill_tier_3"]
    mask1 = X1.notna().all(axis=1) & y1.notna()
    X1, y1 = X1[mask1], y1[mask1]

    results.append(train_and_evaluate(X1, y1, "Game-level, 3-tier, White only"))

    # =========================================================================
    # Test 2: Game-level classification (3 tiers, Both players)
    # =========================================================================
    # Prepare white player data
    white_features = get_feature_columns(df, "white")
    df_white = df[white_features + ["white_elo"]].copy()
    df_white.columns = [c.replace("white_", "") if c.startswith("white_") else c for c in df_white.columns]
    df_white = df_white.rename(columns={"elo": "player_elo"})
    df_white["player_elo"] = df["white_elo"]

    # Prepare black player data
    black_features = get_feature_columns(df, "black")
    # We need to also get black_ prefixed columns
    black_feature_cols = []
    for c in df.columns:
        if c.startswith("black_") and c not in ["black_player", "black_skill_tier", "black_elo"]:
            black_feature_cols.append(c)
    for col in ["avg_position_complexity", "material_imbalance_freq", "piece_activity_score",
                "opening_aggression_score", "book_deviation_move", "num_moves",
                "is_bullet", "is_blitz", "is_rapid"]:
        if col in df.columns:
            black_feature_cols.append(col)

    df_black = df[black_feature_cols + ["black_elo"]].copy()
    df_black.columns = [c.replace("black_", "") if c.startswith("black_") else c for c in df_black.columns]
    df_black = df_black.rename(columns={"elo": "player_elo"})
    df_black["player_elo"] = df["black_elo"]

    # Combine
    df_both = pd.concat([df_white, df_black], ignore_index=True)
    df_both["skill_tier_3"] = df_both["player_elo"].apply(assign_skill_tier_3)

    # Get standardized feature columns (without color prefix)
    std_features = [c.replace("white_", "") for c in white_features if c.startswith("white_")]
    std_features += [c for c in white_features if not c.startswith("white_")]

    X2 = df_both[std_features].select_dtypes(include=[np.number])
    y2 = df_both["skill_tier_3"]
    mask2 = X2.notna().all(axis=1) & y2.notna()
    X2, y2 = X2[mask2], y2[mask2]

    results.append(train_and_evaluate(X2, y2, "Game-level, 3-tier, Both players"))

    # =========================================================================
    # Test 3: Player-level classification (3 tiers) - All Models
    # =========================================================================
    print("\n" + "=" * 60)
    print("Aggregating features to player level...")
    print("=" * 60)

    # Combine white and black data with player identifiers
    df_white_player = df[["white_player", "white_elo"] + white_features].copy()
    df_white_player = df_white_player.rename(columns={"white_player": "player", "white_elo": "elo"})
    df_white_player.columns = [c.replace("white_", "") if c.startswith("white_") else c for c in df_white_player.columns]

    df_black_player = df[["black_player", "black_elo"] + black_feature_cols].copy()
    df_black_player = df_black_player.rename(columns={"black_player": "player", "black_elo": "elo"})
    df_black_player.columns = [c.replace("black_", "") if c.startswith("black_") else c for c in df_black_player.columns]

    # Combine all games for all players
    df_all_games = pd.concat([df_white_player, df_black_player], ignore_index=True)
    print(f"Total player-game records: {len(df_all_games)}")

    # Get numeric feature columns for aggregation
    numeric_cols = df_all_games.select_dtypes(include=[np.number]).columns.tolist()
    if "elo" in numeric_cols:
        numeric_cols.remove("elo")

    # Aggregate by player: mean of features, mean of Elo
    player_agg = df_all_games.groupby("player").agg({
        **{col: "mean" for col in numeric_cols},
        "elo": "mean"  # Average Elo across games
    }).reset_index()

    # Count games per player
    games_per_player = df_all_games.groupby("player").size().reset_index(name="game_count")
    player_agg = player_agg.merge(games_per_player, on="player")

    # Filter to players with at least 5 games (for reliable aggregation)
    min_games = 5
    player_agg = player_agg[player_agg["game_count"] >= min_games]
    print(f"Players with {min_games}+ games: {len(player_agg)}")

    # Assign skill tier
    player_agg["skill_tier_3"] = player_agg["elo"].apply(assign_skill_tier_3)

    # Prepare features
    X3 = player_agg[numeric_cols].select_dtypes(include=[np.number])
    y3 = player_agg["skill_tier_3"]
    mask3 = X3.notna().all(axis=1) & y3.notna()
    X3, y3 = X3[mask3], y3[mask3]

    # Test all three models on player-level data
    results.append(train_and_evaluate(X3, y3, "Player-level, 3-tier, RF", model_type="rf"))
    results.append(train_and_evaluate(X3, y3, "Player-level, 3-tier, XGBoost", model_type="xgboost"))
    results.append(train_and_evaluate(X3, y3, "Player-level, 3-tier, Ensemble", model_type="ensemble"))

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(f"\n{'Approach':<50} {'Samples':>10} {'Accuracy':>12} {'Adjacent':>12}")
    print("-" * 84)
    for r in results:
        print(f"{r['description']:<50} {r['samples']:>10,} {r['test_accuracy']*100:>11.2f}% {r['adjacent_accuracy']*100:>11.2f}%")

    # Best result
    best = max(results, key=lambda x: x["test_accuracy"])
    print(f"\nBest Approach: {best['description']}")
    print(f"  Accuracy: {best['test_accuracy']*100:.2f}%")
    print(f"  Adjacent Accuracy: {best['adjacent_accuracy']*100:.2f}%")


if __name__ == "__main__":
    main()
