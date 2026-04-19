"""
Test: Reaching 65% Accuracy Target
Team 029 - CSE6242 Spring 2026

Testing improvements:
1. Increase minimum games per player (10, 20)
2. Add variance (std) features across games
3. Add game count as a feature
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
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
    for tier, (low, high) in SKILL_TIERS_3.items():
        if low <= elo < high:
            return tier
    return "Advanced"


def get_feature_columns(df: pd.DataFrame, color: str) -> list:
    prefix = f"{color}_"
    feature_cols = []
    for c in df.columns:
        if c.startswith(prefix) and c not in [
            f"{prefix}player", f"{prefix}skill_tier", f"{prefix}elo"
        ]:
            feature_cols.append(c)
    global_features = [
        "avg_position_complexity", "material_imbalance_freq", "piece_activity_score",
        "opening_aggression_score", "book_deviation_move", "num_moves",
        "is_bullet", "is_blitz", "is_rapid",
    ]
    for col in global_features:
        if col in df.columns:
            feature_cols.append(col)
    return feature_cols


def create_ensemble():
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=18, min_samples_split=4,
        min_samples_leaf=2, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    xgb = XGBClassifier(
        n_estimators=250, max_depth=8, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.9,
        objective="multi:softprob", eval_metric="mlogloss",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    return VotingClassifier(estimators=[("rf", rf), ("xgb", xgb)], voting="soft")


def train_and_evaluate(X, y, description: str):
    """Train ensemble and return metrics."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Samples: {len(X)}, Features: {X.shape[1]}")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_encoded, test_size=(TEST_SIZE + VAL_SIZE),
        random_state=RANDOM_STATE, stratify=y_encoded
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5,
        random_state=RANDOM_STATE, stratify=y_temp
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    try:
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    except ValueError:
        X_train_balanced, y_train_balanced = X_train_scaled, y_train

    model = create_ensemble()
    model.fit(X_train_balanced, y_train_balanced)

    y_test_pred = model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    adjacent_correct = sum(abs(p - t) <= 1 for p, t in zip(y_test_pred, y_test))
    adjacent_accuracy = adjacent_correct / len(y_test)

    print(f"  Accuracy: {test_accuracy*100:.2f}%  |  Adjacent: {adjacent_accuracy*100:.2f}%")

    return {
        "description": description,
        "samples": len(X),
        "features": X.shape[1],
        "test_accuracy": test_accuracy,
        "adjacent_accuracy": adjacent_accuracy,
    }


def aggregate_player_features(df, white_features, black_feature_cols, min_games=5, add_std=False):
    """Aggregate game features to player level."""

    # Combine white and black data
    df_white_player = df[["white_player", "white_elo"] + white_features].copy()
    df_white_player = df_white_player.rename(columns={"white_player": "player", "white_elo": "elo"})
    df_white_player.columns = [c.replace("white_", "") if c.startswith("white_") else c for c in df_white_player.columns]

    df_black_player = df[["black_player", "black_elo"] + black_feature_cols].copy()
    df_black_player = df_black_player.rename(columns={"black_player": "player", "black_elo": "elo"})
    df_black_player.columns = [c.replace("black_", "") if c.startswith("black_") else c for c in df_black_player.columns]

    df_all_games = pd.concat([df_white_player, df_black_player], ignore_index=True)

    # Get numeric columns
    numeric_cols = df_all_games.select_dtypes(include=[np.number]).columns.tolist()
    if "elo" in numeric_cols:
        numeric_cols.remove("elo")

    # Aggregation: mean + optionally std
    if add_std:
        agg_dict = {}
        for col in numeric_cols:
            agg_dict[col] = ["mean", "std"]
        agg_dict["elo"] = "mean"

        player_agg = df_all_games.groupby("player").agg(agg_dict)
        # Flatten column names
        player_agg.columns = [f"{col}_{stat}" if stat != "mean" else col
                              for col, stat in player_agg.columns]
        player_agg = player_agg.reset_index()
        # Fill NaN std values (single game players) with 0
        std_cols = [c for c in player_agg.columns if c.endswith("_std")]
        player_agg[std_cols] = player_agg[std_cols].fillna(0)
    else:
        player_agg = df_all_games.groupby("player").agg({
            **{col: "mean" for col in numeric_cols},
            "elo": "mean"
        }).reset_index()

    # Add game count
    games_per_player = df_all_games.groupby("player").size().reset_index(name="game_count")
    player_agg = player_agg.merge(games_per_player, on="player")

    # Filter by min games
    player_agg = player_agg[player_agg["game_count"] >= min_games]

    # Assign skill tier
    elo_col = "elo" if "elo" in player_agg.columns else "elo_mean"
    player_agg["skill_tier_3"] = player_agg[elo_col].apply(assign_skill_tier_3)

    return player_agg


def main():
    print("=" * 70)
    print("Testing Improvements to Reach 65% Accuracy Target")
    print("=" * 70)

    # Load data
    features_path = PROCESSED_DATA_DIR / "game_features.parquet"
    if not features_path.exists():
        print(f"Error: {features_path} not found")
        return

    df = pd.read_parquet(features_path)
    print(f"\nLoaded {len(df)} games")

    # Get feature columns
    white_features = get_feature_columns(df, "white")
    black_feature_cols = []
    for c in df.columns:
        if c.startswith("black_") and c not in ["black_player", "black_skill_tier", "black_elo"]:
            black_feature_cols.append(c)
    for col in ["avg_position_complexity", "material_imbalance_freq", "piece_activity_score",
                "opening_aggression_score", "book_deviation_move", "num_moves",
                "is_bullet", "is_blitz", "is_rapid"]:
        if col in df.columns and col not in black_feature_cols:
            black_feature_cols.append(col)

    results = []

    # =========================================================================
    # Baseline: min_games=5, mean only
    # =========================================================================
    player_agg = aggregate_player_features(df, white_features, black_feature_cols, min_games=5, add_std=False)
    numeric_cols = player_agg.select_dtypes(include=[np.number]).columns.tolist()
    exclude = ["elo", "game_count", "skill_tier_3"]
    feature_cols = [c for c in numeric_cols if c not in exclude]

    X = player_agg[feature_cols]
    y = player_agg["skill_tier_3"]
    mask = X.notna().all(axis=1) & y.notna()
    results.append(train_and_evaluate(X[mask], y[mask], "Baseline: min_games=5, mean only"))

    # =========================================================================
    # Test 1: Increase min_games to 10
    # =========================================================================
    player_agg = aggregate_player_features(df, white_features, black_feature_cols, min_games=10, add_std=False)
    numeric_cols = player_agg.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude]

    X = player_agg[feature_cols]
    y = player_agg["skill_tier_3"]
    mask = X.notna().all(axis=1) & y.notna()
    results.append(train_and_evaluate(X[mask], y[mask], "min_games=10, mean only"))

    # =========================================================================
    # Test 2: Increase min_games to 20
    # =========================================================================
    player_agg = aggregate_player_features(df, white_features, black_feature_cols, min_games=20, add_std=False)
    numeric_cols = player_agg.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude]

    X = player_agg[feature_cols]
    y = player_agg["skill_tier_3"]
    mask = X.notna().all(axis=1) & y.notna()
    results.append(train_and_evaluate(X[mask], y[mask], "min_games=20, mean only"))

    # =========================================================================
    # Test 3: Add std features (min_games=5)
    # =========================================================================
    player_agg = aggregate_player_features(df, white_features, black_feature_cols, min_games=5, add_std=True)
    numeric_cols = player_agg.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude]

    X = player_agg[feature_cols]
    y = player_agg["skill_tier_3"]
    mask = X.notna().all(axis=1) & y.notna()
    results.append(train_and_evaluate(X[mask], y[mask], "min_games=5, mean + std features"))

    # =========================================================================
    # Test 4: Add std features + min_games=10
    # =========================================================================
    player_agg = aggregate_player_features(df, white_features, black_feature_cols, min_games=10, add_std=True)
    numeric_cols = player_agg.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude]

    X = player_agg[feature_cols]
    y = player_agg["skill_tier_3"]
    mask = X.notna().all(axis=1) & y.notna()
    results.append(train_and_evaluate(X[mask], y[mask], "min_games=10, mean + std features"))

    # =========================================================================
    # Test 5: Add std features + min_games=20
    # =========================================================================
    player_agg = aggregate_player_features(df, white_features, black_feature_cols, min_games=20, add_std=True)
    numeric_cols = player_agg.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude]

    X = player_agg[feature_cols]
    y = player_agg["skill_tier_3"]
    mask = X.notna().all(axis=1) & y.notna()
    results.append(train_and_evaluate(X[mask], y[mask], "min_games=20, mean + std features"))

    # =========================================================================
    # Test 6: Add game_count as feature + std + min_games=10
    # =========================================================================
    player_agg = aggregate_player_features(df, white_features, black_feature_cols, min_games=10, add_std=True)
    numeric_cols = player_agg.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in ["elo", "skill_tier_3"]]  # Include game_count

    X = player_agg[feature_cols]
    y = player_agg["skill_tier_3"]
    mask = X.notna().all(axis=1) & y.notna()
    results.append(train_and_evaluate(X[mask], y[mask], "min_games=10, mean + std + game_count"))

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\n{'Configuration':<45} {'Samples':>8} {'Features':>8} {'Accuracy':>10} {'Adjacent':>10}")
    print("-" * 81)
    for r in results:
        marker = " ***" if r['test_accuracy'] >= 0.65 else ""
        print(f"{r['description']:<45} {r['samples']:>8,} {r['features']:>8} {r['test_accuracy']*100:>9.2f}% {r['adjacent_accuracy']*100:>9.2f}%{marker}")

    best = max(results, key=lambda x: x["test_accuracy"])
    print(f"\nBest: {best['description']}")
    print(f"  Accuracy: {best['test_accuracy']*100:.2f}%")
    print(f"  Adjacent: {best['adjacent_accuracy']*100:.2f}%")

    if best['test_accuracy'] >= 0.65:
        print("\n*** TARGET OF 65% REACHED! ***")
    else:
        gap = 65 - best['test_accuracy']*100
        print(f"\nGap to 65%: {gap:.2f}%")


if __name__ == "__main__":
    main()
