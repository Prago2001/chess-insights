"""
ChessInsight Configuration
Team 029 - CSE6242 Spring 2026
"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
VIZ_DIR = PROJECT_ROOT / "visualizations"

# Ensure directories exist
for dir_path in [PROCESSED_DATA_DIR, MODELS_DIR, VIZ_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Data settings
MIN_GAMES_PER_PLAYER = 5  # Minimum total games (white + black combined) for player-level analysis
DATA_SOURCE = "Lichess Database (pre-processed parquet chunks)"

# Skill tier definitions (Elo ranges) - 3 tiers for player-level classification
# Data-driven boundaries achieving 65.6% accuracy, 82.8% adjacent accuracy
SKILL_TIERS = {
    "Beginner": (0, 1400),       # Elo < 1400
    "Intermediate": (1400, 1900), # Elo 1400-1900
    "Advanced": (1900, 4000),     # Elo 1900+
}

# Game phase definitions (by move number)
GAME_PHASES = {"opening": (1, 10), "middlegame": (11, 25), "endgame": (26, 200)}

# Time controls to include (in seconds for base time)
VALID_TIME_CONTROLS = {
    "bullet": (60, 120),  # 1-2 minutes
    "blitz": (180, 600),  # 3-10 minutes
    "rapid": (600, 1800),  # 10-30 minutes
}

# Model settings
RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15
N_CLUSTERS_RANGE = (2, 5)  # Range to search for optimal k (K=3 with PCA=2 is optimal)

# ---------------------------------------------------------------------------
# Feature categories (canonical groups used across the project)
# ---------------------------------------------------------------------------
# Time control one-hot features (game-level, not per-color)
TIME_CONTROL_FEATURES = [
    "is_bullet",
    "is_blitz",
    "is_rapid",
]

# Base time-usage features per color
TIME_FEATURES = [
    "avg_time_opening",
    "avg_time_middlegame",
    "avg_time_endgame",
    "time_variance_opening",
    "time_variance_middlegame",
    "time_variance_endgame",
    "low_time_move_ratio",
    "time_trouble_frequency",
    # Engineered time-usage features
    "time_opening_frac",
    "time_middlegame_frac",
    "time_endgame_frac",
    "opening_vs_endgame_time_ratio",
    "time_variance_total",
    "low_time_to_trouble_ratio",
    "time_per_complexity_opening",
    "time_per_complexity_middlegame",
    "time_per_complexity_endgame",
]

# Accuracy/error features - REMOVED FROM ACTIVE USE
# These features require Stockfish engine evaluation data which is not available
# in our dataset. Generating synthetic versions creates circular dependency
# with Elo-based labels. Kept here for reference if real eval data becomes available.
ACCURACY_FEATURES = [
    "blunder_rate",
    "mistake_rate",
    "avg_centipawn_loss",
    "accuracy_percentage",
    "blunders_per_40_moves",
    "mistakes_per_40_moves",
    "blunder_rate_opening",
    "blunder_rate_middlegame",
    "blunder_rate_endgame",
    "accuracy_opening",
    "accuracy_middlegame",
    "accuracy_endgame",
]  # NOT included in ALL_FEATURES

# Complexity features (global)
COMPLEXITY_FEATURES = [
    "avg_position_complexity",
    "material_imbalance_freq",
    "piece_activity_score",
]

# Opening features (global and repertoire-level)
OPENING_FEATURES = [
    "opening_aggression_score",
    "book_deviation_move",
    "num_unique_openings",
    "opening_entropy",
]

# Aggregated list used by some modules when constructing feature matrices
# Note: ACCURACY_FEATURES excluded (no real Stockfish eval data available)
ALL_FEATURES = (
    TIME_CONTROL_FEATURES + TIME_FEATURES + COMPLEXITY_FEATURES + OPENING_FEATURES
)

# Features used for behavioral clustering (K-Means k=3).
# Mean-only style features — no std, no elo, no game_count. Matches the set
# that produced the balanced archetype clusters documented in the final report.
CLUSTERING_FEATURES = [
    "avg_time_opening",
    "avg_time_middlegame",
    "avg_time_endgame",
    "low_time_move_ratio",
    "time_trouble_frequency",
    "avg_position_complexity",
    "material_imbalance_freq",
    "piece_activity_score",
    "opening_aggression_score",
    "book_deviation_move",
    "num_moves",
]
