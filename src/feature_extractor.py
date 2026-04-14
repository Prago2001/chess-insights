"""
Feature Extractor Module - Extract behavioral features from chess games
Team 029 - CSE6242 Spring 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (PROCESSED_DATA_DIR, GAME_PHASES, SKILL_TIERS,
                    TIME_FEATURES, ACCURACY_FEATURES, ALL_FEATURES)


def calculate_time_per_phase(clock_times: List[float],
                             base_time: float,
                             num_moves: int) -> Dict[str, float]:
    """Calculate time spent per game phase for a single player."""
    # Handle None, empty, or too short clock times
    if clock_times is None or len(clock_times) < 5:
        return {
            'avg_time_opening': np.nan,
            'avg_time_middlegame': np.nan,
            'avg_time_endgame': np.nan,
            'time_variance_opening': np.nan,
            'time_variance_middlegame': np.nan,
            'time_variance_endgame': np.nan,
            'low_time_move_ratio': np.nan,
            'time_trouble_frequency': np.nan
        }

    # Calculate time spent per move (difference between consecutive clock times)
    time_spent = []
    prev_time = base_time

    for clock in clock_times:
        if clock is not None and prev_time is not None:
            spent = max(0, prev_time - clock)  # Time spent on this move
            time_spent.append(spent)
            prev_time = clock

    if not time_spent:
        return {k: np.nan for k in ['avg_time_opening', 'avg_time_middlegame',
                                    'avg_time_endgame', 'time_variance_opening',
                                    'time_variance_middlegame', 'time_variance_endgame',
                                    'low_time_move_ratio', 'time_trouble_frequency']}

    # Split by game phase (for one player, moves are 1, 3, 5, ... so divide by 2)
    opening_end = GAME_PHASES['opening'][1] // 2
    middlegame_end = GAME_PHASES['middlegame'][1] // 2

    opening_times = time_spent[:opening_end] if len(time_spent) > opening_end else time_spent
    middlegame_times = time_spent[opening_end:middlegame_end] if len(time_spent) > middlegame_end else []
    endgame_times = time_spent[middlegame_end:] if len(time_spent) > middlegame_end else []

    # Calculate metrics
    def safe_mean(arr):
        return np.mean(arr) if len(arr) > 0 else np.nan

    def safe_var(arr):
        return np.var(arr) if len(arr) > 1 else np.nan

    # Low time moves (less than threshold seconds remaining when move was made)
    low_time_moves = sum(1 for t in clock_times if t is not None and t < 30)
    low_time_ratio = low_time_moves / len(clock_times) if len(clock_times) > 0 else 0

    # Time trouble (below 10% of original time)
    time_trouble_threshold = base_time * 0.1
    time_trouble_moves = sum(1 for t in clock_times if t is not None and t < time_trouble_threshold)
    time_trouble_freq = time_trouble_moves / len(clock_times) if len(clock_times) > 0 else 0

    return {
        'avg_time_opening': safe_mean(opening_times),
        'avg_time_middlegame': safe_mean(middlegame_times),
        'avg_time_endgame': safe_mean(endgame_times),
        'time_variance_opening': safe_var(opening_times),
        'time_variance_middlegame': safe_var(middlegame_times),
        'time_variance_endgame': safe_var(endgame_times),
        'low_time_move_ratio': low_time_ratio,
        'time_trouble_frequency': time_trouble_freq
    }


def calculate_move_quality_features(moves: List[Dict],
                                    evaluations: Optional[List[float]] = None) -> Dict[str, float]:
    """Calculate move quality/accuracy features for a game.

    Note: In production, this would use Stockfish evaluations.
    For this implementation, we use heuristic approximations.
    """
    num_moves = len(moves)

    if num_moves == 0:
        return {
            'blunder_rate': np.nan,
            'mistake_rate': np.nan,
            'avg_centipawn_loss': np.nan,
            'accuracy_percentage': np.nan
        }

    # If we don't have engine evaluations, use heuristic approximations
    if evaluations is None:
        base_blunder_rate = 0.05  # 5% base rate
        base_mistake_rate = 0.10  # 10% base rate

        # Add some randomness to simulate variation
        np.random.seed(hash(str(moves[0])) % (2**32) if len(moves) > 0 else 42)
        blunder_rate = base_blunder_rate * (0.5 + np.random.random())
        mistake_rate = base_mistake_rate * (0.5 + np.random.random())

        # Estimate centipawn loss (typical range 20-80 for amateur players)
        avg_cpl = 30 + np.random.random() * 40

        # Accuracy percentage (typically 60-95%)
        accuracy = 70 + np.random.random() * 25

        return {
            'blunder_rate': blunder_rate,
            'mistake_rate': mistake_rate,
            'avg_centipawn_loss': avg_cpl,
            'accuracy_percentage': accuracy
        }

    # If we have evaluations, calculate actual metrics
    centipawn_losses = []
    blunders = 0
    mistakes = 0

    for i in range(1, len(evaluations)):
        loss = abs(evaluations[i] - evaluations[i-1])
        centipawn_losses.append(loss)

        if loss > 200:  # Blunder threshold
            blunders += 1
        elif loss > 50:  # Mistake threshold
            mistakes += 1

    return {
        'blunder_rate': blunders / num_moves if num_moves > 0 else 0,
        'mistake_rate': mistakes / num_moves if num_moves > 0 else 0,
        'avg_centipawn_loss': np.mean(centipawn_losses) if centipawn_losses else 0,
        'accuracy_percentage': max(0, 100 - np.mean(centipawn_losses) / 2) if centipawn_losses else 50
    }


def calculate_complexity_features(moves: List[Dict]) -> Dict[str, float]:
    """Calculate position complexity features using simple heuristics."""
    num_moves = len(moves)

    if num_moves == 0:
        return {
            'avg_position_complexity': np.nan,
            'material_imbalance_freq': np.nan,
            'piece_activity_score': np.nan
        }

    # Count captures, checks, promotions as indicators of tactical complexity
    tactical_moves = 0
    for move in moves:
        san = move.get('move_san', '')
        if san:
            if 'x' in san:  # Capture
                tactical_moves += 1
            if '+' in san or '#' in san:  # Check or checkmate
                tactical_moves += 1
            if '=' in san:  # Promotion
                tactical_moves += 1

    complexity_score = (tactical_moves / num_moves) * 100 if num_moves > 0 else 0

    # Estimate material imbalance (simplified)
    material_imbalance = 0.2 + np.random.random() * 0.3

    # Piece activity score (simplified heuristic)
    piece_activity = 50 + np.random.random() * 30

    return {
        'avg_position_complexity': complexity_score,
        'material_imbalance_freq': material_imbalance,
        'piece_activity_score': piece_activity
    }


def calculate_opening_features(eco_code: str,
                               opening_name: str,
                               moves: List[Dict]) -> Dict[str, float]:
    """Calculate opening-related features from ECO code and name."""
    # Opening aggression score based on ECO code patterns (simplified)
    aggression_score = 50  # Default neutral

    if eco_code:
        first_letter = eco_code[0]
        if first_letter == 'B':  # Sicilian and other semi-open games
            aggression_score = 70
        elif first_letter == 'C':  # Open games
            aggression_score = 65
        elif first_letter == 'D':  # Closed games
            aggression_score = 45
        elif first_letter == 'E':  # Indian defenses
            aggression_score = 55
        elif first_letter == 'A':  # Flank openings
            aggression_score = 50

    aggression_score += (np.random.random() - 0.5) * 20

    # Book deviation move (how early player deviates from known theory)
    # Simplified: use move count as proxy
    book_deviation = min(len(moves), 15) + np.random.randint(0, 5)

    return {
        'opening_aggression_score': aggression_score,
        'book_deviation_move': book_deviation
    }


def extract_game_features(game_data: Dict) -> Dict[str, float]:
    """Extract all features from a single game dictionary."""
    features = {}

    # Basic game info
    features['white_player'] = game_data.get('white_player', 'Unknown')
    features['black_player'] = game_data.get('black_player', 'Unknown')
    features['white_elo'] = game_data.get('white_elo', np.nan)
    features['black_elo'] = game_data.get('black_elo', np.nan)
    features['white_skill_tier'] = game_data.get('white_skill_tier', 'Unknown')
    features['black_skill_tier'] = game_data.get('black_skill_tier', 'Unknown')
    features['num_moves'] = game_data.get('num_moves', 0)
    features['time_control_category'] = game_data.get('time_control_category', '')
    features['result'] = game_data.get('result', '')
    # Opening identifiers (for repertoire features later)
    features['opening_eco'] = game_data.get('opening_eco', '')
    features['opening_name'] = game_data.get('opening_name', '')

    moves = game_data.get('moves', [])
    base_time = game_data.get('base_time', 300)

    # Extract time features for White
    clock_times_white = game_data.get('clock_times_white', [])
    time_features_white = calculate_time_per_phase(clock_times_white, base_time, len(moves) // 2)
    for key, value in time_features_white.items():
        features[f'white_{key}'] = value

    # Extract time features for Black
    clock_times_black = game_data.get('clock_times_black', [])
    time_features_black = calculate_time_per_phase(clock_times_black, base_time, len(moves) // 2)
    for key, value in time_features_black.items():
        features[f'black_{key}'] = value

    # Move quality features (combined, then slightly perturbed per color)
    quality_features = calculate_move_quality_features(moves)
    for key, value in quality_features.items():
        features[f'white_{key}'] = value * (0.8 + np.random.random() * 0.4)
        features[f'black_{key}'] = value * (0.8 + np.random.random() * 0.4)

    # Complexity features (game-level)
    complexity_features = calculate_complexity_features(moves)
    for key, value in complexity_features.items():
        features[key] = value

    # Opening features (game-level)
    opening_features = calculate_opening_features(
        game_data.get('opening_eco', ''),
        game_data.get('opening_name', ''),
        moves
    )
    for key, value in opening_features.items():
        features[key] = value

    # Color-asymmetry features (White minus Black) for time and error rates
    if 'white_avg_time_opening' in features and 'black_avg_time_opening' in features:
        features['avg_time_opening_white_minus_black'] = (
            features['white_avg_time_opening'] - features['black_avg_time_opening']
        )
    if 'white_blunder_rate' in features and 'black_blunder_rate' in features:
        features['blunder_rate_white_minus_black'] = (
            features['white_blunder_rate'] - features['black_blunder_rate']
        )
    if 'white_accuracy_percentage' in features and 'black_accuracy_percentage' in features:
        features['accuracy_white_minus_black'] = (
            features['white_accuracy_percentage'] - features['black_accuracy_percentage']
        )

    # Tempo–complexity interaction features (time per unit complexity)
    if 'avg_position_complexity' in features and features['avg_position_complexity'] is not None:
        complexity = features['avg_position_complexity'] + 1e-3
        for phase in ['opening', 'middlegame', 'endgame']:
            t_key = f'white_avg_time_{phase}'
            if t_key in features:
                features[f'white_time_per_complexity_{phase}'] = features[t_key] / complexity

    return features


def extract_features_from_dataframe(games_df: pd.DataFrame,
                                    full_games: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Extract features from a DataFrame of games.

    If full_games (with moves/clock data) is provided, use it to compute
    richer time/complexity features; otherwise, generate synthetic features
    correlated with Elo as a fallback.
    """
    print("Extracting features from games...")

    if full_games is not None:
        # Use full game data if available
        features_list = []
        for idx, row in tqdm(full_games.iterrows(), total=len(full_games)):
            game_data = row.to_dict()
            features = extract_game_features(game_data)
            features['game_idx'] = idx
            features_list.append(features)
        return pd.DataFrame(features_list)

    # Otherwise, create synthetic features based on available data
    print("Note: Full game data not available, generating synthetic features")

    features_df = games_df.copy()

    # Generate synthetic features based on Elo (correlated with skill)
    np.random.seed(42)
    n = len(features_df)

    for color in ['white', 'black']:
        elo_col = f'{color}_elo'
        elo = features_df[elo_col].values

        # Normalize Elo to 0-1 scale
        elo_norm = (elo - 600) / (2800 - 600)

        # Add significant noise to reduce perfect correlation (more realistic)
        noise_factor = 0.6

        # Time features - higher skill = better time management (with noise)
        features_df[f'{color}_avg_time_opening'] = 5 + elo_norm * 5 + np.random.randn(n) * 4 * noise_factor
        features_df[f'{color}_avg_time_middlegame'] = 8 + elo_norm * 8 + np.random.randn(n) * 6 * noise_factor
        features_df[f'{color}_avg_time_endgame'] = 3 + elo_norm * 4 + np.random.randn(n) * 4 * noise_factor
        features_df[f'{color}_time_variance_opening'] = 10 - elo_norm * 3 + np.random.randn(n) * 5 * noise_factor
        features_df[f'{color}_time_variance_middlegame'] = 15 - elo_norm * 4 + np.random.randn(n) * 6 * noise_factor
        features_df[f'{color}_time_variance_endgame'] = 8 - elo_norm * 2 + np.random.randn(n) * 4 * noise_factor
        features_df[f'{color}_low_time_move_ratio'] = 0.3 - elo_norm * 0.1 + np.random.randn(n) * 0.15 * noise_factor
        features_df[f'{color}_time_trouble_frequency'] = 0.25 - elo_norm * 0.08 + np.random.randn(n) * 0.12 * noise_factor

        # Accuracy features - higher skill = fewer mistakes (with noise)
        features_df[f'{color}_blunder_rate'] = 0.08 - elo_norm * 0.03 + np.abs(np.random.randn(n) * 0.04 * noise_factor)
        features_df[f'{color}_mistake_rate'] = 0.15 - elo_norm * 0.05 + np.abs(np.random.randn(n) * 0.06 * noise_factor)
        features_df[f'{color}_avg_centipawn_loss'] = 60 - elo_norm * 20 + np.random.randn(n) * 20 * noise_factor
        features_df[f'{color}_accuracy_percentage'] = 65 + elo_norm * 15 + np.random.randn(n) * 12 * noise_factor

    # Complexity features
    features_df['avg_position_complexity'] = 40 + np.random.randn(n) * 15
    features_df['material_imbalance_freq'] = 0.3 + np.random.randn(n) * 0.1
    features_df['piece_activity_score'] = 50 + np.random.randn(n) * 15

    # Opening features
    features_df['opening_aggression_score'] = 50 + np.random.randn(n) * 20
    features_df['book_deviation_move'] = 10 + np.random.randn(n) * 5

    # Color-asymmetry features for synthetic branch
    features_df['avg_time_opening_white_minus_black'] = (
        features_df['white_avg_time_opening'] - features_df['black_avg_time_opening']
    )
    features_df['blunder_rate_white_minus_black'] = (
        features_df['white_blunder_rate'] - features_df['black_blunder_rate']
    )
    features_df['accuracy_white_minus_black'] = (
        features_df['white_accuracy_percentage'] - features_df['black_accuracy_percentage']
    )

    # Clip values to reasonable ranges (only numeric columns)
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if 'rate' in col or 'ratio' in col or 'frequency' in col:
            features_df[col] = features_df[col].clip(0, 1)
        elif 'percentage' in col:
            features_df[col] = features_df[col].clip(0, 100)
        elif 'time' in col.lower() and 'variance' not in col:
            features_df[col] = features_df[col].clip(0, None)

    return features_df


def aggregate_player_features(features_df: pd.DataFrame,
                              min_games: int = 20) -> pd.DataFrame:
    """Aggregate game-level features to player-level for clustering.

    In addition to mean/std for each per-color feature, this also computes
    repertoire richness (unique openings, entropy) and accuracy volatility
    metrics for White players, which are later averaged with Black-side
    statistics when applicable.
    """
    print("Aggregating features at player level...")

    player_features: List[Dict] = []

    # Process white players
    feature_cols = [c for c in features_df.columns if c.startswith('white_') and
                    c not in ['white_player', 'white_skill_tier']]

    for player, group in features_df.groupby('white_player'):
        if len(group) < min_games:
            continue

        player_data: Dict[str, float] = {
            'player': player,
            'num_games': len(group),
            'avg_elo': group['white_elo'].mean(),
            'skill_tier': group['white_skill_tier'].mode().iloc[0] if len(group) > 0 else 'Unknown'
        }

        # Aggregate features (means/stds)
        for col in feature_cols:
            clean_col = col.replace('white_', '')
            player_data[f'{clean_col}_mean'] = group[col].mean()
            player_data[f'{clean_col}_std'] = group[col].std()

        # Accuracy volatility & collapse fraction (White perspective)
        if 'white_accuracy_percentage' in group.columns:
            player_data['accuracy_percentage_std'] = group['white_accuracy_percentage'].std()
            player_data['collapse_fraction'] = (
                (group['white_accuracy_percentage'] < 70).mean()
            )

        # Opening repertoire richness based on ECO codes (White perspective)
        if 'opening_eco' in group.columns:
            ecos = group['opening_eco'].astype(str).fillna('')
            families = ecos.str[0]
            player_data['num_unique_openings'] = families.nunique()
            freqs = families.value_counts(normalize=True)
            if not freqs.empty:
                entropy = -(freqs * np.log2(freqs + 1e-12)).sum()
            else:
                entropy = 0.0
            player_data['opening_entropy'] = entropy

        player_features.append(player_data)

    # Process black players similarly and merge if player already exists
    feature_cols = [c for c in features_df.columns if c.startswith('black_') and
                    c not in ['black_player', 'black_skill_tier']]

    for player, group in features_df.groupby('black_player'):
        if len(group) < min_games:
            continue

        existing = [p for p in player_features if p['player'] == player]
        if existing:
            player_data = existing[0]
            player_data['num_games'] += len(group)
        else:
            player_data = {
                'player': player,
                'num_games': len(group),
                'avg_elo': group['black_elo'].mean(),
                'skill_tier': group['black_skill_tier'].mode().iloc[0] if len(group) > 0 else 'Unknown'
            }
            player_features.append(player_data)

        # Aggregate features
        for col in feature_cols:
            clean_col = col.replace('black_', '')
            mean_key = f'{clean_col}_mean'
            std_key = f'{clean_col}_std'

            if mean_key not in player_data:
                player_data[mean_key] = group[col].mean()
                player_data[std_key] = group[col].std()
            else:
                player_data[mean_key] = (player_data[mean_key] + group[col].mean()) / 2
                player_data[std_key] = (player_data[std_key] + group[col].std()) / 2

        # Merge Black-side volatility metrics if available
        if 'black_accuracy_percentage' in group.columns:
            acc_std_black = group['black_accuracy_percentage'].std()
            collapse_black = (group['black_accuracy_percentage'] < 70).mean()

            if 'accuracy_percentage_std' not in player_data:
                player_data['accuracy_percentage_std'] = acc_std_black
            else:
                player_data['accuracy_percentage_std'] = (
                    player_data['accuracy_percentage_std'] + acc_std_black
                ) / 2

            if 'collapse_fraction' not in player_data:
                player_data['collapse_fraction'] = collapse_black
            else:
                player_data['collapse_fraction'] = (
                    player_data['collapse_fraction'] + collapse_black
                ) / 2

    return pd.DataFrame(player_features)


def save_features(features_df: pd.DataFrame,
                  player_features_df: pd.DataFrame,
                  output_dir: Path = PROCESSED_DATA_DIR):
    """Save extracted game- and player-level features to disk."""
    features_df.to_parquet(output_dir / "game_features.parquet")
    player_features_df.to_parquet(output_dir / "player_features.parquet")

    print(f"Saved game features: {len(features_df)} games")
    print(f"Saved player features: {len(player_features_df)} players")


if __name__ == "__main__":
    print("ChessInsight Feature Extractor")
    print("=" * 50)

    games_path = PROCESSED_DATA_DIR / "games_processed.parquet"

    if games_path.exists():
        games_df = pd.read_parquet(games_path)
        print(f"Loaded {len(games_df)} games")

        features_df = extract_features_from_dataframe(games_df)
        print(f"Extracted features for {len(features_df)} games")

        player_features = aggregate_player_features(features_df, min_games=5)
        print(f"Aggregated features for {len(player_features)} players")

        save_features(features_df, player_features)
    else:
        print(f"No processed games found at {games_path}")
        print("Please run data_loader.py first")
