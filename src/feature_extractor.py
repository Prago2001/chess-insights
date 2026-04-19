"""
Feature Extractor Module - Extract behavioral features from chess games
Team 029 - CSE6242 Spring 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import PROCESSED_DATA_DIR, GAME_PHASES

# Import python-chess for real feature computation
try:
    import chess
    CHESS_AVAILABLE = True
except ImportError:
    CHESS_AVAILABLE = False
    print("Warning: python-chess not installed. Some features will use fallback values.")


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

    # Low time moves (less than 30 seconds remaining when move was made)
    low_time_moves = sum(1 for t in clock_times if t is not None and t < 30)
    low_time_ratio = low_time_moves / len(clock_times) if len(clock_times) > 0 else 0

    # Time trouble (last 10% of original time)
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

    IMPORTANT: These features require Stockfish engine evaluations to compute.
    If evaluations are not available, returns empty dict (no synthetic data).

    Rationale for not generating synthetic accuracy features:
    1. Synthetic features would be derived from Elo (circular dependency with label)
    2. They would not represent actual player behavior
    3. Testing showed they contribute only 11.6% importance and removing them
       changes accuracy by just 0.13%
    4. Including fake data misrepresents what the model actually learns

    Args:
        moves: List of move dictionaries
        evaluations: Optional list of centipawn evaluations from Stockfish

    Returns:
        Dictionary with accuracy features if evaluations available, else empty dict
    """
    num_moves = len(moves)

    # If we don't have engine evaluations, return empty dict (no synthetic data)
    if evaluations is None or len(evaluations) == 0:
        return {}

    # If we have evaluations, calculate actual metrics
    centipawn_losses = []
    blunders = 0
    mistakes = 0

    for i in range(1, len(evaluations)):
        loss = abs(evaluations[i] - evaluations[i-1])
        centipawn_losses.append(loss)

        if loss > 200:  # Blunder threshold: >2 pawns lost
            blunders += 1
        elif loss > 50:  # Mistake threshold: >0.5 pawns lost
            mistakes += 1

    return {
        'blunder_rate': blunders / num_moves if num_moves > 0 else 0,
        'mistake_rate': mistakes / num_moves if num_moves > 0 else 0,
        'avg_centipawn_loss': np.mean(centipawn_losses) if centipawn_losses else 0,
        'accuracy_percentage': max(0, 100 - np.mean(centipawn_losses) / 2) if centipawn_losses else 50
    }


def calculate_complexity_features(moves: List[Dict]) -> Dict[str, float]:
    """Calculate position complexity features using real game data.

    Features computed:
    - avg_position_complexity: Fraction of tactical moves (captures, checks, promotions)
    - material_imbalance_freq: Fraction of positions with unequal material (using python-chess)
    - piece_activity_score: Average number of legal moves per position (using python-chess)
    """
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

    # Compute real material imbalance and piece activity using python-chess
    if CHESS_AVAILABLE:
        board = chess.Board()
        imbalanced_positions = 0
        total_positions = 0
        activity_scores = []

        # Piece values for material calculation
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9
        }

        for move_dict in moves:
            try:
                move_san = move_dict.get('move_san', '')
                if not move_san:
                    continue

                move = board.parse_san(move_san)
                board.push(move)

                # Calculate material for each side
                white_material = sum(
                    len(board.pieces(pt, chess.WHITE)) * val
                    for pt, val in piece_values.items()
                )
                black_material = sum(
                    len(board.pieces(pt, chess.BLACK)) * val
                    for pt, val in piece_values.items()
                )

                # Check for material imbalance (any difference > 0)
                if abs(white_material - black_material) > 0:
                    imbalanced_positions += 1
                total_positions += 1

                # Count legal moves as piece activity metric
                num_legal_moves = len(list(board.legal_moves))
                activity_scores.append(num_legal_moves)

            except Exception:
                # Skip invalid moves
                continue

        material_imbalance = imbalanced_positions / total_positions if total_positions > 0 else 0.0
        piece_activity = np.mean(activity_scores) if activity_scores else 30.0
    else:
        # Fallback: use move-based heuristics (no random values)
        # Count captures as proxy for material changes
        captures = sum(1 for m in moves if 'x' in m.get('move_san', ''))
        material_imbalance = captures / num_moves if num_moves > 0 else 0.0
        # Estimate activity based on game length (longer games = more complex)
        piece_activity = min(40, 20 + num_moves * 0.3)

    return {
        'avg_position_complexity': complexity_score,
        'material_imbalance_freq': material_imbalance,
        'piece_activity_score': piece_activity
    }


def calculate_opening_features(eco_code: str,
                               opening_name: str,
                               moves: List[Dict]) -> Dict[str, float]:
    """Calculate opening-related features from ECO code and name.

    Features computed (NO random noise - deterministic based on game data):
    - opening_aggression_score: Based on ECO code + early pawn advances/piece development
    - book_deviation_move: Estimated based on opening name length and ECO specificity
    """
    num_moves = len(moves)

    # Opening aggression score based on ECO code patterns
    # Base score from ECO category (well-established chess theory)
    aggression_score = 50  # Default neutral

    if eco_code:
        first_letter = eco_code[0] if eco_code else ''
        if first_letter == 'B':  # Sicilian and other semi-open games (aggressive)
            aggression_score = 70
        elif first_letter == 'C':  # Open games (e4 e5, typically tactical)
            aggression_score = 65
        elif first_letter == 'D':  # Closed games (d4 d5, typically positional)
            aggression_score = 45
        elif first_letter == 'E':  # Indian defenses (flexible, semi-aggressive)
            aggression_score = 55
        elif first_letter == 'A':  # Flank openings (English, Reti - positional)
            aggression_score = 50

        # Refine based on ECO sub-code (second character indicates variation aggressiveness)
        if len(eco_code) >= 2:
            try:
                sub_code = int(eco_code[1:3]) if len(eco_code) >= 3 else int(eco_code[1])
                # Higher sub-codes often indicate sharper/more theoretical lines
                aggression_score += (sub_code - 50) * 0.1  # Small adjustment
            except ValueError:
                pass

    # Adjust aggression based on early game moves (first 10 moves)
    early_moves = moves[:min(10, num_moves)]
    early_captures = sum(1 for m in early_moves if 'x' in m.get('move_san', ''))
    early_checks = sum(1 for m in early_moves if '+' in m.get('move_san', ''))
    aggression_score += early_captures * 2 + early_checks * 3  # Bonus for early tactics

    # Clamp to reasonable range
    aggression_score = max(20, min(100, aggression_score))

    # Book deviation move: estimate based on opening specificity
    # More specific opening names (longer) = deeper theory followed
    # ECO codes with more digits = more specific variation
    if opening_name:
        # Longer opening names suggest more specific/deeper theory
        name_words = len(opening_name.split())
        book_deviation = min(20, 5 + name_words)  # Range: 5-20
    else:
        book_deviation = 8  # Default if no opening name

    # Adjust based on ECO code specificity
    if eco_code and len(eco_code) >= 3:
        # More specific ECO codes (e.g., B90 vs B9) suggest deeper book
        try:
            eco_num = int(eco_code[1:])
            # Specific variations (higher numbers in some categories) = deeper theory
            book_deviation = min(20, book_deviation + eco_num // 20)
        except ValueError:
            pass

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

    # One-hot encode time control type (fixes time variance confounding issue)
    time_control = game_data.get('time_control_category', '').lower()
    features['is_bullet'] = 1 if time_control == 'bullet' else 0
    features['is_blitz'] = 1 if time_control == 'blitz' else 0
    features['is_rapid'] = 1 if time_control == 'rapid' else 0
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

    # Move quality features (only if engine evaluations are available)
    # Note: Without Stockfish evals, this returns empty dict (no synthetic data)
    quality_features = calculate_move_quality_features(moves)
    for key, value in quality_features.items():
        features[f'white_{key}'] = value
        features[f'black_{key}'] = value

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

    Requires full_games DataFrame with moves and clock data to compute
    real behavioral features. Synthetic feature generation has been removed
    to ensure scientific validity.
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

    # DEPRECATED: Synthetic feature generation
    # This branch should not be used - always provide full_games with real clock data
    raise ValueError(
        "Full game data with clock times is required for feature extraction. "
        "Synthetic feature generation has been removed to ensure scientific validity. "
        "Please load games from data/processed/chunks/games_full_part_*.parquet"
    )


def aggregate_player_features(features_df: pd.DataFrame,
                              min_games: int = 5) -> pd.DataFrame:
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

    # Try to load full games data with clock times from chunks
    chunks_dir = PROCESSED_DATA_DIR / "chunks"
    full_game_chunks = sorted(chunks_dir.glob("games_full_part_*.parquet"))

    if full_game_chunks:
        print(f"Loading full game data from {len(full_game_chunks)} chunks...")
        # Load all chunks (filter doesn't work on list columns)
        full_games_df = pd.concat([pd.read_parquet(chunk) for chunk in tqdm(full_game_chunks, desc="Loading chunks")])
        # Filter to games with clock data
        has_clock = full_games_df['clock_times_white'].apply(lambda x: x is not None and len(x) > 0 if hasattr(x, '__len__') else False)
        full_games_df = full_games_df[has_clock].reset_index(drop=True)
        print(f"Loaded {len(full_games_df)} games with clock data")

        # Extract features using real clock data
        features_df = extract_features_from_dataframe(None, full_games=full_games_df)
        print(f"Extracted features for {len(features_df)} games")

        player_features = aggregate_player_features(features_df, min_games=5)
        print(f"Aggregated features for {len(player_features)} players")

        save_features(features_df, player_features)
    else:
        # Fallback to processed games without clock data
        games_path = PROCESSED_DATA_DIR / "games_processed.parquet"
        if games_path.exists():
            games_df = pd.read_parquet(games_path)
            print(f"Loaded {len(games_df)} games (no clock data)")

            features_df = extract_features_from_dataframe(games_df)
            print(f"Extracted features for {len(features_df)} games")

            player_features = aggregate_player_features(features_df, min_games=5)
            print(f"Aggregated features for {len(player_features)} players")

            save_features(features_df, player_features)
        else:
            print(f"No game data found")
            print("Please run data_loader.py first")
