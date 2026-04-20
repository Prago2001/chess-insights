"""
Data Loader Module - Load pre-processed chess game data
Team 029 - CSE6242 Spring 2026

This module provides utilities for loading pre-processed game data from
parquet chunk files in data/processed/chunks/.
"""

import pandas as pd
from pathlib import Path
from typing import Tuple
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import PROCESSED_DATA_DIR


def load_chunks() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load chess game data from pre-processed parquet chunks.

    Returns:
        Tuple of (games_df, full_games_df)
        - games_df: Processed game records with basic metadata
        - full_games_df: Full game data including move sequences
    """
    chunks_dir = PROCESSED_DATA_DIR / "chunks"
    processed_chunks = sorted(chunks_dir.glob("games_processed_part_*.parquet"))
    full_chunks = sorted(chunks_dir.glob("games_full_part_*.parquet"))

    if not processed_chunks or not full_chunks:
        raise FileNotFoundError(
            f"No data chunks found in {chunks_dir}. "
            "Expected games_processed_part_*.parquet and games_full_part_*.parquet files."
        )

    print(f"Loading dataset from {len(processed_chunks)} chunk(s) in {chunks_dir}")
    games_df = pd.concat(
        [pd.read_parquet(p) for p in processed_chunks], ignore_index=True
    )
    full_games_df = pd.concat(
        [pd.read_parquet(p) for p in full_chunks], ignore_index=True
    )
    print(f"Loaded {len(games_df)} games")

    return games_df, full_games_df


def get_dataset_stats(df: pd.DataFrame) -> dict:
    """
    Get basic statistics about the dataset.

    Args:
        df: Games DataFrame

    Returns:
        Dictionary with dataset statistics
    """
    stats = {
        "total_games": len(df),
        "unique_white_players": df["white_player"].nunique() if "white_player" in df.columns else 0,
        "unique_black_players": df["black_player"].nunique() if "black_player" in df.columns else 0,
        "rating_range": (
            int(df["white_elo"].min()) if "white_elo" in df.columns else 0,
            int(df["white_elo"].max()) if "white_elo" in df.columns else 0,
        ),
    }
    return stats


if __name__ == "__main__":
    print("ChessInsight Data Loader")
    print("=" * 50)

    try:
        games_df, full_games_df = load_chunks()
        stats = get_dataset_stats(games_df)

        print(f"\nDataset Statistics:")
        print(f"  Total games: {stats['total_games']:,}")
        print(f"  Unique white players: {stats['unique_white_players']:,}")
        print(f"  Unique black players: {stats['unique_black_players']:,}")
        print(f"  Rating range: {stats['rating_range'][0]} - {stats['rating_range'][1]}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
