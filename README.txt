================================================================================
CHESSINSIGHT - Chess Player Analytics and Behavioral Classification
Team 029 - CSE6242 Data and Visual Analytics - Spring 2026
================================================================================

DESCRIPTION
-----------
ChessInsight is a chess analytics platform that analyzes Lichess game data to:
1. Extract behavioral features from chess games (time usage, complexity, etc.)
2. Classify players into 3 skill tiers using machine learning (65.8% accuracy)
3. Identify player behavioral archetypes through K-Means clustering (K=3)
4. Provide interactive visualizations via Streamlit dashboard

The system processes PGN (Portable Game Notation) files containing chess games
with clock annotations to extract time-based features and behavioral metrics.

Key Results:
- Dataset: 350,060 games from 44,613 players (Lichess February 2026)
- Classification: 3-tier system (Beginner/Intermediate/Advanced) with 65.8% accuracy
- Clustering: 3 behavioral archetypes identified (silhouette score: 0.34)
  * Time Scramblers (68%): Fast play, comfortable in time pressure, highest Elo
  * Positional Grinders (20%): Methodical play, lower time variance
  * Tactical Battlers (12%): Seek complex positions, highest position complexity

================================================================================

QUICK START (Dashboard Only - No Data Download Required)
---------------------------------------------------------
The repository includes pre-computed cached data, so you can run the dashboard
immediately without downloading the 2.2 GB PGN file:

   1. pip install -r requirements.txt
   2. streamlit run streamlit_app.py
   3. Open http://localhost:8501 in your browser

Time to working dashboard: < 5 minutes

================================================================================

SYSTEM REQUIREMENTS
-------------------
Minimum:
- Python 3.9 or higher
- 4 GB RAM
- 500 MB disk space (for dependencies + cached data)

Recommended (for full pipeline with 1M games):
- Python 3.9+
- 8+ GB RAM
- 10 GB disk space (includes 2.2 GB PGN file)

Tested Environments:
- macOS 13+ with Python 3.9-3.13
- Ubuntu 20.04+ with Python 3.9-3.12
- Windows 10/11 with Python 3.9-3.12

================================================================================

INSTALLATION
------------
1. Ensure Python 3.9+ is installed:
   python3 --version

2. (Recommended) Create a virtual environment:
   python3 -m venv venv
   source venv/bin/activate        # On Windows: venv\Scripts\activate

3. Navigate to the project directory:
   cd "cse6242-group-project-main 2"

4. Install required dependencies:
   pip install -r requirements.txt

   Key packages:
   - python-chess (PGN parsing)
   - pandas, numpy (data manipulation)
   - scikit-learn, xgboost (machine learning)
   - matplotlib, seaborn, plotly (visualization)
   - streamlit (interactive dashboard)

5. DATA OPTIONS:

   Option A - Dashboard Only (RECOMMENDED):
   The repository includes pre-computed data in data/processed/ and models/
   No additional download is required to run the dashboard.

   Option B - Full Pipeline (requires 2.2 GB download):
   To re-run the analysis pipeline from raw data:
   - Download from: https://database.lichess.org/
   - Get: lichess_db_standard_rated_2026-02.pgn.zst
   - Decompress: zstd -d lichess_db_standard_rated_2026-02.pgn.zst
   - Rename and place in: data/raw/data_1m_games.pgn

================================================================================

EXECUTION
---------

1. INTERACTIVE DASHBOARD (Primary Interface)
   -----------------------------------------
   streamlit run streamlit_app.py

   Opens at: http://localhost:8501

   Dashboard Tabs:
   - Overview: Dataset stats, model metrics, archetype summaries
   - Player Cluster Map: Interactive 2D PCA visualization with filters
   - Time Analysis: Time usage heatmaps by skill tier and game phase
   - Classification: Confusion matrix, feature importance charts
   - Cluster Analysis: Archetype comparison radar charts, statistics

2. ANALYSIS PIPELINE (Optional - requires PGN file)
   -------------------------------------------------
   python run_analysis.py [options]

   Options:
   --n-games N       Number of games to process (default: 1000000)
   --synthetic       Use synthetic data for testing (no PGN needed)
   --force-reload    Re-parse PGN, ignoring cached data

   Examples:
   python run_analysis.py --synthetic          # Test without data file
   python run_analysis.py --n-games 50000      # Process 50K games
   python run_analysis.py --force-reload       # Force re-processing

3. ALTERNATIVE DASHBOARD (Dash/Plotly)
   ------------------------------------
   python dashboard.py

   Opens at: http://127.0.0.1:8050
   (Same functionality as Streamlit, alternative framework)

================================================================================

PROJECT STRUCTURE
-----------------
cse6242-group-project-main 2/
├── README.txt              # This file (detailed setup)
├── README.md               # Quick reference
├── requirements.txt        # Python dependencies
├── config.py               # Configuration (paths, tiers, features)
├── run_analysis.py         # Main analysis pipeline
├── streamlit_app.py        # Primary interactive dashboard
├── dashboard.py            # Alternative Dash dashboard
│
├── src/                    # Core modules
│   ├── data_loader.py      # PGN parsing and data loading
│   ├── feature_extractor.py # Feature engineering (41 game + 30 player features)
│   ├── classifier.py       # Skill tier classification (RF, XGBoost, Ensemble)
│   ├── clustering.py       # K-Means behavioral clustering
│   └── visualizations.py   # Chart generation
│
├── data/
│   ├── raw/                # Place PGN file here (if running full pipeline)
│   └── processed/          # Pre-computed parquet files (INCLUDED)
│       ├── game_features.parquet      # 350K games, 28 MB
│       ├── player_features.parquet    # 44K players, 12 MB
│       └── player_clustering_embeddings.parquet
│
├── models/                 # Trained models and results (INCLUDED)
│   ├── skill_classifier_*.pkl         # Serialized classifiers
│   ├── player_clustering_results.json # Clustering results
│   ├── cluster_centers_final.csv      # Archetype centroids
│   └── analysis_summary.json          # Final metrics summary
│
└── visualizations/         # Generated charts (PNG)

================================================================================

CONFIGURATION
-------------
Edit config.py to customize:

- SAMPLE_SIZE: Number of games to process (default: 1000000)
- MIN_GAMES_PER_PLAYER: Minimum games for analysis (default: 5)
- SKILL_TIERS: Elo boundaries for classification
    * Beginner: 0-1399
    * Intermediate: 1400-1899
    * Advanced: 1900+
- VALID_TIME_CONTROLS: bullet, blitz, rapid
- N_CLUSTERS_RANGE: (2, 5) - K=3 is optimal

================================================================================

TROUBLESHOOTING
---------------
1. "No module named streamlit":
   pip install streamlit

2. Dashboard shows "Failed to load data":
   - Ensure data/processed/ contains parquet files
   - Check models/ contains JSON and CSV files
   - Try: python -c "import streamlit_app" to see errors

3. Memory errors with large datasets:
   - Reduce --n-games parameter
   - Use --synthetic for testing

4. "PGN file not found" (only for full pipeline):
   - Dashboard works without PGN (uses cached data)
   - For pipeline: place data_1m_games.pgn in data/raw/

5. Slow first run:
   - PGN parsing is CPU-intensive
   - Subsequent runs use parquet cache (much faster)

6. Port 8501 already in use:
   streamlit run streamlit_app.py --server.port 8502

================================================================================

OUTPUT FILES
------------
After running the analysis pipeline:

data/processed/
- games_processed.parquet      # Cleaned game records
- game_features.parquet        # Extracted game-level features
- player_features.parquet      # Aggregated player features
- player_clustering_embeddings.parquet  # PCA embeddings + cluster labels

models/
- skill_classifier_metrics.json         # Accuracy, F1, confusion matrix
- skill_classifier_feature_importance.csv
- player_clustering_results.json        # Archetype definitions
- cluster_centers_final.csv             # Cluster centroids
- analysis_summary.json                 # Overall summary

visualizations/
- confusion_matrix.png
- feature_importance.png
- cluster_visualization.png
- time_heatmap.png
- skill_distribution.png

================================================================================

DATASET INFORMATION
-------------------
Source: Lichess.org open database (https://database.lichess.org/)
Period: February 2026
Format: PGN with clock annotations [%clk H:MM:SS]

Processed Dataset:
- Total games: 350,060
- Unique players: 44,613
- Rating range: 600 - 3265 Elo
- Time controls: Bullet, Blitz, Rapid

Features Extracted:
- Time usage per game phase (opening/middlegame/endgame)
- Time variance and time trouble frequency
- Position complexity metrics
- Opening aggression and book deviation
- Material imbalance frequency

================================================================================

TEAM & CONTACT
--------------
Team 029 - Georgia Tech CSE6242
Data and Visual Analytics - Spring 2026

Project: ChessInsight - Chess Player Behavioral Analytics
Dataset: Lichess Open Database

================================================================================
