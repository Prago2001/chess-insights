# cse6242-group-project

Steps to create PDF using pandoc + weasyprint
```
pandoc team029proposal.md \
  --pdf-engine=weasyprint \
  --css=style.css \
  -o team029proposal.pdf
```

## Running the analysis pipeline

The end-to-end ChessInsight analysis (data load → feature extraction → models → clustering → visualizations) is driven by `run_analysis.py`.

From the repository root:

```bash
# Real Lichess data (requires data/raw/data_1m_games.pgn)
python run_analysis.py --n-games 1000000

# Synthetic data for quick testing (no PGN required)
python run_analysis.py --synthetic --n-games 50000
```

This will populate:

- `data/processed/` — `games_processed.parquet`, `game_features.parquet`, `player_features.parquet`, and chunked parquet files.
- `models/` — trained skill classifiers (Random Forest, XGBoost, ensemble), confusion matrices, feature importance, clustering results, and `analysis_summary.json`.
- `visualizations/` — static plots used by the report and dashboards.

## Running the dashboards

We provide both a Dash prototype and a Streamlit dashboard. For local development we recommend Streamlit.

### Streamlit dashboard (recommended)

1. Install dependencies (once):

```bash
pip install -r requirements.txt
```

2. Run the analysis pipeline at least once (see above) so that `data/processed/` and `models/` are populated.

3. Launch the dashboard from the repo root:

```bash
streamlit run streamlit_app.py
```

4. Open the URL printed in the terminal (typically `http://localhost:8501`).

The Streamlit app exposes:

- Overview of dataset, model performance, and cluster count.
- Player Cluster Map with an adjustable number of clusters (k) and per-player drill‑down.
- Time usage and time-pressure analytics by skill tier.
- Skill-tier classification confusion matrix and feature importances.
- Behavioral cluster analysis and clustering-method comparison.
- Opening patterns by skill tier.

### Dash prototype (optional)

You can still run the original Dash-based dashboard for comparison:

```bash
python dashboard.py
```

This serves a Dash app at `http://127.0.0.1:8050` that uses the same processed data and model artifacts as the Streamlit dashboard.
