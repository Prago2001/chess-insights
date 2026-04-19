# ChessInsight - Chess Player Analytics Platform

**Team 029 | CSE6242 Data and Visual Analytics | Georgia Tech | Spring 2026**

ChessInsight analyzes chess game data from Lichess to classify player skill tiers and identify behavioral archetypes through machine learning.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the interactive dashboard (no data download needed!)
streamlit run streamlit_app.py
```

Open http://localhost:8501 in your browser.

## Key Results

| Metric | Value |
|--------|-------|
| Dataset | 350,060 games, 44,613 players |
| Classification Accuracy | 65.8% (3-tier) |
| Clustering | K=3, Silhouette=0.34 |

### Player Archetypes

| Archetype | % Players | Avg Elo | Characteristics |
|-----------|-----------|---------|-----------------|
| Time Scramblers | 68% | 1707 | Fast play, comfortable in time pressure |
| Positional Grinders | 20% | 1552 | Methodical, lower time variance |
| Tactical Battlers | 12% | 1500 | Seek complex positions |

### Skill Tiers

- **Beginner**: Elo < 1400
- **Intermediate**: Elo 1400-1899
- **Advanced**: Elo 1900+

## Project Structure

```
├── streamlit_app.py     # Primary dashboard (recommended)
├── dashboard.py         # Alternative Dash dashboard
├── run_analysis.py      # Analysis pipeline
├── config.py            # Configuration
├── src/                 # Core modules
│   ├── data_loader.py
│   ├── feature_extractor.py
│   ├── classifier.py
│   ├── clustering.py
│   └── visualizations.py
├── data/processed/      # Pre-computed data (included)
├── models/              # Trained models (included)
└── visualizations/      # Generated charts
```

## Running the Analysis Pipeline

The dashboard works with pre-computed data. To re-run the full pipeline:

```bash
# With real data (requires 2.2 GB PGN file in data/raw/)
python run_analysis.py

# With synthetic data (for testing)
python run_analysis.py --synthetic

# Limit number of games
python run_analysis.py --n-games 50000
```

### Data Source

Download from [Lichess Database](https://database.lichess.org/):
- File: `lichess_db_standard_rated_2026-02.pgn.zst`
- Decompress and place in `data/raw/data_1m_games.pgn`

## Dashboard Features

### Streamlit Dashboard (Recommended)

```bash
streamlit run streamlit_app.py
```

**Tabs:**
1. **Overview** - Dataset stats, model metrics, archetype summaries
2. **Player Cluster Map** - Interactive PCA visualization with filters
3. **Time Analysis** - Time usage heatmaps by skill tier
4. **Classification** - Confusion matrix, feature importance
5. **Cluster Analysis** - Archetype comparison, radar charts

### Dash Dashboard (Alternative)

```bash
python dashboard.py
# Opens at http://127.0.0.1:8050
```

## System Requirements

- Python 3.9+
- 4 GB RAM (8+ GB for full pipeline)
- 500 MB disk (10 GB for full pipeline with data)

## Dependencies

Key packages (see `requirements.txt` for full list):
- `pandas`, `numpy` - Data processing
- `scikit-learn`, `xgboost` - Machine learning
- `plotly`, `matplotlib`, `seaborn` - Visualization
- `streamlit`, `dash` - Interactive dashboards
- `python-chess` - PGN parsing

## Configuration

Edit `config.py` to customize:
- `SAMPLE_SIZE` - Number of games to process
- `SKILL_TIERS` - Elo boundaries for classification
- `N_CLUSTERS_RANGE` - Clustering parameter range

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Module not found | `pip install -r requirements.txt` |
| Dashboard won't load | Check `data/processed/` has parquet files |
| Memory error | Use `--n-games 50000` to reduce dataset |
| Port in use | `streamlit run streamlit_app.py --server.port 8502` |

## Documentation

- `README.txt` - Detailed setup instructions
- `docs/analysis_context.md` - Analysis methodology
- `DOC/PROGRESS_ANALYSIS.md` - Development progress

---

**Team 029** - Georgia Tech CSE6242 Spring 2026
