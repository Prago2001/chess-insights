# ChessInsight: Visual Exploration of Gameplay Dynamics

## Team 029
Nareshkumar Prakash Kumar Jamnani (A)  
Pratik Narayan Gokhale (B)  
Maitreyi Mhaiskar (C)  
Kartik Dutt (D)  
Shashankkumar Shailendrakumar Mittal (E)

---

## Problem & Motivation
Chess players and coaches today have powerful move-by-move analysis tools, but lack visual analytics that summarize how behavior (time usage, complexity management, blunders, and openings) varies across skill tiers. ChessInsight addresses this gap by providing a behavior-based skill profiler and visual exploration tool over hundreds of thousands of Lichess games.

---

## Data & Features
- 350,060 games from the public Lichess dataset (Elo 600–3,265).
- 238,200 unique players; 22,725 players with sufficient history for player-level analysis.
- 39 game-level features → 30 player-level features, capturing:
  - Time allocation per phase (opening, middlegame, endgame), time variance, and time-trouble frequency.
  - Heuristic accuracy signals (blunder rate, mistake rate, average centipawn loss, accuracy percentage).
  - Position complexity, material imbalance, piece activity.
  - Opening style (ECO code families, aggression score, repertoire entropy).
  - Color asymmetry (white vs. black accuracy and time differences).

---

## Methods

### Supervised Skill-Tier Classification
- Labels: four discrete skill tiers (Beginner, Intermediate, Advanced, Expert) derived from Elo thresholds.
- Models: Random Forest, XGBoost, and a soft-voting ensemble (RF + XGBoost).
- Training setup:
  - 328,961 samples, 18 selected behavioral features.
  - Stratified split into train/validation/test (≈70/15/15).
  - SMOTE used to balance tiers in the training set.

### Unsupervised Behavioral Clustering
- Pipeline: standardization → PCA (10 components; ≈88% variance) → clustering.
- Methods: k-means (baseline) and Birch (final).
- Internal metrics: silhouette score, Calinski–Harabasz index, Davies–Bouldin index.

### Visual Analytics System
- Offline pipeline (`run_analysis.py`) generates all features, models, and visual artifacts.
- Online dashboards (Streamlit + Dash prototype) load processed artifacts and expose them via interactive views.

---

## Key Results

### Classification
- Random Forest baseline: ≈42.6% exact-tier accuracy; ≈55.6% adjacent accuracy; macro F1 ≈0.425.
- XGBoost: modest gains in distinguishing Advanced vs. Expert tiers.
- RF + XGBoost ensemble:
  - ≈47% exact-tier accuracy.
  - ≈63% adjacent accuracy (within ±1 tier).
  - Macro F1 ≈0.46.
- Most errors occur between neighboring tiers, consistent with fuzzy Elo boundaries.

### Clustering & Archetypes
- k-means (k = 5): silhouette ≈0.21, CH ≈3440, DB ≈1.35.
- Birch (final): silhouette ≈0.29, DB ≈1.02, improved separation and compactness.
- Representative archetypes:
  - Deliberate Strategists (high middlegame time, low time-trouble, low blunder).
  - Fast Tacticians (strong players with faster time usage, concentrated in Blitz/Bullet).
  - Time Scramblers (frequent time trouble and endgame blunders).
  - Opening Specialists (diverse, aggressive openings; middlegame accuracy drops).
  - Swingy Improvers (high accuracy volatility, transitioning between tiers).

### Behavioral Drivers
- Time-allocation features dominate feature importance (≈50–60% of total importance).
- Accuracy features and opening repertoire measures provide complementary signal.
- Color asymmetry features reveal players who overperform with one color.

---

## Visual Analytics Dashboard

### Overview View
- Dataset statistics, rating distribution, and skill-tier counts.
- Global view of classification metrics and accuracy by tier.

### Player Cluster Map
- 2D t-SNE embedding of players colored by cluster or skill tier.
- Local k-slider to adjust the number of clusters for the view.
- Filters for skill tier and rating range.
- Player search to highlight an individual’s location, archetype, and summary stats.

### Time & Accuracy Views
- Heatmaps of average time per move by phase × tier.
- Bar charts of time variance and time-trouble frequency.
- Tier-wise trends: stronger players use time more steadily and avoid severe time scrambles.

### Model & Cluster Views
- Confusion matrix and feature-importance plots for the skill classifier.
- Cluster size, average Elo, and tier composition for each archetype.
- Comparison table of k-means vs. Birch using multiple internal metrics.

---

## Insights & Takeaways
- Thinking-time allocation is a stronger indicator of playing strength than raw blunder counts or opening choice.
- Behavior-based archetypes (e.g., "time scramblers" vs. "deliberate strategists") help explain *how* players achieve their results, not just *what* their ratings are.
- Visual analytics makes these patterns accessible to non-experts, enabling players and coaches to reason about strengths, weaknesses, and styles.

---

## Future Directions
- Incorporate full-engine evaluations for more accurate blunder and centipawn-loss features.
- Explore Elo regression or probabilistic tier assignments instead of hard buckets.
- Add longitudinal features to track improvement trajectories and opening/style changes over time.
- Conduct more formal user studies to quantify usability and learning impact.
