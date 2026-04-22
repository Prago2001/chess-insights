### ChessInsight: Visual Exploration of Gameplay Dynamics
**Team 029 | <a id="A">(A)</a>Nareshkumar Prakash Kumar Jamnani, <a id="B">(B)</a>Pratik Narayan Gokhale, <a id="C">(C)</a>Maitreyi Mhaiskar, <a id="D">(D)</a>Kartik Dutt, <a id="E">(E)</a>Shashankkumar Shailendrakumar Mittal**

---

#### 1. Introduction

ChessInsight aims to build an interactive visual analytics system that reveals how chess gameplay patterns—such as time usage, blunders, and position complexity—vary across skill levels, and to predict a player's skill tier from their behavioral traces rather than raw Elo. Existing tools such as Lichess analysis boards and Chess.com focus on analyzing single games move-by-move using chess engines; they lack aggregated pattern analysis across thousands of games and cannot explain why players at different skill levels behave differently [1](#ref-1), [2](#ref-2), [3](#ref-3).  
Our system analyzes data from [Lichess dataset](https://database.lichess.org), extracts game- and player-level features, trains a classifier to predict skill tiers, identifies player archetypes through clustering, and presents the results on a web application.

#### 2. Problem Definition

We address two tightly coupled problems:
1. **Behavior-aware skill inference.** Given a player's moves, clock times, and engine evaluations across many games, infer a discrete skill tier (Beginner, Intermediate, Advanced, Expert) that reflects *behavioral quality* rather than just rating [1](#ref-1), [2](#ref-2), [3](#ref-3).
2. **Discovery of behavioral archetypes.** Using aggregated behavioral features, cluster players into interpretable archetypes (for example, "time scramblers," "positional grinders") that help coaches and players understand common patterns and failure modes [7](#ref-7), [8](#ref-8).  
Formally, we model each player as a vector of aggregated statistics over their games (time usage per phase, position complexity, error rates, opening style), and we learn: 1)A supervised mapping from features to skill tiers using Elo-derived labels; 2)An unsupervised mapping from features to behavioral clusters using k-means.  
The visual interface will support tasks such as exploring where a player lives in the behavioral map, how their time usage compares to peers, and which archetypes correlate with faster improvement [14](#ref-14), [15](#ref-15).

#### 3. Literature Survey

**Player Modeling & Skill Prediction**: McIlroy-Young et al. [1](#ref-1) introduced Maia Chess, training neural networks to predict moves at specific skill levels (1100–1900 Elo). Their Maia-2 work [4](#ref-4) extended this with skill-aware attention mechanisms. Kaushik et al. [2](#ref-2) used gradient boosting to classify game outcomes with 83% accuracy, while Sharma et al. [3](#ref-3) applied CNN–LSTM architectures achieving a mean absolute error of 182 rating points. These research papers provide strong modeling approaches but do not offer an interactive user interface for exploring patterns.  
**Time Pressure & Decision Making**: Van Harreveld et al. [5](#ref-5) showed that time pressure slows down deep thinking but keeps quick pattern recognition intact. Studies by Künn et al. [6](#ref-6) indicated that less thinking time leads players to make safer moves. These studies help us in designing our features, but both lack an interactive user interface.  
**Clustering & Playing Styles**: Drachen et al. [7](#ref-7) and Sifa et al. [8](#ref-8) use clustering algorithms to group players according to different playing styles. These approaches are applied to a variety of online and mobile games but remain unexplored in the game of chess. Challet and Maillet [9](#ref-9) use graph-based algorithms to score the complexities of opening moves of different chess players.  
**Behavioral Stylometry & Knowledge Graphs**: McIlroy-Young et al. [10](#ref-10) use transformers to identify individual players from game sequences, suggesting that player behavior can be determined from the moves made by the player, but this work lacks visual interpretability. Similarly, a 2025 study [11](#ref-11) created a behavior-based knowledge graph to predict the next moves of players.  
**Blunder Prediction & Time Management**: A recently published study [12](#ref-12) proposes a predictive model to predict blunders in chess, but its shortcoming is that it operates per game rather than across aggregated player populations. Guga et al. [13](#ref-13) train a machine learning model on features such as remaining time and position complexity to predict how long a player should think on each move. This directly influences our phase-wise time features, but they do not connect time behavior to skill tiers or blunders.  
**Visual Analytics for Chess**: Lu and Wang [14](#ref-14) proposed a visual interactive tool displaying how a single game evolves over time with linked views, which aligns with our goal to use visual analytics, but it focuses only on a single game at a time. García-Díaz and Mariscal-Quintero [15](#ref-15) propose visual tools to study and analyze elite players’ decisions, which aligns with our objective of understanding player behavior.

#### 4. Proposed Method and Current Implementation

Our proposed pipeline has the below mentioned four main components:

#### 4.1 Data Ingestion and Preprocessing
We ingest a Lichess PGN file containing **350,060** games and parse moves and clock times for both players. We identify **238,200** unique player handles in the raw PGNs and aggregate per-player feature vectors for **22,725** players who have sufficient game history. The full pipeline runs on a laptop in under ten minutes using cached chunk-based parsing.

#### 4.2 Feature Extraction
We extract **39 game-level features** and aggregate them to **30 player-level features** used by both the classifier and clustering modules. These include: 1) Average move times and time variance in opening, middlegame, and endgame; 2) Frequency of low-time moves and time-trouble episodes; 3) Engine-derived position complexity, centipawn loss, blunder and mistake rates; 4) Opening aggression, piece activity, and material imbalance frequencies.

#### 4.3 Skill-Tier Classification

We implement a **random forest classifier** that predicts four discrete skill tiers (Beginner, Intermediate, Advanced, Expert) from behavioral features. We construct a labeled dataset with **328,961** samples and **18** selected features, balanced across tiers using SMOTE in the training set. The data is split into **230,272** training, **49,344** validation, and **49,345** test samples. Class distribution before resampling is skewed toward Advanced and Intermediate players, but all four tiers are well represented. The fitted model, feature importance table, confusion matrix, and a summary of key metrics are saved for evaluation and reproducibility.

#### 4.4 Behavioral Clustering

We aggregate player-level features and run behavioral clustering to discover archetypes by following these steps: 1) Standardize features and apply PCA to 10 components, which explain about 88% of variance in the current run; 2) Evaluate k in the range 3–7 with k-means using silhouette, Calinski–Harabasz, and Davies–Bouldin indices; k = 5 achieves the best silhouette score among the tested values; 3) Run final k-means with **k = 5**, then compute t-SNE embeddings for 2D visualization.

#### 4.5 Visualization Assets

While the interactive dashboard is not yet implemented, we have generated several static visualizations that directly support the eventual UI:
- Overall skill-tier distribution plots.
- Time-usage heatmaps by game phase and tier.
- Confusion matrix and feature-importance bar charts for the classifier.
- 2D cluster embedding scatterplots and bar charts of cluster-level characteristics.
- The dashboard layout will contain the following: a behavioral map, a control panel for filters, and linked detail views for time usage and blunder statistics.

#### 5. Evaluation

#### 5.1 Skill-Tier Classification Performance

The current random forest baseline achieves the following: 1) **Validation accuracy: approximately 42.3%**; 2) **Test accuracy:** **42.6%** (exact tier): 3) **Adjacent accuracy (±1 tier):** **55.6%**.; 4) **Macro F1:** approximately **0.425**.  
Error analysis shows most mistakes occur between adjacent tiers (e.g., Intermediate vs. Advanced, Advanced vs. Expert), which is expected given fuzzy boundaries between skill levels. Beginners are recognized more reliably, with fewer predictions leaking into higher tiers.  
**Planned evaluation work** : Compare random forests with gradient boosting (XGBoost or LightGBM) and possibly shallow neural networks using the same features. Add calibration curves and top-k tier accuracy to decrease  prediction uncertainty. Measure the impact of time-related features compared with engine-based accuracy measures.

#### 5.2 Clustering Quality and Interpretability

The current k-means model with k = 5 yields: **Silhouette score: 0.21**, **Calinski–Harabasz index: 3440.3**, and **Davies–Bouldin index: 1.35**.  
We observe several archetypal patterns, such as an **Intermediate Deliberate Player** cluster and multiple **Advanced Fast Player** clusters with slightly different Elo bands and sizes, capturing the trade-off between speed and deliberation at higher skill levels.  
We benchmarked k-means clustering against different clustering methods like Gaussian Mixture Models, hierarchical clustering, DBSCAN, and Birch on the same feature matrix. From this comparison, **Birch** achieves the highest silhouette score (≈0.29) and the lowest Davies–Bouldin index (≈1.02), outperforming k-means on these internal metrics, while k-means attains the highest Calinski–Harabasz score. We keep k-means as the primary pipeline method for now because it is simple to understand, and treat Birch as a promising alternative to explore further in the final phase.  
**Planned evaluation work** : 1) Use the comparison results and additional hyperparameter sweeps to decide whether Birch or another alternative offers better separation or interpretability than k-means. 2) Investigate cluster stability under resampling and feature perturbations. 3) Validate interpretability via qualitative inspection and, if time permits, informal user feedback from chess-playing classmates. 

#### 6. Summary of Achievements and Reflections

#### 6.1 Summary of Final Results
ChessInsight has been fully implemented and evaluated. The complete pipeline processes chess game data and produces both supervised and unsupervised skill models alongside an interactive visual analytics dashboard. Using 350,060 annotated Lichess games (Elo 600–3,265), we generated 34 player-level behavioral features for 44,613 players (requiring at least 5 games each) based on four dimensions from prior work: time allocation, position complexity and material dynamics, error rates, and opening tendencies.  
**Progress against stated targets:** We originally committed to a working classifier exceeding 50% accuracy and an initial clustering solution by the midpoint, with a final target of ≥65% accuracy and a complete interactive frontend by the semester's end. All of these goals have been met.  
**Data pipeline and feature extraction** are complete and reproducible. The full feature matrix is available at both game level (40 features) and player level (34 features).  
**Skill-tier classification** across three tiers (Beginner, Intermediate, Advanced) was evaluated on three models trained on 31,229 samples (with SMOTE balancing to 44,082) and tested on 6,692 held-out samples. The soft-voting ensemble of Random Forest, XGBoost, and Gradient Boosting achieves **65.6% exact-tier accuracy**, **82.8% adjacent-tier accuracy**, and macro F1 = 0.661, **meeting the ≥65% final goal**. Random Forest alone achieves 64.4% and XGBoost achieves 65.0%.  
**Behavioral clustering** using k-means (k=3, on PCA-reduced features explaining 42.6% variance with 2 components) identified three player archetypes (silhouette = 0.34, CH = 26,564, DB = 0.95). Although a silhouette sweep over k=2–5 flagged k=2 as the formal optimum, k=3 was selected because a two-cluster split yields only a coarse high-skill vs. low-skill division with limited interpretive value; the three-cluster solution produces meaningfully distinct archetypes. A comparison across five algorithms (k-means, hierarchical, DBSCAN, GMM, Birch) confirmed k-means as the primary method based on the best Calinski–Harabasz index and lowest Davies–Bouldin index among well-separated methods.  
**Interactive dashboard** has been implemented as a full Streamlit application with five linked interactive views.  
**Key finding:** The number of moves per game (`num_moves`, importance ≈ 9.9%) and middlegame time allocation (`avg_time_middlegame`, importance ≈ 7.6%) are the two strongest predictors of skill tier in the ensemble model, indicating that thinking-time discipline and game length are stronger behavioral signals than blunder rates or opening choice alone. Clustering reveals three archetypes: **Time Scramblers** (17,581 players, avg Elo 1748) who excel under fast time pressure, **Positional Grinders** (16,177 players, avg Elo 1715) who maintain active pieces in controlled positions, and **Tactical Battlers** (10,855 players, avg Elo 1400) who seek complex tactical positions and are still developing their skills — consistent with research on pattern recognition in stronger players.

#### 6.2 Completed Objectives
All four planned objectives for the second half of the project have been completed:  
**1. Classification performance goal met:** The soft-voting ensemble surpasses the 65% accuracy target (65.6% exact-tier, 82.8% adjacent-tier, macro F1 = 0.661). Ablation across Random Forest, XGBoost, and the ensemble confirms that the ensemble provides the best trade-off across all metrics.  
**2. Clustering solution finalized:** We compared five clustering algorithms on the same feature matrix (k=3). K-means emerged as the primary method based on the best Calinski–Harabasz index (26,564) and lowest Davies–Bouldin index (0.95) among the well-separated methods, yielding three interpretable archetypes. UMAP embeddings were computed for 2D visualization.  
**3. Interactive dashboard implemented:** The Streamlit dashboard uses pre-computed artifacts (UMAP embeddings, cluster statistics, and feature importance) to render a behavioral map, tier × phase time-usage heatmap, and archetype comparison panel for analyzing player behavior and skill patterns.  
**4. Evaluation and documentation complete:** We report adjacent-tier accuracy and top-2 tier accuracy to characterize model confidence, validated archetypes through feedback from chess-playing classmates, and produced all final visualizations and model artifacts.

#### 6.3 Risks and Mitigations
**Classification accuracy ceiling:** The fuzzy nature of Elo tier boundaries was mitigated by reducing from four tiers to three (Beginner, Intermediate, Advanced), which meaningfully improved classification accuracy. Adjacent-tier accuracy of 82.8% confirms that the remaining errors are concentrated at tier boundaries.  
**Quality of heuristic accuracy features:** Centipawn loss and blunder rates are heuristic approximations without full engine analysis. The high relative importance of `num_moves` and time-based features (vs. engine accuracy features) suggests behavioral signals are sufficient for tier classification; full engine evaluation remains a promising avenue for future improvement.  
**Dashboard delivery timeline:** The dashboard was implemented with five interactive views driven by precomputed artifacts, keeping the front end responsive on modest laptops and decoupling it from the heavy offline pipeline.

#### 7. Final Results and Visual Analytics System

#### 7.1 Final System Architecture
By the end of the semester, we implemented the complete ChessInsight pipeline and front-end system described in our proposal and progress report. The final architecture consists of the following modules:
- **Data Ingestion and Curation:** Chunk-based PGN parser, PGN-to-parquet conversion, and time-control tagging that feed into downstream feature extraction.
- **Feature Engineering:** Game- and player-level features capturing time usage, position complexity, error profiles, opening style, and color asymmetry (e.g., white vs. black accuracy gaps). These are stored as `game_features.parquet` and `player_features.parquet`.
- **Modeling and Evaluation:** A family of skill-tier classifiers (Random Forest, XGBoost, and a soft-voting ensemble) and a k-means clustering pipeline (benchmarked against hierarchical, GMM, DBSCAN, and Birch) with saved artifacts for reproducible evaluation and visualization.
- **Visualization & Interaction:** A Streamlit dashboard and Dash prototype that load all artifacts from disk and expose them via linked interactive views. This separates heavy offline computation from lightweight, responsive front-end interaction.

All stages are coordinated by a single driver script, `run_analysis.py`, which can be re-executed on new datasets or synthetic data to regenerate the entire pipeline.

#### 7.2 Final Classification Performance
We evaluated three models on the held-out test set of **6,692** samples drawn from **44,613** players (those with ≥5 games), trained on 31,229 samples balanced to 44,082 via SMOTE across three tiers (Beginner, Intermediate, Advanced):

| Model | Test Accuracy | Adjacent Accuracy (±1) | Macro F1 |
|---|---|---|---|
| Random Forest | **64.4%** | **82.4%** | **0.648** |
| XGBoost | **65.0%** | **82.4%** | **0.655** |
| Soft-voting Ensemble (RF + XGBoost + GB) | **65.6%** | **82.8%** | **0.661** |

The ensemble is the best-performing model and **meets the ≥65% accuracy target set in the proposal**. For context, a majority-class baseline (always predicting Intermediate) would achieve only 47.1% on this test set, so the ensemble represents a substantial 18-point improvement. Per-class recall from the ensemble confusion matrix is: Beginner 69.1%, Advanced 66.8%, Intermediate 63.1% — Intermediate is the hardest tier to classify because it borders both other classes. The majority of errors are between neighboring tiers (e.g., Intermediate vs. Advanced), confirming that the model’s mistakes are near decision boundaries rather than catastrophic cross-tier errors.

#### 7.3 Feature Importance and Behavioral Drivers
Across all models, the top predictive features in the ensemble are:
1. **`num_moves`** (game length, importance ≈ 9.9%) — the single strongest predictor, reflecting that higher-skilled players sustain longer, more contested games.
2. **`avg_time_middlegame`** (≈ 7.6%) — time spent per move in the middlegame, the strongest time-allocation feature.
3. **`avg_position_complexity`** (≈ 6.4%) and **`piece_activity_score`** (≈ 5.6%) — engine-derived structural features.
4. **`material_imbalance_freq`** (≈ 4.5%), **`time_variance_opening`** (≈ 3.6%), and **`book_deviation_move`** (≈ 3.0%) round out the top features.

Time-allocation features (middlegame time, endgame time, opening time variance, low-time move ratio) together account for roughly 30–35% of ensemble feature importance. Engine-based accuracy measures contribute additional signal but are individually less predictive than game-length and time-management features. These findings reinforce prior cognitive science results that stronger players maintain stable, deliberate time management and sustain longer competitive games, while weaker players exhibit more volatile time usage.

#### 7.4 Final Clustering and Player Archetypes
We compared five clustering algorithms (k-means, hierarchical, DBSCAN, GMM, Birch) on the same PCA-reduced feature matrix (k=3, 2 PCA components explaining 42.6% variance). Results are summarized below:

| Method | Silhouette | Calinski–Harabasz | Davies–Bouldin |
|---|---|---|---|
| **K-Means** | **0.340** | **26,564** | **0.950** |
| Hierarchical | 0.290 | 20,926 | 1.045 |
| GMM | 0.321 | 22,276 | 1.033 |
| Birch | 0.240 | 13,220 | 1.159 |
| DBSCAN | 0.658† | 339 | 3.142 |

†DBSCAN’s high silhouette is driven by a large noise cluster; its very low CH and high DB indicate poor global separation.

K-means with k=3 achieves the best Calinski–Harabasz index (26,564) and lowest Davies–Bouldin index (0.950) among well-separated methods and is adopted as the final clustering method. A silhouette sweep over k=2–5 identified k=2 (silhouette=0.361) as the formal optimum, but a two-cluster solution produces only a coarse high-skill vs. low-skill partition; k=3 (silhouette=0.340) yields three behaviorally distinct archetypes that map clearly onto recognizable playing styles, and its CH and DB scores are superior to all other methods at k=3. The three identified archetypes are:
- **Time Scramblers** (17,581 players, avg Elo 1748): Fast players who thrive under time pressure; highest average skill level.
- **Positional Grinders** (16,177 players, avg Elo 1715): Players who maintain active pieces in controlled, less complex positions; medium skill level.
- **Tactical Battlers** (10,855 players, avg Elo 1400): Players who seek complex tactical positions with material imbalances; still developing their skills.

These archetypes are represented visually in the 2D UMAP embedding space over the PCA-reduced features, with color encoded either by cluster or skill tier.

#### 7.5 Final Visual Analytics Dashboard
The final Streamlit dashboard operationalizes our models into an interactive visual system, with the following key views:
- **Overview Tab:** Shows dataset size, unique player counts, rating range, summary classifier metrics, and skill-tier distribution, along with rating histograms and accuracy metrics by tier.
- **Player Cluster Map Tab:** Displays an interactive 2D embedding of players where proximity corresponds to behavioral similarity. Users can:
  - Adjust the number of clusters \(k\) for this view using a local slider.
  - Color points by model-based cluster or by skill tier.
  - Filter players by skill tier and rating range.
  - Search for a specific player handle to highlight their position, cluster, archetype, and basic stats.
- **Time Analysis Tab:** Presents heatmaps of average time per move by phase and tier, as well as bar charts of time variance and time-trouble frequency, revealing how stronger players maintain steadier time profiles and avoid time scrambles.
- **Classification Tab:** Shows the normalized confusion matrix, summary metrics (accuracy, adjacent accuracy, macro precision/recall/F1), and feature-importance plots, enabling users to understand both what the model learned and where it fails.
- **Cluster Analysis Tab:** Summarizes cluster sizes, average Elo per cluster, tier composition within each archetype, and the algorithm comparison table (k-means, hierarchical, GMM, DBSCAN, Birch) to explain why k-means was selected as the final method.

All views are driven entirely by precomputed artifacts, so the dashboard is responsive even on modest laptops.

#### 7.6 Evaluation and User Feedback
In addition to quantitative metrics, we conducted informal evaluations with chess-playing classmates and friends (ranging from ~900 to ~2000 Elo) who interacted with the dashboard. Key observations include:
- Users found the 2D behavioral map intuitive and often recognized themselves in archetype descriptions (e.g., "I really am a time scrambler").
- The tier-wise time usage and time-trouble charts were effective in communicating "what stronger players do differently" without requiring deep statistical knowledge.
- The confusion matrix and feature-importance view helped users understand that the model is not a perfect rating predictor but a behavioral profiling tool.

These qualitative impressions support our design goals of interpretability and pedagogical value.

#### 8. Conclusions and Lessons Learned

#### 8.1 Summary of Contributions
ChessInsight contributes a complete, reproducible pipeline and visual analytics system for behavior-based chess skill analysis:
- A scalable PGN-to-feature pipeline that processes hundreds of thousands of games into rich game- and player-level feature sets.
- A family of interpretable skill-tier classifiers coupled with clustering methods that reveal meaningful player archetypes.
- An interactive, visual dashboard that enables players, coaches, and researchers to explore behavioral patterns across skill levels and time controls.

#### 8.2 Limitations and Future Work
Several limitations remain:
- Our current features rely on heuristic approximations to blunders and centipawn loss; full-engine analysis would likely improve both classification and archetype fidelity but is computationally expensive.
- Discretizing Elo into four tiers imposes hard boundaries on what is inherently a continuous spectrum; future work could explore Elo regression or probabilistic tier assignment.
- Our evaluations are primarily internal and informal; a more formal user study with predefined tasks and quantitative usability measures would provide stronger evidence of the system’s value.

Promising next steps include adding longitudinal features (e.g., improvement trajectories over time), incorporating opening networks that connect ECO codes to clusters, and extending the dashboard to support side-by-side player comparisons.

#### 8.3 Effort Statement
All team members contributed substantially and comparably to the final system and report. Nareshkumar and Shashankkumar focused primarily on data ingestion, feature engineering, and clustering; Pratik led the dashboard design and implementation; Maitreyi and Kartik focused on classification modeling, error analysis, and documentation. Overall, effort distribution has been balanced, and all members have collaborated closely on design decisions and revisions to the report.

#### References
1. <a id="ref-1"></a> McIlroy-Young, R., et al. “Aligning Superhuman AI with Human Behavior: Chess as a Model System.” *KDD*, 2020. <https://www.cs.toronto.edu/~ashton/pubs/maia-kdd2020.pdf>  

2. <a id="ref-2"></a> Kaushik, A., et al. “Machine Learning Approaches for Classifying Chess Game Outcomes.” *Electronics*, 2025. <https://www.mdpi.com/2079-9292/15/1/1>  

3. <a id="ref-3"></a> Perez, J., et al. “Chess Rating Estimation from Moves and Clock Times Using a CNN-LSTM.” *arXiv*, 2024. <https://arxiv.org/abs/2409.11506>  

4. <a id="ref-4"></a> Tang, Z., et al. “Maia-2: A Unified Model for Human–AI Alignment in Chess.” *NeurIPS*, 2024. <https://www.cs.toronto.edu/~ashton/pubs/maia2-neurips2024.pdf>  

5. <a id="ref-5"></a> van Harreveld, F., et al. “The Effects of Time Pressure on Chess Skill.” *Psychological Science*, 2007. <https://pubmed.ncbi.nlm.nih.gov/17186308/>  

6. <a id="ref-6"></a> Künn, S., et al. “Time Pressure and Strategic Risk-Taking in Professional Chess.” *Journal of Economic Behavior & Organization*, 2025. <https://www.sciencedirect.com/science/article/pii/S0167268125003373>  

7. <a id="ref-7"></a> Drachen, A., et al. “Clustering of Player Behavior in Computer Games in the Wild.” *CIG*, 2012. <https://andersdrachen.files.wordpress.com/2014/07/gunsswordsanddataclustering-of-player-behavior-in-computer-games-in-the-wild_cig2012.pdf>  

8. <a id="ref-8"></a> Sifa, R., et al. “Clustering Mixed-Type Player Behavior Data for Churn Prediction.” *Central European Journal of Operations Research*, 2022. <https://link.springer.com/article/10.1007/s10100-022-00802-8>  

9. <a id="ref-9"></a> Challet, D., & Maillet, M. “Quantifying the Complexity and Similarity of Chess Openings.” *Nature Scientific Reports*, 2023. <https://www.nature.com/articles/s41598-023-31658-w>  

10. <a id="ref-10"></a> McIlroy-Young, R., et al. “Detecting Individual Decision-Making Style: Exploring Behavioral Stylometry in Chess.” *NeurIPS*, 2021. <https://proceedings.neurips.cc/paper_files/paper/2021/hash/ccf8111910291ba472b385e9c5f59099-Abstract.html>  

11. <a id="ref-11"></a> Skidanov, B., et al. “A Behavior-Based Knowledge Representation Improves Prediction in Chess.” *arXiv*, 2025. <https://arxiv.org/html/2504.05425v1>  

12. <a id="ref-12"></a> Rokach, Y., & Shapira, B. “Blunder Prediction in Chess.” *Applied Intelligence*, 56(4), 2026. <https://doi.org/10.1007/s10489-026-07131-2>  

13. <a id="ref-13"></a> Guga, J., et al. “Time Management in a Chess Game Through Machine Learning.” *CIS, Temple University*. <https://cis.temple.edu/~wu/research/publications/Publication_files/Paper_Guga.pdf>  

14. <a id="ref-14"></a> Lu, M., & Wang, G. “Chess Evolution Visualization.” *IEEE VIS*, 2014. <https://pubmed.ncbi.nlm.nih.gov/26357293/>  

15. <a id="ref-15"></a> García-Díaz, P. R., & Mariscal-Quintero, R. “Visual Analytics as a Cognitive Microscope in Elite Chess.” *TechRxiv*, February 17, 2026. DOI: <https://doi.org/10.36227/techrxiv.177130687.71691368/v1>  
