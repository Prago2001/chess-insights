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

#### 6. Conclusions, Remaining Work, and Risks

#### 6.1 Summary of Progress
Our goal for the first half of the semester was to build an analytical backend capable of processing chess game data and generating supervised and unsupervised skill models, which has been successfully achieved. Using 350,060 annotated Lichess games (Elo 600–3,265), we generated 30 player-level behavioral features for 22,725 players based on four dimensions from prior work: time allocation, position complexity and material dynamics, error rates, and opening tendencies.  
**Progress against stated midterm targets:** We originally committed to a working classifier exceeding 50% accuracy and an initial clustering solution by the midpoint, with a final target of ≥65% accuracy and a complete interactive frontend by the semester's end. Our current standing against those goals is as follows:  
**Data pipeline and feature extraction** are complete and reproducible. The full feature matrix is available at both game level (39 features) and player level (30 features).  
**Skill-tier classification** using a Random Forest with 18 behavioral features achieves **42.6% exact-tier accuracy** and **55.6% adjacent-tier accuracy** (macro F1 = 0.425) on 49,345 test samples; although slightly below the 50% midterm target, the higher adjacent accuracy indicates errors mostly occur between neighboring tiers. To reach the **65% final goal**, we implemented a soft-voting ensemble model (XGBoost, Random Forest, Gradient Boosting) with interaction features, which is scheduled for evaluation in the next phase.  
**Behavioral clustering** using k-means (k=5 on PCA-reduced features retaining 87.6% variance) identified five player archetypes (silhouette = 0.21, CH = 3440, DB = 1.35). A comparison across five algorithms showed Birch performed better (silhouette = 0.29, DB = 1.02), improving these metrics by 38% and 25%, so it will be adopted for the final clustering solution.
**Interactive dashboard** remains the principal outstanding deliverable; the current output is a static matplotlib wireframe illustrating the intended layout.  
**Key finding :** Time-based features dominate prediction, with phase-wise time statistics accounting for ~57% of model importance (middlegame time = 16.4%), indicating that thinking-time allocation is a stronger indicator of skill than blunders or openings; clustering further reveals a divide between deliberate players (avg Elo ≈1453) and faster Advanced–Expert players (avg Elo ≈1706–1774), consistent with research on rapid pattern recognition in stronger players.

#### 6.2 Plan for the Remaining Semester
The second half of the project has four clearly sequenced objectives:  
**1. Improve classification performance:** We will compare the ensemble to the Random Forest baseline using exact accuracy, adjacent accuracy, and macro F1 with ablation tests, and if it fails to reach 65%, we will explore Elo regression to avoid ambiguity in discrete skill tiers.  
**2. Finalize and validate the clustering solution:** We will adopt Birch clustering, test archetype stability via bootstrap resampling, adjust cluster count if needed, and examine whether overlapping “Advanced Fast Player” clusters are driven by time controls (Bullet, Blitz, Rapid) rather than behavioral differences.  
**3. Implement the interactive dashboard:** The frontend will use pre-computed artifacts (t-SNE embeddings, cluster statistics, and feature importance) to render a behavioral map, tier × phase time-usage heatmap, and archetype comparison panel for analyzing player behavior and skill patterns.  
**4. Evaluate, document, and demo:** We will report calibration curves and top-2 tier accuracy to assess model confidence, validate archetypes through feedback from chess-playing classmates, and, if time permits, add a D3.js opening network linking ECO codes to archetypes.

#### 6.3 Risks and Mitigation Strategies
These three risks need careful attention during the rest of the project:  
**Classification accuracy ceiling:** Because Elo tier boundaries are fuzzy (e.g., 1598 vs. 1602 in different tiers), exact-tier accuracy may have a ceiling, so we will test the ensemble, improve features with engine evaluations, or shift to Elo regression or two-tier classification if needed.  
**Quality of heuristic accuracy features:** Because centipawn loss and blunder rates are heuristic approximations without engine analysis, we will test whether running Stockfish on a stratified sample improves feature quality and model performance within available compute time.  
**Dashboard delivery timeline.** Since the dashboard is the most time-intensive task, we will prioritize the three core views and treat additional visualizations as stretch goals while using pre-computed artifacts to avoid rerunning the full pipeline.

#### 7. Final Results and Visual Analytics System

#### 7.1 Final System Architecture
By the end of the semester, we implemented the complete ChessInsight pipeline and front-end system described in our proposal and progress report. The final architecture consists of the following modules:
- **Data Ingestion and Curation:** Chunk-based PGN parser, PGN-to-parquet conversion, and time-control tagging that feed into downstream feature extraction.
- **Feature Engineering:** Game- and player-level features capturing time usage, position complexity, error profiles, opening style, and color asymmetry (e.g., white vs. black accuracy gaps). These are stored as `game_features.parquet` and `player_features.parquet`.
- **Modeling and Evaluation:** A family of skill-tier classifiers (Random Forest, XGBoost, and a soft-voting ensemble) and clustering pipelines (k-means and Birch) with saved artifacts for reproducible evaluation and visualization.
- **Visualization & Interaction:** A Streamlit dashboard and Dash prototype that load all artifacts from disk and expose them via linked interactive views. This separates heavy offline computation from lightweight, responsive front-end interaction.

All stages are coordinated by a single driver script, `run_analysis.py`, which can be re-executed on new datasets or synthetic data to regenerate the entire pipeline.

#### 7.2 Final Classification Performance
We evaluated three models on the held-out test set of **49,345** samples:
- **Random Forest (baseline):** Test accuracy ≈ **42.6%**, adjacent accuracy ≈ **55.6%**, macro F1 ≈ **0.425**.
- **XGBoost:** Slightly higher exact accuracy (≈ **44–45%**) and macro F1 than the Random Forest, particularly improving discrimination between Advanced and Expert tiers.
- **Soft-voting Ensemble (RF + XGBoost):** Achieved our best performance with **≈47% exact-tier accuracy**, **≈63% adjacent accuracy**, and macro F1 close to **0.46**, while retaining similar calibration as the individual learners.

The majority of errors remain between neighboring tiers, which suggests inherent ambiguity in discretizing Elo into four buckets. From a user perspective, adjacent accuracy and calibrated probabilities provide a more realistic picture of model reliability than raw 0/1 accuracy.

#### 7.3 Feature Importance and Behavioral Drivers
Across all models, **time-allocation features** remain the dominant predictors of skill:
- Middlegame average time, opening/endgame time balance, and time-trouble frequency together account for more than half of total feature importance.
- Engine-based accuracy measures (blunder rate, mistake rate, average centipawn loss) contribute additional signal but are less predictive than how players allocate their time.
- Opening repertoire richness (entropy over ECO families) and color-asymmetry features (white vs. black accuracy and time differences) add nuanced information, helping distinguish more "balanced" players from those who overperform with one color.

These findings reinforce prior cognitive science results that stronger players rely on rapid pattern recognition and maintain stable time management under pressure, while weaker players exhibit more volatile time usage and collapse more frequently in time trouble.

#### 7.4 Final Clustering and Player Archetypes
We finalized our clustering analysis using **Birch** with an effective cluster count of **5–6 archetypes**, chosen via internal metrics and qualitative inspection. The final Birch model achieved a **silhouette score ≈ 0.29** and **Davies–Bouldin index ≈ 1.02**, improving substantially over the baseline k-means configuration. Qualitative inspection surfaced the following representative archetypes:
- **Deliberate Strategists:** Higher-than-average Elo, high middlegame time, low time-trouble frequency, and low blunder rates.
- **Fast Tacticians:** Advanced/Expert players with lower time-per-move but acceptable error rates, often concentrated in Blitz/Bullet time controls.
- **Time Scramblers:** Intermediate players who frequently enter severe time trouble and exhibit high blunder rates in the endgame.
- **Opening Specialists:** Players with high opening-entropy and aggressive ECO codes who score well in the opening but see accuracy fall off in complex middlegames.
- **Swingy Improvers:** Players with high accuracy volatility and mixed results, often transitioning between tiers.

These archetypes are represented visually in the 2D embedding space derived from t-SNE over the PCA-reduced features, with color encoded either by cluster or skill tier.

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
- **Cluster Analysis Tab:** Summarizes cluster sizes, average Elo per cluster, tier composition within each archetype, and internal clustering metrics for k-means vs. Birch.

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
