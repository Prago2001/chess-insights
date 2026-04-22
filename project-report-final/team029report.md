### ChessInsight: Visual Exploration of Gameplay Dynamics
**Team 029 | <a id="A">(A)</a>Nareshkumar Prakash Kumar Jamnani, <a id="B">(B)</a>Pratik Narayan Gokhale, <a id="C">(C)</a>Maitreyi Mhaiskar, <a id="D">(D)</a>Kartik Dutt, <a id="E">(E)</a>Shashankkumar Shailendrakumar Mittal**

---

#### 1. Introduction

ChessInsight aims to build an interactive visual analytics system that reveals how chess gameplay patterns—such as time usage, blunders, and position complexity—vary across skill levels, and to predict a player's skill tier from their behavioral traces rather than raw Elo. Existing tools such as Lichess analysis boards and Chess.com focus on analyzing single games move-by-move using chess engines; they lack aggregated pattern analysis across thousands of games and cannot explain why players at different skill levels behave differently [1](#ref-1), [2](#ref-2), [3](#ref-3). Our system addresses this gap using data from the [Lichess dataset](https://database.lichess.org) through a pipeline of feature extraction, skill-tier classification, behavioral clustering, and an interactive web dashboard.

#### 2. Problem Definition

We address two tightly coupled problems:
1. **Behavior-aware skill inference.** Given a player's moves, clock times, and engine evaluations across many games, infer a discrete skill tier (Beginner, Intermediate, Advanced) that reflects *behavioral quality* rather than just rating [1](#ref-1), [2](#ref-2), [3](#ref-3).
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

In practice we run everything through a single driver script (`run_analysis.py`), which reloads cached game features when full PGN re-parsing is unnecessary, retrains the skill models, rebuilds clusters, refreshes plots, and drops the artifacts the dashboard expects—so a teammate can reproduce the full study without retracing manual steps.

#### 4.1 Data Ingestion and Preprocessing
Our working corpus is **350,060** rated Lichess games spanning **600–3,265** Elo. On machines without raw chunk files, we simply read the precomputed `game_features.parquet` and move straight to modeling. For player-level work we insist on at least **five** games per handle so that each profile is not dominated by a single lucky or awful outing; that filter leaves **44,613** players, with broader dataset context in Section 6.1.

#### 4.2 Feature Extraction
We start from **40** per-game signals—mostly clocks, phase timing, structural heuristics, and opening tendencies—and aggregate them into player-level summaries with a few simple engineered ratios (for example, how time is split across phases). The classifier sees **34** of those dimensions, chosen to capture how someone actually spends thinking time, how sharp or messy their typical positions are, and how adventurous their openings tend to be. Clustering deliberately uses a slimmer **11**-feature “playing style” slice (defined in `config.py`): mixing every correlated scale at once had produced one giant blob, so we keep the view that humans can still reason about.

#### 4.3 Skill-Tier Classification

Rather than treat rating as a continuous line, we bucket players into **three** tiers—Beginner (**Elo under 1400**), Intermediate (1400–1900), and Advanced (≥1900)—and ask the models to recover those buckets from behavior alone. After standardizing inputs, we oversample the training fold with **SMOTE** so rare tiers are not ignored, but we never touch validation or test data that way. We train a **Random Forest**, an **XGBoost** model, and a **soft-voting ensemble that averages only those two**; whichever wins on the held-out test set is written out as `skill_classifier.pkl` together with confusion matrices, feature importances, and the usual metric bundle. The stratified split gives **31,229** training players, **6,692** for validation, **6,692** for testing, with SMOTE expanding training rows to **44,082**.

#### 4.4 Behavioral Clustering
Once the style matrix is imputed, we first run a small sanity sweep: k-means for **k = 2…5** on the raw **11**-dimensional style vectors to watch how silhouette and Calinski–Harabasz move. **k = 2** looks best on paper, but it mostly separates “stronger” from “weaker” without telling a rich story, so we settle on **k = 3** archetypes. Every clustering result we report in the paper—including comparisons to other algorithms—first **standardizes** those features, projects them through **two principal components (roughly 42.6% of the variance)**, and only then clusters in that compressed space. We try **k-means, hierarchical clustering, Gaussian mixtures, Birch, and DBSCAN** side by side; **k-means with k = 3** remains our default because it balances strong Calinski–Harabasz scores with a Davies–Bouldin index that suggests reasonably tight, separated blobs among the methods that do not collapse into noise. For the map people actually explore in the app, we compute a **UMAP** layout (falling back to t-SNE if UMAP is missing) so nearby points feel like similar habits, not arbitrary PCA axes.

#### 4.5 Visualization Assets
Alongside the Streamlit app, the pipeline writes a wireframe and a set of static figures—tier distributions, time-by-phase heatmaps, confusion matrices, importances, cluster portraits, and accuracy-by-tier charts—under `CODE/visualizations/`. The live dashboard simply reads `player_features.parquet`, the saved embeddings, and the model summaries from disk, which keeps interaction snappy even on a laptop. A fuller walkthrough of each tab appears in **Section 7.5**.

#### 5. Evaluation

#### 5.1 Skill-Tier Classification Performance

The three models land in the same ballpark on exact-tier accuracy, but the ensemble edges ahead on the metrics we care about most once tiers are this coarse:

| Model                    | Val accuracy | Test accuracy | Adjacent (±1 tier) | Macro F1  |
| ------------------------ | ------------ | ------------- | ------------------ | --------- |
| Random Forest            | 64.1%        | 64.4%         | 82.4%              | 0.648     |
| XGBoost                  | 65.7%        | 65.0%         | 82.4%              | 0.655     |
| Soft ensemble (RF + XGB) | 65.7%        | **65.6%**     | **82.8%**          | **0.661** |

Reading the confusion matrices is reassuring in a quiet way: when the model is wrong, it is usually **off by one tier**, not predicting a beginner from someone who plays like a titled player. **Intermediate** remains the slipperiest label because it sits between two worlds rating-wise, which matches how humans argue about “club player” versus “serious amateur” in real life. Section 7.2 walks through per-class recall and baselines in more detail. Looking ahead, we could still explore calibration or ablations, but the head-to-head between forests, boosting, and the ensemble is already settled for this feature set.

#### 5.2 Clustering Quality and Interpretability

With **k-means (k = 3)** on the PCA-reduced style space we see a **silhouette near 0.34**, **Calinski–Harabasz near 26.6k**, and **Davies–Bouldin near 0.95**—modest silhouette, but the other indices suggest the partitions are not arbitrary. The three groups line up with the names we use in the interface: **Time Scramblers** (**17,581** players, average Elo **1748**) who live in the clock and still keep their rating up; **Tactical Battlers** (**10,855** players, **1400** average) who steer toward sharp, imbalanced fights; and **Positional Grinders** (**16,177** players, **1715** average) who prefer calmer, piece-active structures. Section 7.4 lays out the full algorithm horse race on the same representation; the **DBSCAN** line is worth a glance there—its silhouette looks almost too good until you notice how little global structure survives—so we treat it as a cautionary tale rather than a drop-in replacement. Classmates who tried the dashboard mostly reacted to these archetype stories; their comments are collected in **Section 7.6**.

#### 6. Summary of Achievements, Challenges, and Reflections

**Completed work.** All project goals were met. From 350,060 Lichess games (Elo 600–3,265) we extracted 34 behavioral features for 44,613 players. The soft-voting ensemble exceeded the ≥65% accuracy target (65.6% exact-tier, 82.8% adjacent, macro F1 = 0.661); k-means clustering (k=3) identified three interpretable archetypes (silhouette = 0.34, DB = 0.95); and the Streamlit dashboard was delivered with five interactive views driven entirely by precomputed artifacts.

**Challenges and mitigations.** Several risks from our proposal were encountered and resolved:
- **Tier boundary ambiguity:** The original four-tier schema produced noisy Expert–Advanced boundaries. Consolidating to three tiers sharpened class separation and improved model performance.
- **Feature sufficiency:** It was uncertain whether behavioral features alone could reach ≥65% accuracy without full-engine analysis. The ensemble result confirms that time-allocation and positional features are sufficient.
- **Scale and responsiveness:** Processing 350K games and serving a reactive UI on a laptop required decoupling heavy offline computation from the front end via saved Parquet/JSON artifacts loaded at startup.
- **Clustering method selection:** We ran five algorithms head-to-head and initially expected Birch or GMM to win given their flexibility, but k-means held up better on CH and DB and gave cleaner archetype stories that were easier to explain.

#### 7. Final Results and Visual Analytics System

#### 7.1 Final System Architecture
The system we ended up with has four main layers that hand off to each other. Raw PGN data goes through a chunk-based parser that tags time controls and converts records to Parquet — this was necessary to avoid re-reading gigabytes of text on every run. From there, feature extraction builds `game_features.parquet` and `player_features.parquet` covering timing, complexity, error profiles, opening tendencies, and white/black asymmetries. The modeling layer then reads those files directly: it trains the RF/XGBoost/ensemble classifiers and runs the k-means pipeline (compared against hierarchical, GMM, DBSCAN, and Birch), writing all results to disk as JSON and CSV artifacts. Finally, the Streamlit dashboard just reads those saved artifacts at startup — no retraining, no reclustering — which is what makes it fast enough to explore interactively on a laptop.

#### 7.2 Final Classification Performance
We evaluated three models on the held-out test set of **6,692** samples drawn from **44,613** players (those with ≥5 games), trained on 31,229 samples balanced to 44,082 via SMOTE across three tiers (Beginner, Intermediate, Advanced):

| Model                                    | Test Accuracy | Adjacent Accuracy (±1) | Macro F1  |
| ---------------------------------------- | ------------- | ---------------------- | --------- |
| Random Forest                            | **64.4%**     | **82.4%**              | **0.648** |
| XGBoost                                  | **65.0%**     | **82.4%**              | **0.655** |
| Soft-voting Ensemble (RF + XGBoost + GB) | **65.6%**     | **82.8%**              | **0.661** |

The ensemble **meets the ≥65% accuracy target from the proposal**. A naive baseline that always predicts Intermediate would score 47.1%, so the ensemble's 65.6% is a real 18-point gain. Breaking down by class, per-class recall is Beginner 69.1%, Advanced 66.8%, and Intermediate 63.1%. Intermediate is the toughest tier — it sits between the other two, so predictions naturally bleed in both directions. Looking at the confusion matrix, almost all wrong predictions land on a neighboring tier rather than jumping across, which is the expected failure mode for a three-tier boundary problem.

#### 7.3 Feature Importance and Behavioral Drivers
Looking at ensemble feature importances, `num_moves` (9.9%) and `avg_time_middlegame` (7.6%) top the list, followed by `avg_position_complexity` (6.4%) and `piece_activity_score` (5.6%). Several others — `material_imbalance_freq`, `time_variance_opening`, and `book_deviation_move` — each land in the 3–4.5% range. Taken together, time-related features make up roughly 30–35% of total importance, which is more than engine-derived blunder counts contribute. In practice, how a player manages their clock across the three game phases turns out to be a better skill signal than raw blunder frequency alone.

#### 7.4 Final Clustering and Player Archetypes
We compared five clustering algorithms (k-means, hierarchical, DBSCAN, GMM, Birch) on the same PCA-reduced feature matrix (k=3, 2 PCA components explaining 42.6% variance). Results are summarized below:

| Method       | Silhouette | Calinski–Harabasz | Davies–Bouldin |
| ------------ | ---------- | ----------------- | -------------- |
| **K-Means**  | **0.340**  | **26,564**        | **0.950**      |
| Hierarchical | 0.290      | 20,926            | 1.045          |
| GMM          | 0.321      | 22,276            | 1.033          |
| Birch        | 0.240      | 13,220            | 1.159          |
| DBSCAN       | 0.658†     | 339               | 3.142          |

†DBSCAN’s high silhouette is driven by a large noise cluster; its very low CH and high DB indicate poor global separation.

K-means at k=3 came out ahead on both Calinski–Harabasz (26,564) and Davies–Bouldin (0.950), so we adopted it as the final method. When we swept k from 2 to 5, k=2 actually gave the highest silhouette (0.361), but that solution just splits players into a rough high vs. low skill divide — not very useful. k=3 (silhouette=0.340) produced three groups that correspond to recognizable playing styles and still scores better than every other algorithm at k=3 on CH and DB. The three archetypes are:
- **Time Scramblers** (17,581 players, avg Elo 1748): Fast players who thrive under time pressure; highest average skill level.
- **Positional Grinders** (16,177 players, avg Elo 1715): Players who maintain active pieces in controlled, less complex positions; medium skill level.
- **Tactical Battlers** (10,855 players, avg Elo 1400): Players who seek complex tactical positions with material imbalances; still developing their skills.

All three groups are visualized in a 2D UMAP plot of the PCA-reduced features, where points can be colored by cluster label or skill tier.

#### 7.5 Final Visual Analytics Dashboard
The dashboard is built entirely around the precomputed artifacts — load times stay under a second even for the full 44K-player dataset. There are five tabs:
- **Overview Tab:** Basic numbers for the dataset as a whole: how many players per tier, the rating distribution, and total game counts — mostly useful as a sanity check and for orienting new users.
- **Player Cluster Map Tab:** 2D UMAP scatterplot that can be colored by cluster or tier. Users can filter by rating range or search a specific player to see their archetype and behavioral stats.
- **Time Analysis Tab:** Heatmaps of average time use broken down by game phase and tier, plus charts showing how often players hit time trouble.
- **Classification Tab:** The normalized confusion matrix, model metrics (accuracy, adjacent accuracy, macro F1), and a feature-importance bar chart.
- **Cluster Analysis Tab:** Cluster composition — sizes, average Elo, tier mix — and the algorithm comparison table explaining why k-means was chosen.

#### 7.6 Evaluation and User Feedback
We gathered informal feedback from a few chess-playing classmates across a wide rating range (roughly 900–2000 Elo). Most recognized their playing style in one of the three archetypes without much prompting, and the cluster map was easy to navigate. The time-usage charts prompted more discussion than expected — even non-technical users found it easy to spot differences between tiers. A few testers initially expected the tool to output a rating prediction; after seeing the confusion matrix they understood that it is profiling behavioral tendencies, not estimating Elo.

#### 8. Conclusions and Lessons Learned

#### 8.1 Summary of Contributions
At a high level, ChessInsight delivers three things: a data pipeline that goes from raw PGN files to a clean player-level feature set at scale; a modeling layer that classifies skill tiers and groups players into behavioral archetypes; and an interactive dashboard that ties it together for end users. The pipeline handles 350K+ games on a laptop without a GPU. The ensemble classifier cleared the 65% accuracy bar we set in the proposal. The clustering produced three archetypes that were recognizable to actual players. Together these form a self-contained, reproducible system that can be re-run on a new Lichess dump with a single script.

#### 8.2 Limitations and Future Work
There are a few things we know are imperfect. First, features like centipawn loss and blunder counts are heuristic — we approximate them without running a full engine on every position, which introduces noise. Second, collapsing a continuous Elo range into three discrete bins is always going to create fuzzy edges; a regression approach or soft tier probabilities would be more principled. Third, our user evaluation was small and informal, so usability claims should be taken loosely.

On the future-work side, the most useful extension would be longitudinal features — tracking how a player's behavior changes over time rather than just averaging across all their games. Adding an opening-network view that connects ECO codes to clusters, and a side-by-side player comparison mode in the dashboard, would also increase the tool's practical value.

#### 8.3 Effort Statement
All team members contributed substantially and comparably to the final system and report. Nareshkumar and Shashankkumar focused primarily on data ingestion, feature engineering, and clustering; Pratik led the dashboard design and implementation; Maitreyi and Kartik focused on classification modeling, error analysis, and documentation. Overall, effort distribution has been balanced, and all members have collaborated closely on design decisions and revisions to the report.

---

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
