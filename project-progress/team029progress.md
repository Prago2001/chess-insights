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

##### 4.1 Data Ingestion and Preprocessing
We ingest a Lichess PGN file containing **350,060** games and parse moves and clock times for both players. We identify **238,200** unique player handles in the raw PGNs and aggregate per-player feature vectors for **22,725** players who have sufficient game history. The full pipeline runs on a laptop in under ten minutes using cached chunk-based parsing.

##### 4.2 Feature Extraction
We extract **39 game-level features** and aggregate them to **30 player-level features** used by both the classifier and clustering modules. These include: 1)Average move times and time variance in opening, middlegame, and endgame; 2)Frequency of low-time moves and time-trouble episodes; 3)Engine-derived position complexity, centipawn loss, blunder and mistake rates; 4)Opening aggression, piece activity, and material imbalance frequencies.

##### 4.3 Skill-Tier Classification

We implement a **random forest classifier** that predicts four discrete skill tiers (Beginner, Intermediate, Advanced, Expert) from behavioral features. We construct a labeled dataset with **328,961** samples and **18** selected features, balanced across tiers using SMOTE in the training set. The data is split into **230,272** training, **49,344** validation, and **49,345** test samples. Class distribution before resampling is skewed toward Advanced and Intermediate players, but all four tiers are well represented. The fitted model, feature importance table, confusion matrix, and a summary of key metrics are saved for evaluation and reproducibility.

##### 4.4 Behavioral Clustering

We aggregate player-level features and run behavioral clustering to discover archetypes by following these steps: 1)Standardize features and apply PCA to 10 components, which explain about 88% of variance in the current run; 2)Evaluate k in the range 3–7 with k-means using silhouette, Calinski–Harabasz, and Davies–Bouldin indices; k = 5 achieves the best silhouette score among the tested values; 3)Run final k-means with **k = 5**, then compute t-SNE embeddings for 2D visualization.

##### 4.5 Visualization Assets

While the interactive dashboard is not yet implemented, we have generated several static visualizations that directly support the eventual UI:
- Overall skill-tier distribution plots.
- Time-usage heatmaps by game phase and tier.
- Confusion matrix and feature-importance bar charts for the classifier.
- 2D cluster embedding scatterplots and bar charts of cluster-level characteristics.
- The dashboard layout will contain the following: a behavioral map, a control panel for filters, and linked detail views for time usage and blunder statistics.

#### 5. Evaluation (Current Results and Planned Work)

##### 5.1 Skill-Tier Classification Performance

The current random forest baseline achieves:

- **Validation accuracy:** approximately 42.3%.
- **Test accuracy:** **42.6%** (exact tier).
- **Adjacent accuracy (±1 tier):** **55.6%**.
- **Macro F1:** approximately **0.425**.

Error analysis shows most mistakes occur between adjacent tiers (e.g., Intermediate vs. Advanced, Advanced vs. Expert), which is expected given fuzzy boundaries between skill levels. Beginners are recognized more reliably, with fewer predictions leaking into higher tiers.

**Planned evaluation work:**

- Compare random forests with gradient boosting (XGBoost or LightGBM) and possibly shallow neural networks using the same features [2, 3].
- Add calibration curves and top-k tier accuracy to better quantify prediction uncertainty.
- Conduct ablation studies to understand the contribution of time-based features [5, 6, 13] versus engine-based accuracy metrics [12].

##### 5.2 Clustering Quality and Interpretability

The current k-means model with k = 5 yields:

- **Silhouette score:** **0.21**.
- **Calinski–Harabasz index:** **3440.3**.
- **Davies–Bouldin index:** **1.35**.

We observe several archetypal patterns, such as an **Intermediate Deliberate Player** cluster and multiple **Advanced Fast Player** clusters with slightly different Elo bands and sizes, capturing the trade-off between speed and deliberation at higher skill levels [7, 8].


We also integrated a `compare_clustering_methods` helper that benchmarks k-means against Gaussian Mixture Models, hierarchical clustering, DBSCAN, and Birch on the same feature matrix. From the latest comparison, **Birch** achieves the highest silhouette score (≈0.29) and the lowest Davies–Bouldin index (≈1.02), outperforming k-means on these internal metrics, while k-means attains the highest Calinski–Harabasz score. We keep k-means as the primary pipeline method for now because it is simple and well-understood, and treat Birch as a promising alternative to explore further in the final phase.

**Planned evaluation work:**

- Use the comparison results and additional hyperparameter sweeps to decide whether Birch or another alternative offers better separation or interpretability than k-means [7, 8].
- Investigate cluster stability under resampling and feature perturbations.
- Validate interpretability via qualitative inspection and, if time permits, informal user feedback from chess-playing classmates [14, 15].

#### 6. Conclusions, Remaining Work, and Risks

##### 6.1 Summary of Progress

This report marks the midpoint of ChessInsight, where our goal was to build an analytical backend capable of processing chess game data and generating supervised and unsupervised skill models, which has been successfully achieved.
Using 350,060 annotated Lichess games (Elo 600–3,265), we generated 30 player-level features for 22,725 players with sufficient game history.
These features capture four behavioral dimensions from prior research: time allocation across phases, position complexity and material dynamics, error rates, and opening tendencies [5,6,9,12,13].

**Progress against stated midterm targets:**

We originally committed to a working classifier exceeding 50% accuracy and an initial clustering solution by the midpoint, with a final target of ≥65% accuracy and a complete interactive frontend by the semester's end. Our current standing against those goals is as follows:

- **Data pipeline and feature extraction** are complete and reproducible. The full feature matrix is available at both game level (39 features) and player level (30 features).

- **Skill-tier classification** using a Random Forest with 18 behavioral features achieves **42.6% exact-tier accuracy** and **55.6% adjacent-tier accuracy** (macro F1 = 0.425) on 49,345 test samples; although slightly below the 50% midterm target, the higher adjacent accuracy indicates errors mostly occur between neighboring tiers. To reach the **65% final goal**, we implemented a soft-voting ensemble model (XGBoost, Random Forest, Gradient Boosting) with interaction features, which is scheduled for evaluation in the next sprint.

- **Behavioral clustering** using k-means (k=5 on PCA-reduced features retaining 87.6% variance) identified five player archetypes (silhouette = 0.21, CH = 3440, DB = 1.35). A comparison across five algorithms showed Birch performed better (silhouette = 0.29, DB = 1.02), improving these metrics by 38% and 25%, so it will be adopted for the final clustering solution.

- **Interactive dashboard** remains the principal outstanding deliverable; the current output is a static matplotlib wireframe illustrating the intended layout.

**Key finding — time as the primary behavioral signal:** Feature importance analysis shows that phase-wise time statistics account for ~57% of total importance, with middlegame average time alone contributing 16.4%, supporting our hypothesis that time allocation across game phases is a stronger indicator of skill than blunders or openings [5,6,13]. Error-rate features still provide secondary signal but appear underweighted due to heuristic computation without full Stockfish evaluation [12].

**Key finding — speed vs. deliberation as the dominant clustering axis:** Across the five archetypes, the main divide is between deliberate players (higher think times and variance, avg Elo ≈1453) and fast players who move quickly and are concentrated in Advanced–Expert tiers (avg Elo ≈1706–1774). This pattern aligns with cognitive research suggesting stronger players rely more on rapid pattern recognition, while intermediate players deliberate longer [5,6].

##### 6.2 Plan for the Remaining Semester

The second half of the project has four clearly sequenced objectives:

**1. Improve classification performance.**
We will evaluate the ensemble classifier against the Random Forest baseline on the same test split using exact accuracy, adjacent accuracy, and macro F1, along with ablation experiments to measure the contribution of time-based features and interaction terms. If the ensemble fails to reach the 65% target, we will explore Elo regression as an alternative approach to avoid ambiguity in discrete skill-tier boundaries [2,3].

**2. Finalize and validate the clustering solution.**
We will adopt Birch as the primary clustering algorithm and evaluate archetype stability using bootstrap resampling; if clusters vary significantly, we will adjust the number of clusters or apply stronger regularization. We will also examine whether the overlapping “Advanced Fast Player” sub-clusters are explained by different time controls (Bullet, Blitz, Rapid) rather than intrinsic behavioral differences [7,8].

**3. Implement the interactive dashboard.**
The frontend will use pre-computed artifacts (t-SNE embeddings, cluster statistics, and feature importance) to render three key views—a behavioral map, a tier × phase time-usage heatmap, and an archetype comparison panel—allowing users to locate players in behavioral space, compare time-use patterns, and interpret archetype–skill relationships [14,15].

**4. Evaluate, document, and demo.**
We will report calibration curves and top-2 tier accuracy to evaluate model confidence and ordinal performance, validate archetype interpretability through feedback from chess-playing classmates, and—if time permits—implement a D3.js opening network linking ECO codes to archetypes [9].

##### 6.3 Risks and Mitigation Strategies

Three risks warrant active monitoring in the second half of the project:

**Classification accuracy ceiling.** Because Elo tier boundaries are inherently fuzzy (e.g., ratings like 1598 vs. 1602 fall into different tiers despite similar strength), exact-tier accuracy may have a practical ceiling, so we will evaluate the ensemble, explore improved feature engineering with engine evaluations, and, if necessary, shift to Elo regression or a coarser two-tier classification task [2,3].

**Quality of heuristic accuracy features.** Because centipawn loss and blunder rates are currently heuristic approximations without engine analysis, their predictive value may be limited, so we will assess whether running Stockfish on a stratified sample of games can improve feature quality and model performance within available compute time [12].

**Dashboard delivery timeline.** Since building a polished linked-view dashboard is the most time-intensive task, we will prioritize the three core views and treat additional visualizations as stretch goals, while using pre-computed artifacts so frontend development can proceed without rerunning the full pipeline [14,15].

#### 7. Effort Statement

All five team members have contributed roughly equal effort during the first half of the project, spanning literature review, data processing, model development, and documentation. Going forward, we anticipate a similar distribution, with individual focus areas shifting toward dashboard development, model evaluation, and final write-up as the semester progresses.

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
