# ChessInsight: Visual Exploration of Gameplay Dynamics
**Team 029 | <a id="A">(A)</a>Nareshkumar Prakash Kumar Jamnani, <a id="B">(B)</a>Pratik Narayan Gokhale, <a id="C">(C)</a>Maitreyi Mhaiskar, <a id="D">(D)</a>Kartik Dutt, <a id="E">(E)</a>Shashankkumar Shailendrakumar Mittal**

---

## 1. Introduction & Problem Statement

**What are we trying to do?**  
We will build an interactive visual analytics system that reveals how chess game patterns—time usage, move complexity, and blunders—differ across skill levels, and predicts player skill from game behavior.

**How is it done today?**  
The tools available today, such as Lichess analysis boards and Chess.com, focus on analyzing single games move by move using chess engines; however, they lack aggregated pattern analysis across thousands of games and cannot explain why players at different skill levels behave differently. Maia Chess [1](#ref-1) shows that human play can be modeled but does not have an interactive user interface. Research on skill prediction [2](#ref-2), [3](#ref-3) has achieved 83% accuracy in classifying game outcomes but does not reveal behavioral patterns in a visual manner.

**Who cares?**  
Millions of active chess players, chess coaches, analysts, researchers, scientists, and curious fans can benefit by exploring patterns in gameplay, identifying recurring mistakes, and understanding how players of different skill levels behave across various phases of the game.


## 2. Literature Survey

**Player Modeling & Skill Prediction**: McIlroy-Young et al. [1](#ref-1) introduced Maia Chess, training neural networks to predict moves at specific skill levels (1100–1900 Elo). Their Maia-2 work [4](#ref-4) extended this with skill-aware attention mechanisms. Kaushik et al. [2](#ref-2) used gradient boosting to classify game outcomes with 83% accuracy, while Sharma et al. [3](#ref-3) applied CNN–LSTM architectures achieving a mean absolute error of 182 rating points. These research papers provide strong modeling approaches but do not offer an interactive user interface for exploring patterns.  
**Time Pressure & Decision Making**: Van Harreveld et al. [5](#ref-5) showed that time pressure slows down deep thinking but keeps quick pattern recognition intact. Studies by Künn et al. [6](#ref-6) indicated that less thinking time leads players to make safer moves. These studies help us in designing our features, but both lack an interactive user interface.  
**Clustering & Playing Styles**: Drachen et al. [7](#ref-7) and Sifa et al. [8](#ref-8) use clustering algorithms to group players according to different playing styles. These approaches are applied to a variety of online and mobile games but remain unexplored in the game of chess. Challet and Maillet [9](#ref-9) use graph-based algorithms to score the complexities of opening moves of different chess players.  
**Behavioral Stylometry & Knowledge Graphs**: McIlroy-Young et al. [10](#ref-10) use transformers to identify individual players from game sequences, suggesting that player behavior can be determined from the moves made by the player, but this work lacks visual interpretability. Similarly, a 2025 study [11](#ref-11) created a behavior-based knowledge graph to predict the next moves of players.  
**Blunder Prediction & Time Management**: A recently published study [12](#ref-12) proposes a predictive model to predict blunders in chess, but its shortcoming is that it operates per game rather than across aggregated player populations. Guga et al. [13](#ref-13) train a machine learning model on features such as remaining time and position complexity to predict how long a player should think on each move. This directly influences our phase-wise time features, but they do not connect time behavior to skill tiers or blunders.  
**Visual Analytics for Chess**: Lu and Wang [14](#ref-14) proposed a visual interactive tool displaying how a single game evolves over time with linked views, which aligns with our goal to use visual analytics, but it focuses only on a single game at a time. García-Díaz and Mariscal-Quintero [15](#ref-15) propose visual tools to study and analyze elite players’ decisions, which aligns with our objective of understanding player behavior.


## 3. Proposed Approach & Innovation

**What’s new? Why will it succeed?**

**Algorithmic Innovation**  
1. **Skill Tier Classification** – We will train models that can infer which skill group a player belongs to using features such as time usage in different phases of the game, number of blunders made (after adjusting for move complexity), and engine scores evaluating the aggression of their opening moves.  
2. **Behavioral Pattern Clustering** – Using clustering algorithms on aggregated behavioral vectors, we will identify recurring player archetypes (for example, “time scrambler,” “positional grinder”). This represents one of the first systematic attempts to derive chess-specific behavioral clusters from large-scale data.

**Visualization Innovation**  
We will build an interactive dashboard that lets users:
- See a 2D map of players where nearby points have similar behavior.  
- View time-usage heatmaps in different phases of the game across different skill levels.  
- Explore an interactive network graph where users can filter by skill level to see which opening moves and lines are popular, risky, or safe in particular groups of players.


## 4. Impact & Measurement

**Impact**: Using this dashboard, players, coaches, and analysts will be able to see behavior patterns across thousands of games. Players can gain a better understanding of their strengths and weaknesses, and these insights can be used to create better training plans.  
**Measurement**  The improvement in rating, reduction in the number of blunders, and improvement in time-management skills of players can be used to determine whether the tool is successful in helping chess players.


## 5. Risks & Payoffs

**Risks**: Features may not sufficiently distinguish skill tiers, which may lead to poor classification performance. Clusters may be difficult to interpret, which would reduce their usefulness to coaches and players. Visualization complexity may hinder usability and overwhelm users.  
**Payoffs**: Players and coaches can see clear patterns in gameplay. Insights can be used to create targeted and efficient training plans.

## 6. Cost & Timeline

The project will incur no direct data cost because the Lichess dataset is free, and we will use our own machines and Georgia Tech GitHub for collaboration. If we decide to host our web application and machine learning models, additional infrastructure costs may be incurred.

**Timeline:** 8 weeks total (see Gantt chart below) starting from `2nd March 2026`.

### Gantt Chart

| Task                 | Member | W1  | W2  | W3  | W4  | W5  | W6  | W7  | W8  |
| -------------------- | ------ | --- | --- | --- | --- | --- | --- | --- | --- |
| Data & Features      | All    | ■   | ■   |     |     |     |     |     |     |
| Clustering           | [A](#A), [E](#E)   |     |     | ■   | ■   |     |     |     |     |
| Classification       | [C](#C), [D](#D)  |     |     | ■   | ■   |     |     |     |     |
| Visualization        | [B](#B)      |     |     |     | ■   | ■   | ■   |     |     |
| Integration & Report | All    |     |     |     |     |     | ■   | ■   | ■   |

**Milestones**  
Midterm: Working classifier with >50% accuracy plus initial clusters.  
Final: Deliver a complete interactive dashboard and reach at least 65% classification accuracy.  

**Effort Statement**: Member [A](#A) recorded the project proposal video, members [B](#B) and [C](#C) drafted the project proposal document and members [D](#D) and [E](#E) created the project proposal presentation slides. All team members contributed equally to the literature survey. 


## References

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
