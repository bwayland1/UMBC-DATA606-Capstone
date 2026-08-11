# 1. Title and Author

## Project Title

**Predicting NFL Point Spreads Using Machine Learning, Historical Betting Lines, and Advanced Team Performance Metrics**

## Prepared for

UMBC Data Science Master Degree Capstone by Dr. Chaojie (Jay) Wang

## Author Name

Ben Wayland

## Author's GitHub Repository

https://github.com/bwayland1/UMBC-DATA606-Capstone

## Author's LinkedIn Profile

https://www.linkedin.com/in/benjamin-wayland-4104701a3/

## PowerPoint Presentation

**To be added after the final presentation pieces are combined.**

## YouTube Presentation

**To be added after the final presentation video is recorded and uploaded.**

---

# 2. Background

Sports betting markets attempt to estimate the expected outcome of a game before it is played. In NFL point-spread betting, the sportsbook publishes a spread representing the expected scoring margin between the two teams. In the nflverse game data used in this project, `spread_line` aligns with the expected margin from the home-team perspective. A positive value indicates that the home team is expected to win by that many points, while a negative value indicates that the home team is the underdog.

This project uses machine learning to predict the final NFL scoring margin from the home team's perspective and compare that prediction with the sportsbook's expected margin. The model is not used to automatically bet every game. Instead, the difference between the model prediction and the sportsbook expectation is treated as a potential betting edge:

`model_edge = predicted_home_margin - market_home_margin`

A positive model edge indicates that the model is more favorable toward the home team than the sportsbook. A negative model edge indicates that the model is more favorable toward the away team. A simulated bet is placed only when the absolute model edge reaches a selected threshold.

The topic is useful for a data science capstone because it combines several parts of the analytics workflow: collecting public data, cleaning and merging multiple sources, engineering leakage-safe time-dependent features, exploratory data analysis, supervised machine learning, time-based model validation, model interpretation, and historical backtesting.

NFL play-by-play data also allows the project to move beyond traditional box-score statistics. Advanced team measurements such as Expected Points Added (EPA), success rate, passing and rushing efficiency, early-down performance, pressure rate, explosive-play rate, turnovers, and conversion rates can be calculated at the team-game level and transformed into pregame features.

The objective of this project is not to claim that a machine-learning system can consistently beat NFL sportsbooks. The sportsbook line is itself a strong prediction of the final margin. The project instead evaluates whether football-performance features can add useful information to the market baseline and whether the model is more effective when it is selective about which games are treated as betting opportunities.

## Research Questions

1. Can machine-learning models predict NFL final scoring margin using sportsbook information, game context, and advanced play-by-play performance metrics?

2. Which advanced team-performance variables are most useful for predicting final home margin?

3. How does machine-learning prediction accuracy compare with the sportsbook's expected margin?

4. Does reducing the feature set improve model performance compared with using the full high-dimensional dataset?

5. Can a threshold-based strategy identify games where model disagreement with the sportsbook has historically been more useful?

6. Which betting thresholds provide the strongest balance between number of bets, win rate, profit, and return on investment?

7. Are there specific game situations in which the model performs better, such as away-favorite games, certain spread ranges, particular parts of the season, or division versus non-division games?

---

# 3. Data

This project uses public NFL schedule, betting, and play-by-play data from the nflverse data ecosystem. The analysis covers completed NFL regular-season games from **2010 through 2025**.

The project begins with game-level schedule and betting information, combines that information with play-level data, engineers team-game advanced statistics, converts those statistics into leakage-safe pregame values, and finally creates one tidy row per NFL game for machine learning.

## Data Sources

### Dataset 1: nflverse Schedule and Game Data

The schedule dataset comes from the nflverse game data and contains one row per NFL game. It includes season, week, date, teams, final scores, rest days, sportsbook lines, moneylines, spread prices, totals, weather, stadium environment, quarterbacks, coaches, and other game context.

Source:

`https://github.com/nflverse/nfldata/raw/master/data/games.csv`

Saved files:

- `data/raw/nflverse_schedules_2010_2025.csv`
- `data/processed/nflverse_games_cleaned_2010_2025.csv`

### Dataset 2: nflverse Play-by-Play Data

The play-by-play data are loaded from nflverse parquet files, one season at a time.

Source pattern:

`https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet`

where `{season}` ranges from 2010 through 2025.

The full combined play-by-play data contain 372 columns. The EDA notebook selects 71 fields relevant to offensive and defensive efficiency, play context, passing/rushing behavior, pressure, turnovers, success, expected passing, and expected yards after catch.

### Dataset 3: Team-Game Advanced Statistics

The team-game advanced statistics dataset is engineered from the selected play-by-play data. Each row represents one team in one game.

Saved file:

`data/processed/team_game_advanced_stats_2010_2025.csv`

The resulting team-game data include offensive and defensive metrics such as EPA per play, success rate, passing and rushing EPA, early-down efficiency, pressure allowed and created, explosive plays, turnovers, and third/fourth-down conversion rates.

### Dataset 4: Final EDA Dataset

The final EDA dataset combines the game schedule/betting information with home-team and away-team pregame advanced statistics.

Saved file:

`data/processed/nfl_spread_eda_dataset_2010_2025.csv`

The final EDA dataset contains **4,175 unique games and 212 columns**, including **59 home-minus-away differential features**.

## Data Size and Shape

The executed notebooks produced the following shapes:

| Dataset | Rows | Columns | Row Meaning |
|---|---:|---:|---|
| Schedule dataset, 2010–2025 regular season | 4,175 | 46 | One row per NFL game |
| Cleaned completed regular-season games | 4,175 | 52 | One row per completed game with betting and context variables |
| Full play-by-play dataset | 770,337 | 372 | One row per NFL play |
| Selected play-by-play dataset | 770,337 | 71 | One row per NFL play using selected variables |
| Team-game advanced statistics | 8,350 | 64 | One row per team per game |
| Final EDA dataset | 4,175 | 212 | One row per NFL game |
| Model-ready dataset after removing Week 1 | 3,920 | 212 source columns | One row per NFL game |
| Initial ML feature matrix | 3,920 | 194 features | One row per NFL game used for modeling |

The notebooks did not record the exact on-disk MB/GB size of each CSV/parquet file. Those file sizes should be copied from the final GitHub repository or local project directory before final submission if required.

## Time Period

**2010 NFL season through 2025 NFL season**

The analysis is limited to regular-season games.

## Unit of Analysis

The final unit of analysis is:

**One row per NFL game**

Each game contains:

- game and schedule information,
- sportsbook information,
- home-team pregame advanced metrics,
- away-team pregame advanced metrics,
- home-minus-away differential features,
- contextual features such as roof, surface, rest, weather, and division status,
- the final target `home_margin`.

The intermediate team-game statistics dataset contains two rows per game, one for each team.

## Data Preparation and Leakage Prevention

A central requirement of the project is that a model prediction should use only information that would have been known before kickoff.

For each selected team statistic, the EDA notebook sorts each team's games chronologically and calculates a current-season expanding mean using previous games only:

`expanding().mean().shift(1)`

The `shift(1)` is critical because it prevents the current game's performance from being included in its own predictor values.

Week 1 is removed from the modeling dataset because no current-season prior games exist for those pregame rolling statistics. This reduces the modeling data from 4,175 to **3,920 games**.

Relocated franchise abbreviations were also standardized before merging:

- `OAK` → `LV`
- `SD` → `LAC`
- `STL` → `LA`

Playing-surface categories were consolidated so similar surface labels were not treated as separate categories.

## Machine Learning Target

The primary target is:

| Target Column | Data Type | Definition |
|---|---|---|
| `home_margin` | Numeric | Final home-team point differential: `home_score - away_score` |

The model is therefore a regression model that predicts the number of points by which the home team is expected to win or lose.

## Model Features / Predictors

After excluding identifiers, final scores, direct outcome variables, spread-result variables, spread prices, moneylines, and diagnostic columns, the modeling notebook begins with **194 candidate features**:

- **185 numeric features**
- **9 categorical features**

The categorical variables are:

- `roof`
- `stadium`
- `home_qb_id`
- `home_qb_name`
- `home_coach`
- `away_qb_id`
- `away_qb_name`
- `away_coach`
- `surface_clean`

The broad feature set includes:

- `market_home_margin`
- `total_line`
- home and away rest
- `div_game`
- temperature and wind
- stadium environment
- home and away quarterback/coach information
- offensive and defensive EPA
- success rate
- passing and rushing EPA
- yards per play
- early-down efficiency
- CPOE and expected passing
- pressure, sack, and quarterback-hit rates
- explosive-play rates
- turnover rates
- third/fourth-down rates
- home-minus-away differential features

The Random Forest feature screen shows that `market_home_margin` is the most important individual predictor by a wide margin, receiving approximately **37.3%** of total Random Forest importance.

The validation experiment identifies **Ridge Regression using the top 20 Random Forest-ranked features** as the strongest machine-learning configuration on the 2023 validation season.

One implementation detail is important for reproducibility: the later final holdout code selects `feature_importance_df.head(5)`. Therefore, the executed 2024–2025 final holdout model uses these five variables:

1. `market_home_margin`
2. `explosive_pass_rate_diff`
3. `def_pass_success_allowed_diff`
4. `home_offensive_plays`
5. `rush_success_rate_diff`

The historical walk-forward routine, by contrast, explicitly recalculates and selects the **top 20 features** using only the seasons available before each test year.

---

# 4. Exploratory Data Analysis (EDA)

EDA was performed in the Jupyter notebook:

`notebooks/01_nflverse_data_exploration.ipynb`

The goals of the EDA were to validate the data structure, understand the target and market variables, identify missingness and duplicates, examine relationships between advanced features and final margin, and produce a tidy leakage-safe dataset for modeling.

## Data Merging and Reshaping

The raw play-by-play data contain one row per play, while the final model requires one row per game.

The EDA workflow therefore:

1. loaded 770,337 play-by-play rows from 2010–2025,
2. selected relevant columns,
3. filtered to regular-season plays with valid offensive/defensive teams and EPA,
4. aggregated pass/run plays into offensive and defensive team-game statistics,
5. created 8,350 team-game rows,
6. calculated leakage-safe pregame expanding averages,
7. split the team-game statistics into home and away tables,
8. merged those values onto the 4,175-game schedule,
9. created 59 home-minus-away differential features.

The home and away merges preserved the 4,175 unique game rows, confirming that the joins did not duplicate observations.

## Tidy Final EDA Dataset

The final EDA dataset contains:

- **4,175 rows**
- **212 columns**
- **4,175 unique `game_id` values**
- **59 differential features**
- regular-season games from **2010 through 2025**

Each row represents one game and each column represents a game property, sportsbook variable, pregame team metric, or matchup differential.

## Duplicate Analysis

The final dataset contains:

- **0 fully duplicated rows**
- **0 duplicated `game_id` values**

This confirms that the final data remain tidy at the game level.

## Missing Values

Missing values are primarily structural rather than evidence of incorrect data.

Key findings:

- `temp` and `wind` are missing for **1,348 games (32.3%)** in the full EDA data.
- After Week 1 is removed, temperature and wind are each missing for **1,248 modeling games (31.8%)**.
- Many current-season pregame differential features are missing for **257 games (6.2%)** in the full EDA dataset because at least one team has not yet played a previous current-season game.
- Fourth-down conversion variables have approximately **10–13% missingness** in the full EDA because the rate can be undefined when no prior fourth-down attempts exist.

Week 1 is removed before modeling. Remaining numeric missing values are handled through median imputation, while categorical missing values are imputed with the most frequent category.

## Target Variable: `home_margin`

The target is:

`home_margin = home_score - away_score`

EDA results:

- Mean home margin: **+2.02 points**
- Median home margin: **+3 points**
- Standard deviation: **14.50 points**
- Minimum: **-49 points**
- Maximum: **+58 points**

The positive mean indicates a modest home advantage across the full sample, but the large standard deviation shows that individual NFL game margins are highly variable.

## Sportsbook Baseline

The sportsbook variable is:

`market_home_margin = spread_line`

The sportsbook expectation is strongly related to the final margin:

- Correlation between `market_home_margin` and `home_margin`: **0.435**
- Mean market expected home margin: approximately **+1.95 points**
- Mean actual home margin: **+2.02 points**

The sportsbook line is therefore an important baseline and also an important model feature.

## Spread Result and Cover Distribution

The spread residual is:

`spread_result = home_margin - market_home_margin`

Across the full EDA dataset:

- Mean `spread_result`: **+0.06 points**
- Median `spread_result`: **0**
- Home cover rate excluding pushes: **49.1%**
- Push rate: **2.5%**

These values indicate that the sportsbook market is close to balanced overall. There is little systematic tendency for either the home or away side to outperform the line across all games.

## Correlation Analysis

Final scores, direct outcome variables, cover/push variables, spread residuals, and sportsbook margin variables were excluded from the advanced-feature correlation ranking to avoid leakage and misleading relationships.

The strongest individual engineered correlations with `home_margin` include:

| Feature | Correlation with `home_margin` |
|---|---:|
| `off_epa_per_play_diff` | **0.295** |
| `off_success_rate_diff` | **0.278** |
| `pass_epa_per_play_diff` | **0.276** |
| `avg_xpass_diff` | **-0.267** |

The sportsbook's 0.435 correlation is stronger than any single engineered football feature. This supports using multiple advanced metrics together rather than expecting one statistic to outperform the market by itself.

## Roof and Surface Analysis

Roof categories are imbalanced:

- Outdoors: **2,949 games (70.6%)**
- Dome: **629 games (15.1%)**
- Closed roof: **524 games (12.6%)**
- Open roof: **73 games (1.7%)**

After surface cleaning:

- Grass: **2,341 games (56.1%)**
- Fieldturf: **1,526 games (36.6%)**
- Astroturf: **264 games (6.3%)**
- Missing: **44 games (1.1%)**

Average home margin differs by environment, but these are descriptive relationships rather than causal effects. Team strength, stadium, season, and schedule composition can all contribute to the observed differences.

## Division Games

The final data contain:

- **1,536 division games (36.8%)**
- **2,639 non-division games (63.2%)**

Average home margin:

- Division games: **+1.74 points**
- Non-division games: **+2.17 points**

Average spread result:

- Division games: approximately **-0.17 points**
- Non-division games: approximately **+0.19 points**

Division games are slightly closer in raw scoring margin, but the spread residuals remain near zero, suggesting that the market generally accounts for divisional context.

## EDA Conclusions

The main EDA findings are:

1. The sportsbook spread is a strong baseline but is not perfectly correlated with the final margin.
2. Offensive efficiency and passing-efficiency differentials are among the strongest advanced football relationships with final margin.
3. The betting market is nearly balanced overall, with home and away sides covering at roughly equal rates.
4. Missingness is largely explainable by early-season rolling statistics, weather availability, and sparse fourth-down opportunities.
5. The final dataset is tidy, contains no duplicate games, and is suitable for time-based predictive modeling.
6. Advanced features contain useful signals, but no single football metric approaches the predictive relationship of the sportsbook line by itself.

---

# 5. Model Training

Model development and backtesting were performed in:

`notebooks/03_modeling_and_backtesting_with_scenarios.ipynb`

## Development Environment and Python Packages

The project was developed primarily in **Google Colab**, with files stored in **Google Drive** and version-controlled through **GitHub**.

Primary Python tools include:

- `pandas`
- `numpy`
- `matplotlib`
- `plotly`
- `scikit-learn`
- `joblib`
- `pathlib`

Scikit-learn components include preprocessing pipelines, imputers, standardization, one-hot encoding, regression models, feature importance, and regression metrics.

## Model-Ready Dataset

After excluding Week 1:

- Rows: **3,920 games**
- Candidate features: **194**
- Numeric features: **185**
- Categorical features: **9**

Outcome variables, final scores, spread results, cover indicators, IDs, moneylines, and spread prices used for profit calculations are excluded from predictive features.

`market_home_margin` is intentionally retained because it is known before kickoff and provides the market's pregame expectation.

## Time-Based Train, Validation, and Test Split

A random train/test split was not used because that would allow later seasons to influence predictions of earlier games.

The primary split is:

| Dataset | Seasons | Games |
|---|---|---:|
| Training | 2010–2022 | 3,152 |
| Validation | 2023 | 256 |
| Test | 2024–2025 | 512 |

After model selection, the final holdout pipeline is trained on 2010–2023 and evaluated only on 2024–2025.

This design preserves chronological order and more closely represents real model deployment.

## Preprocessing

All transformations are performed inside scikit-learn pipelines.

### Numeric Features

- Median imputation
- Missing-value indicators with `add_indicator=True`
- Standard scaling

### Categorical Features

- Most-frequent-category imputation
- One-hot encoding
- `handle_unknown="ignore"` for categories not observed during training

Using pipelines ensures that preprocessing is learned only from the training data and then applied consistently to validation/test data.

## Candidate Models

Five regression models were compared:

1. Ridge Regression
2. Random Forest Regressor
3. K-Nearest Neighbors Regressor
4. Gradient Boosting Regressor
5. Histogram Gradient Boosting Regressor

The sportsbook baseline was also evaluated by treating `market_home_margin` as the predicted final margin.

## Evaluation Metrics

Prediction performance is measured using:

### Mean Absolute Error (MAE)

Average absolute difference between predicted and actual final margin.

Lower is better.

### Root Mean Squared Error (RMSE)

Similar to MAE but gives more weight to large misses.

Lower is better.

### R²

Measures the proportion of variation in final margin explained by the model.

Higher is better.

Betting performance is evaluated separately using:

- number of bets,
- wins,
- losses,
- pushes,
- win rate excluding pushes,
- simulated profit,
- total dollars risked,
- return on investment (ROI).

## Full-Feature Validation Results

Using all 194 candidate features on the 2023 validation season:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Sportsbook Baseline | **9.818** | **12.957** | **0.175** |
| Random Forest | 10.149 | 13.230 | 0.140 |
| Gradient Boosting | 10.443 | 13.598 | 0.091 |
| KNN Regressor | 10.572 | 13.648 | 0.084 |
| Hist Gradient Boosting | 10.637 | 13.864 | 0.055 |
| Ridge Regression | 12.715 | 15.924 | -0.247 |

Random Forest is the strongest full-feature ML model, but the sportsbook baseline is still better.

The poor all-feature Ridge result suggests that the 194-feature representation contains noise and multicollinearity that reduce the effectiveness of a regularized linear model.

## Feature Selection

A Random Forest model is used to rank feature importance. `market_home_margin` is the dominant variable, receiving approximately **37.3%** of total importance.

The model is then retested using the:

- Top 20 RF features
- Top 40 RF features
- Top 60 RF features

The best validation result is:

**Top 20 Random Forest features + Ridge Regression**

Validation performance:

- MAE: **10.030**
- RMSE: **13.060**
- R²: **0.162**

Feature reduction improves Ridge MAE from 12.715 to 10.030, showing that a smaller and more targeted feature set generalizes substantially better than the full Ridge model.

## Final 2024–2025 Holdout Test

As noted earlier, the executed final holdout code uses the first five Random Forest-ranked features when constructing the final Ridge pipeline.

The executed holdout results are:

| Method | MAE | RMSE | R² |
|---|---:|---:|---:|
| Final executed Ridge model | **9.774** | **12.465** | **0.256** |
| Sportsbook Baseline | 9.804 | 12.531 | 0.248 |

The model improves MAE by approximately **0.03 points per game**. The improvement is small, but the model also has slightly better RMSE and R² in the holdout.

This result reinforces an important distinction: the sportsbook remains very difficult to improve upon in overall margin prediction, so the project's betting analysis focuses on whether the model's strongest disagreements with the market are more useful than its average prediction.

## Betting Backtest

The model edge is:

`model_edge = predicted_home_margin - market_home_margin`

Decision rule:

- If `model_edge >= threshold`, bet the home team against the spread.
- If `model_edge <= -threshold`, bet the away team against the spread.
- Otherwise, place no bet.

Every simulated wager risks **$100**.

Winning profit is calculated from the actual American spread price. A loss is `-$100`, and a push returns `$0` profit.

ROI is calculated as:

`ROI = total_profit / total_risked`

## Walk-Forward Historical Testing

Because a two-season test set can contain relatively few high-edge bets, a season-by-season walk-forward evaluation is used to measure historical stability.

The walk-forward test covers **2011 through 2025**.

For each season:

1. only earlier seasons are used for training,
2. Random Forest feature importance is recalculated using only that training history,
3. the top 20 features are selected,
4. Ridge Regression is trained,
5. the next season is predicted,
6. multiple betting thresholds are evaluated.

This produces out-of-sample predictions for each historical test season without allowing future-season information into the model.

## Walk-Forward Threshold Results

| Edge Threshold | Total Bets | Win Rate Excluding Pushes | Total Profit | ROI |
|---|---:|---:|---:|---:|
| 0.5 | 2,803 | 51.1% | -$1,546.72 | -0.6% |
| 1.5 | 1,472 | 51.2% | -$66.45 | approximately 0.0% |
| 2.5 | 749 | **52.6%** | **+$2,077.15** | **2.8%** |
| 3.5 | 402 | **54.2%** | **+$2,364.81** | **5.9%** |
| 4.5 | 230 | **53.8%** | **+$1,260.60** | **5.5%** |

The result shows a clear selectivity-versus-volume tradeoff.

Small model-market disagreements do not generate profitable historical results. Profitability begins at the 2.5-point threshold, and the **3.5-point threshold produces the highest overall ROI** among the thresholds tested.

The 3.5-point threshold is profitable in **11 of 15 seasons (73.3%)**, compared with **9 of 15 seasons (60.0%)** for the 2.5-point threshold.

The average walk-forward season MAE is approximately **10.22 points**, with season MAE ranging from **8.774 in 2022** to **11.812 in 2014**.

## Situational Testing

The 2.5-point walk-forward threshold is also analyzed across different game situations to identify where model disagreement with the sportsbook has historically been most useful.

Situations include:

- division versus non-division games,
- home favorites versus away favorites,
- sportsbook spread ranges,
- sportsbook total ranges,
- early/middle/late season,
- home-side versus away-side bets,
- different model-edge ranges.

The scenario ranking requires at least **25 bets across at least 3 seasons** to reduce the influence of very small samples.

Selected findings:

| Situation | Bets | Win Rate | ROI |
|---|---:|---:|---:|
| Away favorites | 266 | **56.9%** | **10.4%** |
| Model edge 3.5–4.9 points | 223 | **55.0%** | **7.3%** |
| Weeks 11–14 | 120 | **55.2%** | **7.1%** |
| Market spread 10+ points | 92 | — | **5.7%** |
| Market spread 3–6.5 points | 358 | — | **5.6%** |
| Non-division games | 492 | — | **3.6%** |

The away-favorite category is the strongest large-sample scenario tested and is profitable in **11 of 15 walk-forward seasons**.

Division games produce approximately **1.2% ROI**, compared with **3.6% ROI** for non-division games.

An important result is that model MAE is still worse than sportsbook MAE in the major profitable scenarios. For example, away-favorite games have a model MAE of approximately **10.131** versus **10.033** for the sportsbook.

This demonstrates one of the main conclusions of the project: the model does not have to be the better overall margin predictor within every category to have possible value as a selective directional filter.

## Retrospective Final-Model Diagnostic

The final executed five-feature Ridge pipeline is also refit using all 2010–2025 data and applied back to each season.

This is intentionally an **in-sample retrospective diagnostic**, not a valid out-of-sample performance estimate.

At a 2.5-point threshold, the retrospective backcast produces:

- 106 bets
- 60 wins
- 41 losses
- 5 pushes
- approximately **+$1,699 profit**
- approximately **16.0% ROI**
- positive profit in **11 of 16 seasons**

Average yearly model MAE is approximately **10.095**, compared with **10.108** for the sportsbook.

Because the same seasons were used to train and evaluate this version, these results are expected to be optimistic. The walk-forward results are the more defensible historical performance estimate.

## Model Training Conclusions

The model-development phase produced several major findings:

1. The sportsbook line is an extremely strong predictor and baseline.
2. Feature reduction substantially improves Ridge Regression.
3. The best validation configuration is Ridge Regression using a reduced Random Forest-ranked feature set.
4. Overall predictive improvement over the sportsbook is small.
5. Betting every model disagreement is not successful.
6. Historical betting performance improves when the model is more selective.
7. A 3.5-point edge produces the strongest aggregate walk-forward ROI among the tested thresholds.
8. Situational analysis suggests that some game types—especially away-favorite games—may be better candidates for model-based screening than others.
9. Walk-forward testing is more informative than a single holdout or retrospective in-sample backcast when evaluating historical stability.

---

# 6. Application of the Trained Models

A Streamlit application is the next planned stage of the project.

**Application development has not yet been completed, so this section documents the intended role of the application without claiming functionality that has not yet been implemented.**

The application is expected to provide an interface through which a user can interact with the trained NFL spread-prediction workflow. The final design will be determined after the Streamlit implementation is completed.

Potential functionality includes:

- loading the saved trained model,
- displaying the model's predicted home margin,
- displaying the sportsbook expected home margin,
- calculating the model edge,
- showing whether the edge clears a selected betting threshold,
- presenting relevant model and game information in a simple interface.

The application should clearly distinguish a model prediction from a recommendation and should preserve the project's conclusion that the model is best treated as a selective analytical tool rather than a guaranteed betting system.

This report section should be updated after the Streamlit application is completed and deployed.

---

# 7. Conclusion

## Summary of the Work

This project developed an end-to-end NFL spread-prediction and backtesting workflow using public nflverse data from 2010 through 2025.

The project:

- collected schedule, sportsbook, and play-by-play data,
- engineered team-level offensive and defensive efficiency features,
- converted those statistics to pregame rolling values,
- created a tidy 4,175-game EDA dataset,
- produced 59 matchup differential features,
- performed exploratory analysis,
- compared multiple regression algorithms,
- used Random Forest importance for feature reduction,
- trained Ridge Regression models,
- compared ML predictions with the sportsbook baseline,
- simulated $100 spread bets using actual spread prices,
- performed season-by-season walk-forward testing,
- evaluated multiple betting thresholds,
- analyzed performance in different game situations.

## Answers to the Research Questions

### 1. Can machine learning predict NFL scoring margin?

Yes, the models explain some variation in final scoring margin and produce MAE around 10 points, but the task remains difficult because NFL outcomes contain substantial game-to-game variation.

### 2. Do advanced football metrics provide useful information?

Yes. EPA, offensive success, passing efficiency, defensive pass efficiency, explosive plays, and related matchup differentials appear among the stronger engineered variables.

However, no individual advanced metric is as strongly related to the final margin as the sportsbook baseline.

### 3. How does the model compare with the sportsbook?

The sportsbook is an extremely strong baseline.

On 2023 validation data, the sportsbook MAE is **9.818**, better than every full-feature machine-learning model.

The executed final Ridge holdout model produces **9.774 MAE** on 2024–2025, compared with **9.804** for the sportsbook. This is a very small improvement.

The main value of the ML workflow therefore appears to come from selective disagreement with the market rather than a large improvement in average game prediction.

### 4. Does feature selection help?

Yes.

All-feature Ridge Regression produces **12.715 validation MAE**, while Ridge using the top 20 Random Forest-ranked features improves to **10.030**.

This is one of the clearest modeling results in the project.

### 5. Does threshold-based betting help?

The walk-forward analysis suggests that the size of the model-market disagreement matters.

Thresholds of 0.5 and 1.5 points do not produce meaningful positive returns. At 2.5 points and above, the historical walk-forward results improve.

The strongest tested threshold is 3.5 points, which produces:

- 402 bets
- 54.2% win rate excluding pushes
- approximately +$2,365 profit
- 5.9% ROI
- profitable results in 11 of 15 historical test seasons

### 6. Where does the model appear to work best?

The strongest large-sample situation identified at the 2.5-point walk-forward threshold is **away-favorite games**:

- 266 bets
- 56.9% win rate
- 10.4% ROI
- profitable in 11 of 15 seasons

Other stronger groups include 3.5–4.9-point model edges and Weeks 11–14.

These findings should be treated as exploratory. They identify where future model development may be most useful rather than proving that those situations will remain profitable.

## Limitations

### Sportsbook Efficiency

The sportsbook line already incorporates a large amount of information. `market_home_margin` dominates Random Forest feature importance and is more strongly correlated with final margin than any single engineered football metric.

### Limited NFL Sample Size

Even 16 seasons contain only a few thousand games. Higher betting thresholds reduce the sample to a relatively small number of wagers, which can make ROI unstable.

### Historical Simulation Is Not Live Betting

The backtest assumes that the recorded spread line and price were available at the intended betting point. Real betting also involves line movement, timing, limits, and market availability.

### Player Availability and Injuries

The project does not currently contain a detailed injury model or an explicit quantitative quarterback-change adjustment.

### Opening Versus Closing Lines

The project uses the sportsbook line available in nflverse but does not model opening-to-closing movement.

### Model-Selection Risk

Testing multiple models, feature sets, thresholds, and situations creates a risk of finding patterns that occurred by chance. Walk-forward testing reduces this risk, but situational findings still require future confirmation.

### Final Holdout Feature-Count Implementation

The validation experiment identifies the top-20 Ridge configuration as best, while the executed final holdout code uses the first five ranked features. The report distinguishes these workflows so the results remain reproducible.

### Retrospective Backcast

The full-history backcast is in sample and should not be used as evidence of expected future profitability.

## Lessons Learned

1. **Preventing leakage is essential in sports prediction.** Rolling features must be shifted so the game being predicted never contributes to its own inputs.
2. **Time-based validation is more appropriate than random splitting.** Future seasons should not help predict earlier seasons.
3. **A strong benchmark matters.** The sportsbook itself is a strong predictive baseline.
4. **More features are not always better.** Feature reduction dramatically improved Ridge Regression.
5. **Prediction accuracy and betting profitability are different goals.**
6. **Sample size must be considered with ROI.**
7. **Model selectivity is important.** Larger model-market disagreements performed better than betting every small edge.

## Future Research and Improvements

1. Build and deploy the **Streamlit application**.
2. Add more detailed **quarterback and injury information**.
3. Add explicit starting-quarterback changes and player-availability adjustments.
4. Capture and model **opening versus closing spread movement**.
5. Test a model that directly predicts **probability of covering the spread**.
6. Calibrate cover probabilities and compare expected value rather than relying only on point-edge thresholds.
7. Continue testing whether away-favorite and other situational results remain stable on future seasons.
8. Standardize the final feature-selection rule so validation, holdout, and walk-forward workflows use the same number of selected features.
9. Update this report with the final Streamlit URL, application screenshots, presentation link, and YouTube link.

The overall conclusion is that the model is most useful as a **selective screening tool**. The sportsbook remains the strongest general benchmark, but the walk-forward analysis shows that larger model-market disagreements and certain game situations may contain more useful historical signals than betting every game.

---

# 8. References

1. nflverse. **NFL Game Data (`games.csv`)**.  
   https://github.com/nflverse/nfldata/raw/master/data/games.csv

2. nflverse. **NFL Play-by-Play Data Releases**.  
   https://github.com/nflverse/nflverse-data/releases

3. nflverse. **nflverse GitHub Organization**.  
   https://github.com/nflverse

4. scikit-learn. **Machine Learning in Python Documentation**.  
   https://scikit-learn.org/

5. pandas. **Python Data Analysis Library Documentation**.  
   https://pandas.pydata.org/docs/

6. NumPy. **Numerical Python Documentation**.  
   https://numpy.org/doc/

7. Plotly. **Plotly Express Documentation**.  
   https://plotly.com/python/plotly-express/

8. Streamlit. **Streamlit Documentation**.  
   https://docs.streamlit.io/

9. Wayland, Ben. **UMBC DATA606 Capstone GitHub Repository**.  
   https://github.com/bwayland1/UMBC-DATA606-Capstone

## Data Dictionary

### Schedule and Game-Level Variables

| Column Name | Data Type | Definition | Potential Values |
|---|---|---|---|
| `game_id` | Object/String | Unique identifier for each NFL game | Example: `2024_01_BAL_KC` |
| `season` | Integer | NFL season year | 2010–2025 |
| `game_type` | Object/String | Type of game | `REG`, `POST` |
| `week` | Integer | NFL week number | 1–18 for recent regular seasons |
| `gameday` | Object/Date | Date the game was played | Calendar date |
| `weekday` | Object/String | Day of week | Sunday, Monday, Thursday, etc. |
| `gametime` | Object/String | Scheduled game time | Time value |
| `away_team` | Object/String | Away-team abbreviation | BAL, KC, DAL, etc. |
| `home_team` | Object/String | Home-team abbreviation | BAL, KC, DAL, etc. |
| `away_score` | Numeric | Away team's final points | Numeric score |
| `home_score` | Numeric | Home team's final points | Numeric score |
| `result` | Numeric | Home score minus away score from nflverse | Numeric margin |
| `total` | Numeric | Combined final points | Numeric total |
| `away_rest` | Numeric | Away team's rest days before the game | Numeric days |
| `home_rest` | Numeric | Home team's rest days before the game | Numeric days |
| `away_moneyline` | Numeric | Away-team moneyline odds | American odds |
| `home_moneyline` | Numeric | Home-team moneyline odds | American odds |
| `spread_line` | Numeric | nflverse sportsbook expected margin from the home-team perspective | Positive = home favorite; negative = home underdog |
| `away_spread_odds` | Numeric | Price for the away team against the spread | American odds |
| `home_spread_odds` | Numeric | Price for the home team against the spread | American odds |
| `total_line` | Numeric | Sportsbook over/under total | Numeric total |
| `under_odds` | Numeric | Price for the under | American odds |
| `over_odds` | Numeric | Price for the over | American odds |
| `div_game` | Numeric/Binary | Indicator for divisional game | 0 or 1 |
| `roof` | Object/String | Stadium roof type | outdoors, dome, closed, open |
| `surface` | Object/String | Raw playing surface | grass, fieldturf, sportturf, etc. |
| `temp` | Numeric | Game temperature | Degrees Fahrenheit |
| `wind` | Numeric | Game wind speed | Miles per hour |
| `stadium` | Object/String | Stadium name | Stadium names |
| `home_qb_name` | Object/String | Starting home quarterback | Player names |
| `away_qb_name` | Object/String | Starting away quarterback | Player names |
| `home_coach` | Object/String | Home head coach | Coach names |
| `away_coach` | Object/String | Away head coach | Coach names |

### Engineered Game-Level Variables

| Column Name | Data Type | Definition | Potential Values |
|---|---|---|---|
| `home_margin` | Numeric | Final home point differential: `home_score - away_score` | Positive, zero, or negative |
| `market_home_margin` | Numeric | Sportsbook expected home margin; copied from nflverse `spread_line` | Positive = home expected to win; negative = away expected to win |
| `home_spread` | Numeric | Betting-ticket home spread: `-market_home_margin` | Negative for a home favorite |
| `spread_result` | Numeric | `home_margin - market_home_margin` | Positive = home covered; negative = away covered; 0 = push |
| `home_cover` | Binary | Whether the home team covered | 1 or 0 |
| `push` | Binary | Whether actual margin exactly equaled market expected margin | 1 or 0 |
| `surface_clean` | Object/String | Consolidated playing-surface category | grass, fieldturf, astroturf, missing |
| `*_diff` | Numeric | Home pregame value minus away pregame value | Numeric differential |

### Selected Play-by-Play Variables

| Column Name | Data Type | Definition | Potential Values |
|---|---|---|---|
| `play_id` | Numeric | Unique play identifier within game | Numeric ID |
| `game_id` | Object/String | Unique game identifier | Game ID |
| `season` | Integer | NFL season | 2010-2025 |
| `season_type` | Object/String | Season type | `REG`, `POST` |
| `week` | Integer | NFL week | 1-18 |
| `game_date` | Object/Date | Game date | Calendar date |
| `home_team` | Object/String | Home team abbreviation | NFL team codes |
| `away_team` | Object/String | Away team abbreviation | NFL team codes |
| `posteam` | Object/String | Team with possession | NFL team codes |
| `posteam_type` | Object/String | Whether possession team is home or away | home, away |
| `defteam` | Object/String | Defensive team | NFL team codes |
| `qtr` | Numeric | Game quarter | 1-5 |
| `down` | Numeric | Down number | 1, 2, 3, 4 |
| `ydstogo` | Numeric | Yards to go for first down | Numeric yards |
| `yardline_100` | Numeric | Distance from opponent end zone | 0-100 |
| `game_seconds_remaining` | Numeric | Seconds remaining in game | 0-3600+ |
| `score_differential` | Numeric | Possession team's score differential before play | Numeric value |
| `play_type` | Object/String | Type of play | pass, run, punt, field_goal, etc. |
| `qb_dropback` | Numeric/Binary | Whether the play was a quarterback dropback | 0 or 1 |
| `pass_attempt` | Numeric/Binary | Whether the play was a pass attempt | 0 or 1 |
| `rush_attempt` | Numeric/Binary | Whether the play was a rush attempt | 0 or 1 |
| `pass` | Numeric/Binary | nflverse pass indicator | 0 or 1 |
| `rush` | Numeric/Binary | nflverse rush indicator | 0 or 1 |
| `shotgun` | Numeric/Binary | Whether the offense lined up in shotgun | 0 or 1 |
| `no_huddle` | Numeric/Binary | Whether the offense used no-huddle | 0 or 1 |
| `qb_scramble` | Numeric/Binary | Whether the quarterback scrambled | 0 or 1 |
| `yards_gained` | Numeric | Yards gained on the play | Numeric yards |
| `air_yards` | Numeric | Air yards on pass attempt | Numeric yards |
| `yards_after_catch` | Numeric | Yards after catch | Numeric yards |
| `passing_yards` | Numeric | Passing yards credited on the play | Numeric yards |
| `rushing_yards` | Numeric | Rushing yards credited on the play | Numeric yards |
| `receiving_yards` | Numeric | Receiving yards credited on the play | Numeric yards |
| `ep` | Numeric | Expected points before the play | Numeric expected points |
| `epa` | Numeric | Expected points added by the play | Numeric expected points added |
| `qb_epa` | Numeric | EPA credited to the quarterback | Numeric expected points added |
| `air_epa` | Numeric | EPA from air yards component | Numeric expected points added |
| `yac_epa` | Numeric | EPA from yards-after-catch component | Numeric expected points added |
| `wp` | Numeric | Win probability before the play | 0 to 1 |
| `wpa` | Numeric | Win probability added by the play | Numeric value |
| `vegas_wpa` | Numeric | Vegas win probability added | Numeric value |
| `success` | Numeric/Binary | Whether the play was successful based on expected points | 0 or 1 |
| `first_down` | Numeric/Binary | Whether the play gained a first down | 0 or 1 |
| `series_success` | Numeric/Binary | Whether the offensive series was successful | 0 or 1 |
| `cp` | Numeric | Completion probability | 0 to 1 |
| `cpoe` | Numeric | Completion percentage over expected | Numeric value |
| `xpass` | Numeric | Expected pass probability | 0 to 1 |
| `pass_oe` | Numeric | Pass rate over expected | Numeric value |
| `xyac_epa` | Numeric | Expected yards-after-catch EPA | Numeric expected points added |
| `xyac_mean_yardage` | Numeric | Mean expected yards after catch | Numeric yards |
| `xyac_median_yardage` | Numeric | Median expected yards after catch | Numeric yards |
| `xyac_success` | Numeric | Expected YAC success probability | 0 to 1 |
| `xyac_fd` | Numeric | Expected first down probability after catch | 0 to 1 |
| `sack` | Numeric/Binary | Whether the play resulted in a sack | 0 or 1 |
| `qb_hit` | Numeric/Binary | Whether the quarterback was hit | 0 or 1 |
| `tackled_for_loss` | Numeric/Binary | Whether the ball carrier was tackled for a loss | 0 or 1 |
| `interception` | Numeric/Binary | Whether the play resulted in an interception | 0 or 1 |
| `fumble` | Numeric/Binary | Whether the play included a fumble | 0 or 1 |
| `fumble_lost` | Numeric/Binary | Whether the offense lost a fumble | 0 or 1 |
| `third_down_converted` | Numeric/Binary | Whether the play converted a third down | 0 or 1 |
| `third_down_failed` | Numeric/Binary | Whether the play failed on third down | 0 or 1 |
| `fourth_down_converted` | Numeric/Binary | Whether the play converted a fourth down | 0 or 1 |
| `fourth_down_failed` | Numeric/Binary | Whether the play failed on fourth down | 0 or 1 |
| `touchdown` | Numeric/Binary | Whether the play scored a touchdown | 0 or 1 |
| `pass_touchdown` | Numeric/Binary | Whether the play scored a passing touchdown | 0 or 1 |
| `rush_touchdown` | Numeric/Binary | Whether the play scored a rushing touchdown | 0 or 1 |

### Engineered Team-Game Advanced Statistics

| Column Name | Data Type | Definition |
|---|---|---|
| `offensive_plays` | Numeric | Number of offensive pass/run plays for the team in the game |
| `off_epa_per_play` | Numeric | Average offensive EPA per play |
| `off_success_rate` | Numeric | Average offensive success rate |
| `yards_per_play` | Numeric | Average yards gained per offensive play |
| `pass_rate` | Numeric | Share of offensive plays that were pass attempts |
| `pass_epa_per_play` | Numeric | Average EPA on pass attempts |
| `pass_success_rate` | Numeric | Success rate on pass attempts |
| `yards_per_pass` | Numeric | Average yards gained on pass attempts |
| `avg_air_yards` | Numeric | Average air yards on pass attempts |
| `avg_yac` | Numeric | Average yards after catch |
| `avg_cpoe` | Numeric | Average completion percentage over expected |
| `avg_xpass` | Numeric | Average expected pass probability |
| `avg_pass_oe` | Numeric | Average pass rate over expected |
| `rush_rate` | Numeric | Share of offensive plays that were rush attempts |
| `rush_epa_per_play` | Numeric | Average EPA on rush attempts |
| `rush_success_rate` | Numeric | Success rate on rush attempts |
| `yards_per_rush` | Numeric | Average yards gained on rush attempts |
| `early_down_pass_rate` | Numeric | Share of plays that were early-down passes |
| `early_down_pass_success_rate` | Numeric | Share of all offensive plays that were successful early-down passes |
| `early_down_epa_per_play` | Numeric | Average EPA on first and second down plays |
| `early_down_pass_epa` | Numeric | Average EPA on first and second down pass attempts |
| `pressure_allowed_rate` | Numeric | Rate of sacks or quarterback hits allowed |
| `sack_rate` | Numeric | Sack rate allowed by the offense |
| `qb_hit_rate_allowed` | Numeric | QB hit rate allowed by the offense |
| `negative_play_rate` | Numeric | Rate of negative plays, sacks, or tackles for loss |
| `turnover_rate` | Numeric | Rate of interceptions or lost fumbles by the offense |
| `explosive_play_rate` | Numeric | Rate of explosive plays by the offense |
| `explosive_pass_rate` | Numeric | Rate of pass plays gaining at least 20 yards |
| `explosive_rush_rate` | Numeric | Rate of rush plays gaining at least 10 yards |
| `third_down_conversion_rate` | Numeric | Third down conversion rate |
| `fourth_down_conversion_rate` | Numeric | Fourth down conversion rate |
| `avg_xyac_epa` | Numeric | Average expected YAC EPA |
| `avg_xyac_yards` | Numeric | Average expected yards after catch |
| `avg_xyac_success` | Numeric | Average expected YAC success |
| `defensive_plays` | Numeric | Number of defensive pass/run plays for the team in the game |
| `def_epa_allowed_per_play` | Numeric | Average EPA allowed per play by the defense |
| `def_success_rate_allowed` | Numeric | Success rate allowed by the defense |
| `def_yards_allowed_per_play` | Numeric | Average yards allowed per play |
| `def_pass_epa_allowed` | Numeric | Average EPA allowed on opponent pass attempts |
| `def_pass_success_allowed` | Numeric | Success rate allowed on opponent pass attempts |
| `def_yards_allowed_per_pass` | Numeric | Average yards allowed per opponent pass attempt |
| `def_rush_epa_allowed` | Numeric | Average EPA allowed on opponent rush attempts |
| `def_rush_success_allowed` | Numeric | Success rate allowed on opponent rush attempts |
| `def_yards_allowed_per_rush` | Numeric | Average yards allowed per opponent rush attempt |
| `def_early_down_epa_allowed` | Numeric | Average EPA allowed on first and second downs |
| `def_early_down_pass_epa_allowed` | Numeric | Average EPA allowed on early-down passes |
| `def_early_down_pass_success_allowed` | Numeric | Success rate allowed on early-down passes |
| `pressure_created_rate` | Numeric | Rate of sacks or QB hits created by the defense |
| `sack_created_rate` | Numeric | Defensive sack rate |
| `qb_hit_created_rate` | Numeric | Defensive QB hit rate |
| `tackle_for_loss_rate` | Numeric | Defensive tackle-for-loss rate |
| `explosive_allowed_rate` | Numeric | Rate of explosive plays allowed |
| `explosive_pass_allowed_rate` | Numeric | Rate of explosive passes allowed |
| `explosive_rush_allowed_rate` | Numeric | Rate of explosive rushes allowed |
| `turnover_forced_rate` | Numeric | Rate of interceptions or fumbles lost forced by the defense |
