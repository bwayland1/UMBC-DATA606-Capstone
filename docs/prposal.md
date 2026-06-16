# 1. Title and Author

## Project Title

**Predicting NFL Point Spreads Using Machine Learning, Historical Betting Lines, and Advanced Team Performance Metrics**

## Prepared for

UMBC Data Science Master Degree Capstone by Dr. Chaojie (Jay) Wang

## Author Name

Ben Wayland

## Author's GitHub Repository

[Insert GitHub repository link here]

## Author's LinkedIn Profile

[Insert LinkedIn profile link here]

## PowerPoint Presentation File

[Insert PowerPoint presentation file link here]

## YouTube Video

[Insert YouTube video link here]

---

# 2. Background

Sports betting markets are designed to estimate the expected difference between two teams in a game. In NFL spread betting, sportsbooks publish a point spread that represents the number of points by which one team is favored. For example, if a team is listed as -3.5, that team must win by at least 4 points to cover the spread. Spread betting is especially interesting from a data science perspective because the sportsbook line already contains a large amount of public, expert, and market-based information.

This project focuses on whether machine learning models can predict NFL game point differentials using historical game data, betting lines, and advanced football performance metrics. Instead of simply predicting whether a team will win or lose, the project will train regression models to predict the final margin of victory from the home team's perspective. The model's predicted spread will then be compared against the sportsbook closing spread. If the model's predicted margin differs from the sportsbook spread by more than a chosen threshold, the game may be flagged as a potential bet. If the difference is small, the model will recommend no bet.

This topic matters because it combines sports analytics, predictive modeling, feature engineering, and market evaluation. NFL games generate rich structured data, including team performance, play-by-play results, expected points added, success rate, game location, weather, rest, and betting lines. A successful project must not only build a machine learning model, but also avoid data leakage by ensuring that only information available before each game is used as input.

The goal of this project is not to claim that machine learning can consistently beat sportsbooks. Instead, the purpose is to evaluate whether advanced football statistics such as expected points added, success rate, and recent team performance can improve point spread prediction beyond simpler baseline models.

## Research Questions

1. Can machine learning models predict NFL game point differentials using historical team performance, betting lines, and advanced statistics?

2. Do advanced metrics such as expected points added per play, success rate, passing EPA, rushing EPA, and defensive EPA improve model performance compared with traditional team statistics?

3. How closely do model-predicted point spreads align with sportsbook closing spreads?

4. Can a threshold-based betting strategy identify games where the model's predicted spread differs enough from the sportsbook spread to justify a simulated bet?

5. Which threshold levels produce the best historical performance in terms of win rate, number of bets, profit/loss, and return on investment?

---

# 3. Data

This project will use historical NFL game, betting, and play-by-play data. The final modeling dataset will be built at the game level, where each row represents one NFL game. The target variable will be the final home-team margin of victory.

## Data Sources

### Dataset 1: NFL Scores and Betting Data

The first data source is the Kaggle dataset **NFL scores and betting data**. This dataset contains historical NFL game results and betting information. According to the Kaggle dataset description, it includes NFL game results since 1966 and betting odds information since 1979. The dataset includes game-level information such as season, week, teams, final scores, favorite, spread, over/under, stadium, weather, and other descriptive game variables.

Source: Kaggle, “NFL scores and betting data”

### Dataset 2: nflverse / nflfastR Play-by-Play Data

The second data source is the nflverse ecosystem, especially `nflfastR` and `nflreadr`. `nflfastR` provides NFL play-by-play data and includes expected points added and win probability modeling. The `nflreadr` package provides access to precomputed nflverse datasets, including play-by-play, schedules, team statistics, player statistics, and data dictionaries.

This data will be used to create advanced team performance features before each game. Examples include offensive EPA per play, defensive EPA allowed per play, passing EPA, rushing EPA, success rate, and completion percentage over expected.

Source: nflverse / nflfastR / nflreadr

## Estimated Data Size

The exact size will be confirmed after downloading the datasets into the `data` subfolder. Based on the planned scope:

| Dataset                     |                                              Estimated Size | Notes                                                   |
| --------------------------- | ----------------------------------------------------------: | ------------------------------------------------------- |
| NFL scores and betting data |                                             Less than 10 MB | Game-level CSV dataset                                  |
| nflfastR play-by-play data  | Several hundred MB to multiple GB depending on seasons used | Play-by-play data contains one row per play             |
| Final modeling dataset      |                                             Less than 50 MB | Aggregated game-level dataset after feature engineering |

To keep the project manageable, the initial analysis may focus on the 2010 through 2024 NFL seasons. This provides a modern sample while avoiding major differences in older eras of NFL play.

## Estimated Data Shape

The exact data shape will be confirmed in the exploratory Jupyter Notebook.

| Dataset                     |                                Estimated Rows |                                     Estimated Columns | Row Meaning          |
| --------------------------- | --------------------------------------------: | ----------------------------------------------------: | -------------------- |
| NFL scores and betting data |                        Several thousand games |                           Approximately 15–25 columns | One row per NFL game |
| nflfastR play-by-play data  |                Hundreds of thousands of plays |                                   Hundreds of columns | One row per NFL play |
| Final modeling dataset      | Approximately 3,500–4,500 games for 2010–2024 | Approximately 30–80 columns after feature engineering | One row per NFL game |

## Time Period

The planned project time period is:

**2010 NFL season through 2024 NFL season**

This period will allow the project to use modern NFL data while still providing enough games for model training and evaluation.

## Unit of Analysis

The final modeling dataset will use:

**One row per NFL game**

Each row will include the teams, date, week, final score, sportsbook spread, sportsbook total, and engineered pregame features for the home and away teams.

## Target Variable

The primary target variable will be:

| Column Name   | Data Type | Definition                                                                                                                                                                          |
| ------------- | --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `home_margin` | Numeric   | Final point differential from the home team's perspective. Calculated as `home_score - away_score`. Positive values mean the home team won. Negative values mean the away team won. |

This target allows the project to train regression models that predict the expected final margin of victory.

A secondary target may be created for betting evaluation:

| Column Name  | Data Type | Definition                                                                                                                     |
| ------------ | --------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `home_cover` | Binary    | Indicates whether the home team covered the sportsbook spread. Calculated after comparing `home_margin` to the closing spread. |

## Potential Feature Variables

The following variables may be selected as features or predictors in the machine learning models.

### Game and Betting Features

| Column Name           | Data Type   | Definition                                                    | Potential Values                                                    |
| --------------------- | ----------- | ------------------------------------------------------------- | ------------------------------------------------------------------- |
| `season`              | Integer     | NFL season year                                               | 2010–2024                                                           |
| `week`                | Integer     | NFL week number                                               | 1–22 depending on regular season/playoffs                           |
| `home_team`           | Categorical | Home team abbreviation                                        | BAL, KC, DAL, etc.                                                  |
| `away_team`           | Categorical | Away team abbreviation                                        | BAL, KC, DAL, etc.                                                  |
| `spread_favorite`     | Numeric     | Sportsbook point spread for the favored team                  | Negative or positive numeric value                                  |
| `over_under_line`     | Numeric     | Sportsbook projected total points                             | Numeric value                                                       |
| `home_spread`         | Numeric     | Point spread from the home team's perspective                 | Negative if home team is favored, positive if home team is underdog |
| `market_home_margin`  | Numeric     | Sportsbook expected home margin, calculated as `-home_spread` | Numeric value                                                       |
| `stadium`             | Categorical | Stadium where the game was played                             | Stadium names                                                       |
| `roof`                | Categorical | Stadium roof type                                             | Outdoor, dome, retractable                                          |
| `surface`             | Categorical | Playing surface                                               | Grass, turf, fieldturf                                              |
| `weather_temperature` | Numeric     | Game-time temperature if available                            | Numeric value                                                       |
| `weather_wind_mph`    | Numeric     | Game-time wind speed if available                             | Numeric value                                                       |

### Traditional Team Performance Features

| Column Name                  | Data Type | Definition                                                  | Potential Values |
| ---------------------------- | --------- | ----------------------------------------------------------- | ---------------- |
| `home_win_pct_entering_game` | Numeric   | Home team's winning percentage before the game              | 0 to 1           |
| `away_win_pct_entering_game` | Numeric   | Away team's winning percentage before the game              | 0 to 1           |
| `home_points_scored_pg`      | Numeric   | Home team's average points scored per game before the game  | Numeric value    |
| `away_points_scored_pg`      | Numeric   | Away team's average points scored per game before the game  | Numeric value    |
| `home_points_allowed_pg`     | Numeric   | Home team's average points allowed per game before the game | Numeric value    |
| `away_points_allowed_pg`     | Numeric   | Away team's average points allowed per game before the game | Numeric value    |
| `home_turnover_margin`       | Numeric   | Home team's turnover margin before the game                 | Numeric value    |
| `away_turnover_margin`       | Numeric   | Away team's turnover margin before the game                 | Numeric value    |
| `rest_days_home`             | Numeric   | Number of days since the home team's previous game          | Numeric value    |
| `rest_days_away`             | Numeric   | Number of days since the away team's previous game          | Numeric value    |
| `rest_diff`                  | Numeric   | Home rest days minus away rest days                         | Numeric value    |

### Advanced Team Performance Features

| Column Name                     | Data Type | Definition                                                             | Potential Values |
| ------------------------------- | --------- | ---------------------------------------------------------------------- | ---------------- |
| `home_off_epa_per_play`         | Numeric   | Home team's offensive expected points added per play entering the game | Numeric value    |
| `away_off_epa_per_play`         | Numeric   | Away team's offensive expected points added per play entering the game | Numeric value    |
| `home_def_epa_per_play_allowed` | Numeric   | Home team's defensive EPA allowed per play entering the game           | Numeric value    |
| `away_def_epa_per_play_allowed` | Numeric   | Away team's defensive EPA allowed per play entering the game           | Numeric value    |
| `home_pass_epa_per_play`        | Numeric   | Home team's passing EPA per play entering the game                     | Numeric value    |
| `away_pass_epa_per_play`        | Numeric   | Away team's passing EPA per play entering the game                     | Numeric value    |
| `home_rush_epa_per_play`        | Numeric   | Home team's rushing EPA per play entering the game                     | Numeric value    |
| `away_rush_epa_per_play`        | Numeric   | Away team's rushing EPA per play entering the game                     | Numeric value    |
| `home_success_rate`             | Numeric   | Home team's offensive success rate entering the game                   | 0 to 1           |
| `away_success_rate`             | Numeric   | Away team's offensive success rate entering the game                   | 0 to 1           |
| `home_cpoe`                     | Numeric   | Home team's completion percentage over expected entering the game      | Numeric value    |
| `away_cpoe`                     | Numeric   | Away team's completion percentage over expected entering the game      | Numeric value    |
| `epa_diff`                      | Numeric   | Difference between home and away EPA-based performance                 | Numeric value    |
| `success_rate_diff`             | Numeric   | Difference between home and away success rate                          | Numeric value    |

## Data Preparation Plan

1. Download the NFL scores and betting data into the `data` folder.

2. Load nflfastR/nflverse play-by-play data for the selected seasons.

3. Clean team abbreviations and ensure that team names match across datasets.

4. Create the target variable `home_margin`.

5. Convert sportsbook spread into a home-team perspective variable called `home_spread`.

6. Create `market_home_margin` as the sportsbook expected home-team margin.

7. Aggregate play-by-play data into team-week statistics.

8. Use only data available before each game to avoid data leakage.

9. Merge home and away team statistics into one game-level modeling dataset.

10. Train regression models to predict `home_margin`.

11. Compare predicted home margin to sportsbook expected home margin.

12. Create a simulated betting signal using thresholds such as 1.5, 2.5, 3.5, and 4.5 points.

13. Evaluate model performance using both prediction metrics and betting simulation metrics.

## Machine Learning Target and Features

The main machine learning target will be:

`home_margin`

The main model output will be:

`predicted_home_margin`

The betting decision variable will be:

`model_edge = predicted_home_margin - market_home_margin`

Potential betting rule:

* If `model_edge > threshold`, bet the home team against the spread.
* If `model_edge < -threshold`, bet the away team against the spread.
* If the absolute value of `model_edge` is below the threshold, do not bet.

Potential thresholds include:

* 1.5 points
* 2.5 points
* 3.5 points
* 4.5 points

The final project will compare these thresholds by number of bets, win rate, profit/loss, and return on investment.
