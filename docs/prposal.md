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

---

# 2. Background

Sports betting markets attempt to estimate the expected outcome of a game before it is played. In NFL spread betting, sportsbooks publish a point spread that represents the expected margin between two teams. For example, if the home team is listed at -3.5, that means the home team is favored by 3.5 points and must win by at least 4 points to cover the spread. Because point spreads are influenced by team performance, injuries, location, rest, weather, and public market activity, they provide an interesting application for data science and machine learning.

This project focuses on predicting NFL game point differentials using historical game data, sportsbook betting lines, and advanced team performance metrics from the nflverse data ecosystem. Rather than only predicting whether a team will win or lose, the project will model the final margin of victory from the home team's perspective. The model's predicted margin can then be compared to the sportsbook spread line. If the model's predicted margin differs from the market's expected margin by more than a selected threshold, the game may be flagged as a possible simulated betting opportunity. If the difference is small, the model would recommend no bet.

This topic matters because it combines several major areas of data science: data collection, data cleaning, feature engineering, predictive modeling, sports analytics, and model evaluation. NFL data is especially useful for this project because it includes both game-level betting information and play-level data. The play-by-play dataset makes it possible to engineer advanced features such as expected points added, success rate, early-down passing efficiency, pressure rate, explosive play rate, and turnover rate.

The goal of this project is not to claim that a machine learning model can consistently beat sportsbooks. Instead, the goal is to evaluate whether advanced football metrics improve spread prediction compared with simpler baseline information such as the sportsbook line itself. The project will also evaluate whether different model-market difference thresholds would have produced stronger or weaker historical performance in a simulated betting backtest.

## Research Questions

1. Can machine learning models predict NFL game point differentials using historical betting lines, team information, and advanced play-by-play metrics?

2. Do advanced team performance features such as EPA per play, success rate, early-down passing efficiency, pressure rate, explosive play rate, and turnover rate improve point spread prediction?

3. How closely do model-predicted point spreads align with sportsbook closing spreads?

4. Can a threshold-based betting strategy identify games where the model's predicted spread differs enough from the sportsbook spread to justify a simulated bet?

5. Which betting thresholds produce the best historical results in terms of number of bets, win rate, profit/loss, and return on investment?

---

# 3. Data

This project uses NFL game, betting, and play-by-play data from the nflverse data ecosystem. The datasets were accessed directly from public nflverse GitHub release files and saved into the project's `data` folder. The project currently uses regular season data from the 2010 through 2025 NFL seasons.

## Data Sources

### Dataset 1: nflverse Schedule and Game Data

The schedule dataset was pulled from nflverse game data. This dataset contains one row per NFL game and includes information such as season, week, game type, game date, teams, final scores, rest days, moneylines, spread line, total line, weather, roof, surface, stadium, coaches, and referees.

Source used in the notebook:

`https://github.com/nflverse/nfldata/raw/master/data/games.csv`

Saved file:

`data/raw/nflverse_schedules_2010_2025.csv`

A cleaned version of the regular season game-level dataset was also saved as:

`data/processed/nflverse_games_cleaned_2010_2025.csv`

### Dataset 2: nflverse Play-by-Play Data

The play-by-play dataset was pulled from nflverse parquet files, one file per season. This dataset contains one row per NFL play and includes detailed play context, team identifiers, field position, down and distance, play type, yards gained, EPA, success, passing indicators, rushing indicators, pressure-related variables, turnovers, expected passing variables, and expected yards-after-catch variables.

Source pattern used in the notebook:

`https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet`

where `{season}` ranges from 2010 through 2025.

A selected-column version of the play-by-play dataset was created in the notebook using 71 relevant columns from the full 372-column play-by-play dataset.

### Dataset 3: Team-Game Advanced Statistics

The team-game advanced statistics dataset was engineered from the play-by-play data. Each row represents one team in one game. This dataset includes offensive and defensive efficiency features such as EPA per play, success rate, passing EPA, rushing EPA, early-down passing success, pressure allowed, pressure created, explosive play rate, turnover rate, and third/fourth down conversion rates.

Saved file:

`data/processed/team_game_advanced_stats_2010_2025.csv`

## Data Size and Shape

The exact file sizes will be confirmed after the data files are saved locally and uploaded to GitHub. The current notebook produced the following dataset shapes:

| Dataset | Rows | Columns | Row Meaning |
|---|---:|---:|---|
| Schedule dataset, 2010-2025 regular season | 4,175 | 46 | One row per NFL game |
| Cleaned completed regular season games dataset | 4,175 | 51 | One row per completed regular season NFL game with betting variables and target variables |
| Full play-by-play dataset | 770,337 | 372 | One row per NFL play |
| Selected play-by-play dataset | 770,337 | 71 | One row per NFL play using selected relevant columns |
| Team-game advanced stats dataset | 8,350 | 64 | One row per team per game |

## Time Period

The planned time period for this project is:

**2010 NFL season through 2025 NFL season**

The project currently focuses on regular season games.

## Unit of Analysis

The final machine learning modeling dataset will use:

**One row per NFL game**

Each row will contain game-level information, betting market information, home-team pregame advanced statistics, away-team pregame advanced statistics, and engineered difference features comparing the two teams.

The team-game advanced statistics dataset uses:

**One row per team per game**

This intermediate dataset will later be converted into pregame rolling averages so that the model only uses information available before each game.

## Data Dictionary

### Schedule and Game-Level Variables

| Column Name | Data Type | Definition | Potential Values |
|---|---|---|---|
| `game_id` | Object/String | Unique identifier for each NFL game | Example: `2024_01_BAL_KC` |
| `season` | Integer | NFL season year | 2010-2025 |
| `game_type` | Object/String | Type of game | `REG`, `POST` |
| `week` | Integer | NFL week number | 1-18 for recent regular seasons |
| `gameday` | Object/Date | Date the game was played | Calendar date |
| `weekday` | Object/String | Day of week | Sunday, Monday, Thursday, etc. |
| `gametime` | Object/String | Scheduled game time | Time value |
| `away_team` | Object/String | Away team abbreviation | BAL, KC, DAL, etc. |
| `home_team` | Object/String | Home team abbreviation | BAL, KC, DAL, etc. |
| `away_score` | Numeric | Away team's final points | Numeric score |
| `home_score` | Numeric | Home team's final points | Numeric score |
| `result` | Numeric | Home score minus away score from nflverse | Numeric margin |
| `total` | Numeric | Combined final points | Numeric total |
| `away_rest` | Numeric | Away team's rest days before the game | Numeric days |
| `home_rest` | Numeric | Home team's rest days before the game | Numeric days |
| `away_moneyline` | Numeric | Away team's moneyline odds | American odds |
| `home_moneyline` | Numeric | Home team's moneyline odds | American odds |
| `spread_line` | Numeric | Sportsbook spread line from the home team's perspective | Negative if home team favored, positive if home team underdog |
| `away_spread_odds` | Numeric | Odds for the away team against the spread | American odds |
| `home_spread_odds` | Numeric | Odds for the home team against the spread | American odds |
| `total_line` | Numeric | Sportsbook over/under total | Numeric total |
| `under_odds` | Numeric | Odds for the under | American odds |
| `over_odds` | Numeric | Odds for the over | American odds |
| `div_game` | Numeric/Binary | Indicator for divisional game | 0 or 1 |
| `roof` | Object/String | Stadium roof type | outdoors, dome, closed, open, retractable |
| `surface` | Object/String | Field surface | grass, turf, fieldturf, etc. |
| `temp` | Numeric | Game temperature | Numeric degrees |
| `wind` | Numeric | Game wind speed | Numeric miles per hour |
| `stadium` | Object/String | Stadium name | Stadium names |
| `home_qb_name` | Object/String | Starting home quarterback name | Player names |
| `away_qb_name` | Object/String | Starting away quarterback name | Player names |
| `home_coach` | Object/String | Home team's head coach | Coach names |
| `away_coach` | Object/String | Away team's head coach | Coach names |

### Engineered Game-Level Variables

| Column Name | Data Type | Definition | Potential Values |
|---|---|---|---|
| `home_margin` | Numeric | Home score minus away score. This is the primary ML target. | Positive, zero, or negative numeric value |
| `home_spread` | Numeric | Home-team perspective spread, copied from `spread_line` | Negative if home team favored, positive if underdog |
| `market_home_margin` | Numeric | Market's expected home margin, calculated as `-home_spread` | Numeric value |
| `spread_result` | Numeric | Home margin plus home spread. Used to evaluate cover result. | Positive, zero, or negative numeric value |
| `home_cover` | Binary | Whether the home team covered the spread | 1 = covered, 0 = did not cover |
| `push` | Binary | Whether the final result exactly matched the spread | 1 = push, 0 = not push |

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

## Machine Learning Target

The primary machine learning target variable will be:

| Target Column | Data Type | Definition |
|---|---|---|
| `home_margin` | Numeric | Home team's final point differential, calculated as `home_score - away_score` |

This target will allow the model to predict the expected point margin from the home team's perspective.

## Potential ML Features / Predictors

The following variables may be used as model features after being converted into pregame rolling averages:

- market_home_margin
- spread_line
- total_line
- home_rest
- away_rest
- rest_diff
- roof
- surface
- temp
- wind
- div_game
- Offensive EPA per play
- Defensive EPA allowed per play
- Passing EPA per play
- Rushing EPA per play
- Offensive success rate
- Defensive success rate allowed
- Early-down EPA
- Early-down pass EPA
- Early-down pass success rate
- Pressure allowed rate
- Pressure created rate
- Sack rate
- QB hit rate
- Explosive play rate
- Explosive allowed rate
- Turnover rate
- Turnover forced rate
- Pass rate
- Pass rate over expected
- Completion percentage over expected
- Third down conversion rate
- Fourth down conversion rate

The final modeling dataset will likely include home-team features, away-team features, and difference features. Examples of difference features include:

- `epa_diff = home_off_epa_per_play - away_off_epa_per_play`
- `success_rate_diff = home_off_success_rate - away_off_success_rate`
- `pass_epa_diff = home_pass_epa_per_play - away_pass_epa_per_play`
- `def_epa_diff = home_def_epa_allowed_per_play - away_def_epa_allowed_per_play`
- `pressure_diff = home_pressure_created_rate - away_pressure_created_rate`
- `explosive_play_diff = home_explosive_play_rate - away_explosive_play_rate`
- `turnover_diff = home_turnover_rate - away_turnover_rate`
- `rest_diff = home_rest - away_rest`
