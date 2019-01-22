import pandas as pd

a = pd.read_csv("nba-players-stats/Players.csv")
b = pd.read_csv("nba-players-stats/Seasons_Stats.csv")
merged = b.merge(a, on='Player')

keep_col = ['Player', 'Tm', 'Year', 'Pos', 'Age', 'FTr', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%',
            'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'height',
            'weight']
filtered = merged[keep_col]
salaries = pd.read_csv("nba-players-stats/nba_salaries_1990_to_2018.csv")
filtered = filtered.merge(salaries, on=['Player', 'Tm'])
keep_col += ["season_end", "season_start", "salary"]
filtered = filtered[keep_col]
viewNum = filtered["Year"] == filtered["season_start"]
filtered = filtered.loc[viewNum]
filtered.sort_values('Year', inplace=True)
filtered.to_csv("output.csv", index=False)
