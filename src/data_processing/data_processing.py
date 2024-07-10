
import pandas as pd



def merge_scraped_dataframes(scraped_dataframes: list) -> pd.DataFrame:
    """
    Merges the scraped dataframes into a single dataframe by first standardizing the column names and then merging on GAME_ID and TEAM_ID.

    Args:
        scraped_dataframes (list): the list of scraped dataframes

    Returns:
        the merged dataframe with duplicated columns removed

    """

    # adjust column names so that they are all in standard format
    for df in scraped_dataframes:
        if 'TEAM' in df.columns:
            df = df.rename(columns={'TEAM':'Team', 'MATCH UP':'Match Up', 'GAME DATE':'Game Date'})
        df.columns = df.columns.str.replace('\xa0', ' ') # weird web text artifact in some column names

    # merge all the dataframes, marking any columns that are duplicates with a suffix of '_dupe' so we can drop them later
    for i, df in enumerate(scraped_dataframes):
        if i == 0:
            merged = df
        else:
            merged = pd.merge(merged, df, on=['GAME_ID', 'TEAM_ID'],suffixes=('', '_dupe'))

    # drop the duplicate columns
    merged = merged.drop(columns=merged.filter(regex='_dupe').columns)

    return merged


def process_nans(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the non-numeric data. These are rare cases where a game was postponed or cancelled or the team did not attempt a free throw.

    Args:
        df (pd.DataFrame): the df dataframe

    Returns:
        the cleaned dataframe

    """
    
    # drop rows where W/L is null (this represents a postponed or cancelled game and is very rare and even rarer that it is recorded in the data)
    df = df.dropna(subset=['W/L'])

    # Convert every field that has % in the field name to numeric.
    # These are calculated fields, and though very rarely, sometimes the divisor is 0 and NBA.com puts a dash mark '-' in the calculated field.
    # 
    # e.g. FT% (free throw percentage is calculated as FT/FTA (free throws made / free throws attempted)
    # On the rare occasion that the team did not attempt a free throw, the field FT% (calculated 0/0) is marked as '-'
    # 
    # Since these are so EXTREMELY rare (only 1 case in NBA history as of 2024), we are going to just convert to 75% (~ league average) 
    # for these cases so that we can make the field numeric.
    # Had there been more of these cases, it MIGHT be worth investigating a better way to handle these cases or better way to impute the data
    # or maybe use feature engineering to mediate the impact of these cases (maybe create a new field with a flag for these cases,
    # or a new field that is FT% * FT to "weight" the FT% field by the number of free throws made, etc.).

    for col in df.columns:
        if '%' in col:
            df[col] = pd.to_numeric(df[col], errors='coerce') 
            df[col] = df[col].fillna(75)

    return df


def rename_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:

    """
    Renames the columns in the dataframe to make them more readable and to standardize the naming convention.

    Args:
        df (pd.DataFrame): the dataframe to rename the columns for

    Returns:
        the dataframe with renamed columns and a dictionary mapping the original column names to the new column names

    """

    original_columns = df.columns.tolist()

    # convert to lower case and replace spaces with underscores
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    # replace the % sign with pct_ to make the field names easier to work with since % is a special character in many programming languages
    df.columns = df.columns.str.replace('%', 'pct_')

    # remove the underscores that show up at the end of the column names
    df.columns = df.columns.str.replace('_$', '')

    # save a dictionary of the original column names to the new column names
    column_mapping = dict(zip(original_columns, df.columns.tolist()))

    return df, column_mapping


def extract_new_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts information from the existing columns to create new columns
    This includes: 
        flagging if the team won the game, 
        flagging if the team was the home team, 
        flagging if the game went into overtime, 
        flagging if the game was a playoff game,
        and extracting the season from the GAME_ID.

    Args:
        df (pd.DataFrame): the games dataframe

    Returns:
        the dataframe with new columns extracted

    """
    
    # convert 'W/L' to numeric and rename it
    df["is_win"] = df["W/L"].str.contains("W").astype(int)
    df = df.drop(columns=['W/L'])

    # flag the home team
    df["is_home_team"] = df["Match Up"].str.contains("vs.").astype(int)

    # flag if the game went into overtime
    df["is_overtime"] = (df["MIN"] > 240).astype(int)

    # The first digit of the GAME_ID denotes whether the game was played in the regular season (2) or the playoffs (4) or play-in (5)
    # The second and third digits denote the season (e.g. 21 for the 2021-2022 season)
    # To make it easier to extract this info, first let's convert GAME_ID to a string
    df["GAME_ID"] = df["GAME_ID"].astype(str)

    df["season"] = df["GAME_ID"].str[1:3].astype(int) + 2000

    df["is_playoff"] = (df["GAME_ID"].str[0].astype(int) > 2).astype(int)

    return df


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorders the columns in the dataframe for better readability and to keep related columns together in logical groupings.
        - game info
        - team info
        - team stats for that game

    Args:
        df (pd.DataFrame): the dataframe to reorder the columns for

    Returns:
        the dataframe with reordered columns

    """
    
    all_columns = games.columns.tolist()

    game_info = ["GAME_ID", "SEASON", "Game Date", "PLAYOFF", "OVERTIME", "MIN",]
    team_info = ["TEAM_ID", "HOME_TEAM", "Team", "Match Up"] 
    team_stats = [col for col in all_columns if col not in game_info + team_info]

    games = games[game_info + team_info + team_stats]
    
    return df       
    





def add_TARGET(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a TARGET column to the dataframe by copying HOME_TEAM_WINS.

    Args:
        df (pd.DataFrame): the dataframe to add the TARGET column to

    Returns:
        the games dataframe with a TARGET column

    """

    df['TARGET'] = df['HOME_TEAM_WINS']
    
    return df


def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataframe into train and test sets.

    Splits the latest season as the test set and the rest as the train set.
    The second latest season included with the test set to allow for feature engineering.

    Args:
        df (pd.DataFrame): the dataframe to split

    Returns:
        the train and test dataframes

    """

    latest_season = df['SEASON'].unique().max()

    train = df[df['SEASON'] < (latest_season)]
    test = df[df['SEASON'] >= (latest_season - 1)]
    
    return train, test

