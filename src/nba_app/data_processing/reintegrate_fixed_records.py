"""reintegrate_fixed_records.py

This script reintegrates manually fixed invalid records back into the main dataset.

Process:
1. Load the fixed team-centric records from teams_boxscores-fixed.csv
2. Update the Match Up fields in the cumulative scraped files based on fixed is_home_team values
3. Generate game-centric records from the fixed team records using merge_team_data()
4. Merge the fixed records back into teams_boxscores.csv and games_boxscores.csv
5. Clean up temporary files
"""

import sys
import traceback
import logging
import pandas as pd

from .di_container import DIContainer

LOG_FILE = "data_processing.log"


def update_scraped_matchup_fields(
    fixed_team_df: pd.DataFrame,
    data_access,
    app_logger,
    app_file_handler,
    config
) -> None:
    """
    Update the MATCH UP field in the cumulative scraped files based on fixed is_home_team values.

    Args:
        fixed_team_df: DataFrame with fixed team records
        data_access: Data access object for loading/saving scraped data
        app_logger: Logger instance
        app_file_handler: File handler instance
        config: Configuration object
    """
    app_logger.structured_log(logging.INFO, "Updating MATCH UP fields in scraped files",
                           fixed_records=len(fixed_team_df))

    # Load all scraped files
    scraped_dataframes, file_names = data_access.load_scraped_data(cumulative=True)

    # Create a mapping of (original_game_id, team_id) -> is_home_team
    # Use original_game_id to match with the scraped GAME_ID
    home_team_mapping = {}
    for _, row in fixed_team_df.iterrows():
        original_game_id = str(int(row['original_game_id']))  # Ensure string type, no decimals
        team_id = int(row['team_id'])  # Ensure int type
        is_home = int(row['is_home_team'])
        home_team_mapping[(original_game_id, team_id)] = is_home

    app_logger.structured_log(logging.INFO, "Created home team mapping",
                           mapping_size=len(home_team_mapping),
                           sample_keys=list(home_team_mapping.keys())[:3])

    # Update each scraped dataframe
    updated_count = 0
    for i, (df, file_name) in enumerate(zip(scraped_dataframes, file_names)):
        app_logger.structured_log(logging.INFO, f"Processing file {file_name}",
                               file_name=file_name,
                               row_count=len(df))

        # Find column names (case-insensitive)
        matchup_col = None
        team_col = None
        game_id_col = None
        team_id_col = None

        for col in df.columns:
            col_upper = col.upper().replace(' ', '').replace('_', '')
            if col_upper == 'MATCHUP':
                matchup_col = col
            elif col_upper == 'TEAM':
                team_col = col
            elif col_upper == 'GAMEID':
                game_id_col = col
            elif col_upper == 'TEAMID':
                team_id_col = col

        app_logger.structured_log(logging.INFO, f"Found columns in {file_name}",
                               matchup_col=matchup_col,
                               team_col=team_col,
                               game_id_col=game_id_col,
                               team_id_col=team_id_col)

        # Skip files that don't have required columns
        if matchup_col is None:
            app_logger.structured_log(logging.WARNING, "No MATCH UP column found in file, skipping",
                                   file_name=file_name)
            continue

        if team_col is None or game_id_col is None or team_id_col is None:
            app_logger.structured_log(logging.WARNING, "Missing required columns in file, skipping",
                                   file_name=file_name,
                                   has_team=team_col is not None,
                                   has_game_id=game_id_col is not None,
                                   has_team_id=team_id_col is not None)
            continue

        # Update MATCH UP field for records that were fixed
        for idx, row in df.iterrows():
            game_id = str(int(row[game_id_col]))  # Ensure string, no decimals
            team_id = int(row[team_id_col])  # Ensure int
            team_abbrev = row[team_col]

            key = (game_id, team_id)
            if key in home_team_mapping:
                is_home = home_team_mapping[key]

                # Parse the current matchup to get opponent
                current_matchup = row[matchup_col]
                if ' vs. ' in current_matchup:
                    opponent = current_matchup.split(' vs. ')[1]
                elif ' @ ' in current_matchup:
                    opponent = current_matchup.split(' @ ')[1]
                else:
                    app_logger.structured_log(logging.WARNING, "Could not parse matchup",
                                           matchup=current_matchup)
                    continue

                # Create new matchup based on fixed is_home_team
                if is_home == 1:
                    new_matchup = f"{team_abbrev} vs. {opponent}"
                else:
                    new_matchup = f"{team_abbrev} @ {opponent}"

                # Only update if it changed
                if new_matchup != current_matchup:
                    df.at[idx, matchup_col] = new_matchup
                    updated_count += 1
                    app_logger.structured_log(logging.INFO, "Updated MATCH UP field",
                                           game_id=game_id,
                                           team=team_abbrev,
                                           old_matchup=current_matchup,
                                           new_matchup=new_matchup)

        # Save the updated dataframe
        scraped_dataframes[i] = df

    app_logger.structured_log(logging.INFO, "Updated MATCH UP fields",
                           total_updates=updated_count)

    # Save updated scraped files
    data_access.save_dataframes(scraped_dataframes, file_names, cumulative=True)
    app_logger.structured_log(logging.INFO, "Saved updated scraped files")


def main() -> None:
    """
    Main function to reintegrate fixed records back into the main dataset.
    """
    container = DIContainer()
    app_logger = None
    error_handler = None

    try:
        # Initialize dependencies
        config = container.config()
        app_logger = container.app_logger()
        app_logger.setup(LOG_FILE)

        data_access = container.data_access()
        app_file_handler = container.app_file_handler()
        process_scraped_NBA_data = container.process_scraped_NBA_data()
        error_handler = container.error_handler_factory()

        app_logger.structured_log(
            logging.INFO,
            "Starting reintegration of fixed records",
            app_version=config.app_version,
            environment=config.environment
        )

        with app_logger.log_context(app_version=config.app_version, environment=config.environment):

            # 1. Load the fixed team-centric records
            fixed_team_file = app_file_handler.join_paths(
                config.processed_data_directory,
                "teams_boxscores-fixed.csv"
            )

            app_logger.structured_log(logging.INFO, "Loading fixed team records",
                                   file_path=fixed_team_file)
            fixed_team_df = app_file_handler.read_csv(fixed_team_file)

            app_logger.structured_log(logging.INFO, "Loaded fixed team records",
                                   record_count=len(fixed_team_df))

            # 2. Update MATCH UP fields in cumulative scraped files
            update_scraped_matchup_fields(
                fixed_team_df,
                data_access,
                app_logger,
                app_file_handler,
                config
            )

            # 3. Generate game-centric records from fixed team records
            app_logger.structured_log(logging.INFO, "Generating game-centric records from fixed team data")
            fixed_game_df = process_scraped_NBA_data.merge_team_data(fixed_team_df)

            app_logger.structured_log(logging.INFO, "Generated game-centric records",
                                   game_count=len(fixed_game_df))

            # Save the fixed game records for review
            fixed_game_file = app_file_handler.join_paths(
                config.processed_data_directory,
                "games_boxscores-fixed.csv"
            )
            app_file_handler.write_csv(fixed_game_df, fixed_game_file)
            app_logger.structured_log(logging.INFO, "Saved fixed game records",
                                   file_path=fixed_game_file)

            # 4. Merge fixed records back into main files
            app_logger.structured_log(logging.INFO, "Merging fixed records into main dataset")

            # Load current main files
            teams_file = app_file_handler.join_paths(
                config.processed_data_directory,
                config.team_centric_data_file
            )
            games_file = app_file_handler.join_paths(
                config.processed_data_directory,
                config.game_centric_data_file
            )

            teams_df = app_file_handler.read_csv(teams_file)
            games_df = app_file_handler.read_csv(games_file)

            app_logger.structured_log(logging.INFO, "Loaded main files",
                                   teams_records=len(teams_df),
                                   games_records=len(games_df))

            # Get the game_ids that were fixed (use game_id, not original_game_id)
            fixed_game_ids = fixed_team_df['game_id'].unique()

            app_logger.structured_log(logging.INFO, "Removing old versions of fixed games",
                                   fixed_game_ids=fixed_game_ids.tolist())

            # Remove old versions of fixed records
            teams_df = teams_df[~teams_df['game_id'].isin(fixed_game_ids)]
            games_df = games_df[~games_df['game_id'].isin(fixed_game_ids)]

            # Drop original_game_id from fixed_team_df if it exists (to match schema)
            if 'original_game_id' in fixed_team_df.columns:
                fixed_team_df = fixed_team_df.drop(columns=['original_game_id'])

            # Append fixed records
            teams_df = pd.concat([teams_df, fixed_team_df], ignore_index=True)
            games_df = pd.concat([games_df, fixed_game_df], ignore_index=True)

            # Sort by date and game_id
            teams_df[config.new_date_column] = pd.to_datetime(teams_df[config.new_date_column])
            teams_df = teams_df.sort_values(
                by=[config.new_date_column, config.new_game_id_column, config.home_team_column],
                ascending=[True, True, True]
            )

            games_df[config.new_date_column] = pd.to_datetime(games_df[config.new_date_column])
            games_df = games_df.sort_values(
                by=[config.new_date_column, config.new_game_id_column],
                ascending=[True, True]
            )

            app_logger.structured_log(logging.INFO, "Merged and sorted datasets",
                                   teams_records=len(teams_df),
                                   games_records=len(games_df))

            # Save updated main files
            app_file_handler.write_csv(teams_df, teams_file)
            app_file_handler.write_csv(games_df, games_file)

            app_logger.structured_log(logging.INFO, "Saved updated main files")

            # 5. Clean up temporary files (optional - keep for audit trail)
            app_logger.structured_log(logging.INFO, "Reintegration completed successfully",
                                   fixed_games=len(fixed_game_ids),
                                   final_teams_records=len(teams_df),
                                   final_games_records=len(games_df))

        app_logger.structured_log(logging.INFO, "Reintegration completed successfully")

    except Exception as e:
        if error_handler and app_logger:
            error_handler_obj = error_handler.create_error_handler(
                'data_processing',
                "Reintegration failed",
                error_message=str(e),
                traceback=traceback.format_exc()
            )
            app_logger.structured_log(
                logging.ERROR,
                f"{type(error_handler_obj).__name__} occurred",
                error_message=str(error_handler_obj),
                error_type=type(error_handler_obj).__name__,
                traceback=traceback.format_exc()
            )
        elif app_logger:
            app_logger.structured_log(
                logging.CRITICAL,
                "Unexpected error occurred",
                error_message=str(e),
                error_type=type(e).__name__,
                traceback=traceback.format_exc()
            )
        else:
            print(f"CRITICAL: Unexpected error occurred: {str(e)}")
            print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
