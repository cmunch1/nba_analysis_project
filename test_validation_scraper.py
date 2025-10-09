#!/usr/bin/env python3
"""
Quick test script for ValidationScraper date-based approach.

Tests scraping a known date from basketball-reference.com to verify:
1. URL building works correctly
2. Scoreboard page parsing works
3. Ordinal positioning correctly identifies visitor/home teams
4. Team abbreviation extraction works
5. Score extraction works
"""

import sys
import logging
from datetime import datetime

# Add src to path
sys.path.insert(0, '/home/chris/projects/nba_analysis_project/src')

from nba_app.webscraping.di_container import DIContainer

def main():
    """Test validation scraper with a known date."""

    # Initialize DI container
    container = DIContainer()

    try:
        # Get dependencies
        config = container.config()
        app_logger = container.app_logger()
        app_logger.setup("test_validation_scraper.log")

        # Get validation scraper
        validation_scraper = container.validation_scraper()

        # Test with a known date that should have games
        # October 31, 2006 - First day of 2006-07 season (should have games)
        test_date = "10/31/2006"

        app_logger.structured_log(logging.INFO, "Testing validation scraper",
                                 test_date=test_date)

        print(f"\n=== Testing ValidationScraper with date: {test_date} ===\n")

        # Test URL building
        url = validation_scraper._build_date_url(test_date)
        print(f"1. Built URL: {url}")
        assert "month=10" in url
        assert "day=31" in url
        assert "year=2006" in url
        print("   ✓ URL format correct\n")

        # Test scraping games for this date
        print(f"2. Scraping games for {test_date}...")
        games = validation_scraper._scrape_games_for_date(test_date)

        if not games:
            print("   ✗ No games found!")
            app_logger.structured_log(logging.ERROR, "No games found for test date",
                                     test_date=test_date)
            return False

        print(f"   ✓ Found {len(games)} games\n")

        # Display first game details
        if games:
            first_game = games[0]
            print("3. First game details:")
            print(f"   Date: {first_game['DATE']}")
            print(f"   Visitor: {first_game['VISITOR_TEAM_ABBREV']} (ID: {first_game['VISITOR_TEAM_ID']}) - {first_game['VISITOR_SCORE']}")
            print(f"   Home: {first_game['HOME_TEAM_ABBREV']} (ID: {first_game['HOME_TEAM_ID']}) - {first_game['HOME_SCORE']}")
            print(f"   Basketball-Reference Game ID: {first_game['BBREF_GAME_ID']}")
            print(f"   ✓ Game parsed successfully\n")

            # Verify home team is at end of game ID
            expected_ending = first_game['HOME_TEAM_ABBREV']
            actual_ending = first_game['BBREF_GAME_ID'][-3:]

            if expected_ending.upper() == actual_ending.upper():
                print(f"4. Home team verification:")
                print(f"   Game ID ends with: {actual_ending}")
                print(f"   Home team abbrev: {expected_ending}")
                print(f"   ✓ Home team verification passed\n")
            else:
                print(f"4. Home team verification:")
                print(f"   ✗ Mismatch! Game ID: {first_game['BBREF_GAME_ID']}, Home: {expected_ending}")
                return False

        # Test full scraping and saving
        print(f"5. Testing full scrape_and_save_validation_data with [{test_date}]...")
        validation_scraper.scrape_and_save_validation_data([test_date])
        print(f"   ✓ Validation data saved to {config.validation_data_file}\n")

        # Read back and verify
        import pandas as pd
        from ml_framework.framework.data_access.base_data_access import BaseDataAccess

        data_access = container.data_access()
        validation_df = data_access.load_dataframe(config.validation_data_file)

        print(f"6. Verification of saved data:")
        print(f"   Records saved: {len(validation_df)}")
        print(f"   Columns: {list(validation_df.columns)}")
        print(f"   ✓ Data saved and loaded successfully\n")

        print("\n=== All tests passed! ===\n")

        app_logger.structured_log(logging.INFO, "Validation scraper test completed successfully",
                                 games_found=len(games),
                                 records_saved=len(validation_df))

        return True

    except Exception as e:
        print(f"\n✗ Test failed with error: {str(e)}\n")
        import traceback
        traceback.print_exc()

        if 'app_logger' in locals():
            app_logger.structured_log(logging.ERROR, "Validation scraper test failed",
                                     error_message=str(e),
                                     error_type=type(e).__name__)

        return False

    finally:
        # Close web driver
        try:
            web_driver = container.web_driver_factory()
            if web_driver:
                web_driver.close_driver()
        except:
            pass

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
