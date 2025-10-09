#!/usr/bin/env python3
"""
Quick diagnostic test for ValidationScraper - checks config loading.
"""

import sys
sys.path.insert(0, '/home/chris/projects/nba_analysis_project/src')

from nba_app.webscraping.di_container import DIContainer

def main():
    """Test config loading."""

    container = DIContainer()

    # Get config
    config = container.config()

    # Check if validation_scraper_config attributes exist
    print("Checking config attributes:")
    print(f"1. scoreboard_base_url: {getattr(config, 'scoreboard_base_url', 'NOT FOUND')}")
    print(f"2. selectors: {getattr(config, 'selectors', 'NOT FOUND')}")

    if hasattr(config, 'selectors'):
        print(f"   - box_score_link: {getattr(config.selectors, 'box_score_link', 'NOT FOUND')}")
        print(f"   - team_links: {getattr(config.selectors, 'team_links', 'NOT FOUND')}")
        print(f"   - score_spans: {getattr(config.selectors, 'score_spans', 'NOT FOUND')}")

    print(f"3. parsing: {getattr(config, 'parsing', 'NOT FOUND')}")

    if hasattr(config, 'parsing'):
        print(f"   - verify_home_team_from_url: {getattr(config.parsing, 'verify_home_team_from_url', 'NOT FOUND')}")

    print(f"4. team_abbrev: {getattr(config, 'team_abbrev', 'NOT FOUND')}")

    if hasattr(config, 'team_abbrev'):
        print(f"   - url_path_index: {getattr(config.team_abbrev, 'url_path_index', 'NOT FOUND')}")

    print(f"5. error_handling: {getattr(config, 'error_handling', 'NOT FOUND')}")

    if hasattr(config, 'error_handling'):
        print(f"   - fail_on_verification_mismatch: {getattr(config.error_handling, 'fail_on_verification_mismatch', 'NOT FOUND')}")
        print(f"   - warn_on_unknown_teams: {getattr(config.error_handling, 'warn_on_unknown_teams', 'NOT FOUND')}")

    print(f"6. rate_limiting: {getattr(config, 'rate_limiting', 'NOT FOUND')}")

    if hasattr(config, 'rate_limiting'):
        print(f"   - delay_between_dates: {getattr(config.rate_limiting, 'delay_between_dates', 'NOT FOUND')}")

    print("\n✓ Config check complete")

if __name__ == "__main__":
    main()
