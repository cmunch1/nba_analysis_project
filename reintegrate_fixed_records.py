#!/usr/bin/env python3
"""
Command-line wrapper for reintegrating fixed records.

Usage:
    python reintegrate_fixed_records.py
    OR
    uv run reintegrate_fixed_records.py
"""

from src.nba_app.data_processing.reintegrate_fixed_records import main

if __name__ == "__main__":
    main()
