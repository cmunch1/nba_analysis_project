# Game ID Conversion - Implementation Summary

## ✅ Problem Solved

The ValidationScraper can now correctly convert NBA game IDs to basketball-reference.com URLs by using game metadata (date + home team) instead of algorithmic conversion.

## 🔧 Changes Made

### Files Modified

1. **[validation_scraper.py](../src/nba_app/webscraping/validation_scraper.py)**
   - Added `TEAM_ID_TO_ABBREV` reverse mapping dictionary
   - Changed `scrape_validation_data_for_games()` to accept `game_metadata: pd.DataFrame` instead of `game_ids: List[str]`
   - Updated `_scrape_single_game_validation()` to accept `game_id`, `game_date`, and `home_team_id` parameters
   - Implemented proper date conversion (MM/DD/YYYY → YYYYMMDD)
   - Implemented team ID to abbreviation lookup
   - Removed placeholder `_convert_nba_to_bbref_game_id()` method

2. **[nba_scraper.py](../src/nba_app/webscraping/nba_scraper.py)**
   - Added `import pandas as pd`
   - Changed `scrape_and_save_validation_data()` to accept `game_metadata: pd.DataFrame`

3. **[main.py](../src/nba_app/webscraping/main.py)**
   - Updated `scrape_validation_data()` function to:
     - Extract game metadata from scraped dataframes
     - Filter for home teams (those with "vs." in MATCH UP)
     - Combine and deduplicate game metadata
     - Pass properly formatted metadata to validation scraper

4. **[test_validation_scraper.py](../src/nba_app/webscraping/test_validation_scraper.py)**
   - Updated to use actual game metadata instead of just game IDs
   - Uses known games from 10/31/2006 for reliable testing

### Documentation Created

5. **[game_id_conversion_solution.md](game_id_conversion_solution.md)**
   - Comprehensive documentation of the conversion approach
   - Includes examples, edge cases, and troubleshooting

## 🎯 How It Works

### Previous Approach (Broken)
```python
# ❌ Could not convert NBA game ID alone to basketball-reference format
game_ids = ['20600001', '20600002']
validation_scraper.scrape_validation_data_for_games(game_ids)
# No way to know: What date? What home team?
```

### New Approach (Working)
```python
# ✅ Extract metadata from scraped data
game_metadata = pd.DataFrame([
    {
        'GAME_ID': '20600001',           # For tracking
        'GAME_DATE': '10/31/2006',       # For YYYYMMDD conversion
        'HOME_TEAM_ID': '1610612748'     # For team abbreviation lookup
    }
])

validation_scraper.scrape_validation_data_for_games(game_metadata)

# Internally converts to: https://www.basketball-reference.com/boxscores/200610310MIA.html
```

## 📊 Conversion Flow

```
NBA Scraped Data
    ↓
[GAME_ID: 20600001, MATCH UP: "MIA vs. CHI", GAME DATE: "10/31/2006", TEAM_ID: 1610612748]
    ↓
Extract home team (has "vs." in MATCH UP)
    ↓
[GAME_ID: 20600001, GAME_DATE: "10/31/2006", HOME_TEAM_ID: 1610612748]
    ↓
Convert date: "10/31/2006" → "20061031"
Convert team: 1610612748 → "MIA"
    ↓
Basketball-Reference Game ID: "200610310MIA"
    ↓
URL: https://www.basketball-reference.com/boxscores/200610310MIA.html
    ↓
Scrape validation data (true home/visitor designation)
```

## 🧪 Testing

### Run Test Script
```bash
cd /home/chris/projects/nba_analysis_project/src/nba_app/webscraping
python test_validation_scraper.py
```

### Expected Behavior
- ✅ Converts 3 sample games from 2006
- ✅ Constructs correct basketball-reference.com URLs
- ✅ Scrapes home team, visitor team, and scores
- ✅ Returns validation dataframe with proper structure

### Test Output Format
```
GAME_ID    HOME_TEAM_ID VISITOR_TEAM_ID HOME_SCORE VISITOR_SCORE SOURCE               SCRAPED_AT
20600001   1610612748   1610612741      66         108           basketball-reference 2025-10-05T...
```

## 🔑 Key Components

### Team ID Mappings
Located in `validation_scraper.py`:

```python
TEAM_ID_TO_ABBREV = {
    '1610612748': 'MIA',  # Miami Heat
    '1610612747': 'LAL',  # LA Lakers
    # ... 30 current teams
}

TEAM_ABBREV_TO_ID = {
    'MIA': '1610612748',
    'LAL': '1610612747',
    # ... reverse mapping
}
```

**Includes historical teams**:
- Seattle SuperSonics (SEA) → OKC Thunder
- New Jersey Nets (NJN) → Brooklyn Nets
- New Orleans Hornets (NOH) → Pelicans

### Metadata Extraction
From scraped boxscore data:
1. Filter rows where `MATCH UP` contains "vs." (home teams)
2. Extract: `GAME_ID`, `GAME DATE`, `TEAM_ID`
3. Deduplicate by `GAME_ID` (one row per game)
4. Pass to ValidationScraper

## ⚠️ Important Notes

### Date Format
- **Input**: MM/DD/YYYY (from NBA.com data)
- **Output**: YYYYMMDD (for basketball-reference URL)
- Example: `10/31/2006` → `20061031`

### Home Team Identification
Home team is identified by "vs." in the MATCH UP column:
- `"MIA vs. CHI"` → Miami is home
- `"CHI @ MIA"` → Miami is home (Chicago is visitor)

### Team Abbreviations
Use basketball-reference's abbreviations, not NBA's:
- ✅ `BRK` (Brooklyn Nets)
- ❌ `BKN` (NBA uses this)

### Doubleheaders
If two games on same date at same arena:
- First game: `200610310MIA.html`
- Second game: `200610311MIA.html`

Currently, we only scrape the first game. If it fails, it's logged and skipped.

## 🐛 Troubleshooting

### "Unknown home team ID" Warning
**Fix**: Add team ID to `TEAM_ID_TO_ABBREV` mapping in `validation_scraper.py`

### "Failed to load page" Warning
**Check**:
1. URL in logs - is it correct format?
2. Does the game exist on basketball-reference.com?
3. Is the team abbreviation correct?

### Empty Validation DataFrame
**Causes**:
- No home teams found in scraped data
- All scraping attempts failed
- Network issues

**Fix**: Check logs for specific errors per game

## 📋 Maintenance

### When Teams Relocate/Rename
1. Update `TEAM_ID_TO_ABBREV` with new abbreviation
2. Keep old abbreviation for historical games
3. Test with recent games

Example:
```python
# If Seattle returns as expansion team in 2026
TEAM_ID_TO_ABBREV = {
    '1610612760': 'OKC',  # Current OKC Thunder
    '1610612999': 'SEA',  # New Seattle team (hypothetical)
}

# Old Seattle games still use SEA → OKC mapping
```

## 🎉 Benefits Achieved

1. ✅ **No Complex Algorithm**: Don't need to reverse-engineer NBA game ID structure
2. ✅ **Reliable**: Uses actual game date and home team from NBA data
3. ✅ **Maintainable**: Simple team mapping dictionary
4. ✅ **Flexible**: Works for all game types (regular, playoffs, play-in)
5. ✅ **Testable**: Can test with known historical games

## 🔗 Related Documentation

- [Validation Scraper Implementation](validation_scraper_implementation.md) - Phase 1 overview
- [Game ID Conversion Solution](game_id_conversion_solution.md) - Detailed technical docs

## ✨ Next Steps

This completes **Phase 1** of the validation system. The scraper now works correctly.

**Phase 2** will implement:
- Data validation module to compare NBA.com vs basketball-reference data
- Automatic correction of home/visitor mismatches
- Correction logging and audit trail

See [validation_scraper_implementation.md](validation_scraper_implementation.md) for Phase 2 roadmap.
