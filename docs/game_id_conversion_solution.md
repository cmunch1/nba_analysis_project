# Game ID Conversion Solution

## Problem
Basketball-reference.com and NBA.com use different game ID formats, making it challenging to scrape validation data from basketball-reference for games identified by NBA game IDs.

## ID Format Comparison

### NBA.com Game ID Format
**Format**: `SSYYMGGGGG` (8-10 digits)
- **S**: Season type (2=regular, 3=playoffs, 4=preseason, 5=play-in)
- **YY**: Season year (06 = 2006-07 season)
- **M**: Month/game group indicator
- **GGGGG**: Sequential game number

**Examples**:
- `20600001` - First regular season game of 2006-07
- `52300121` - Play-in tournament game #121 from 2023-24 season

### Basketball-Reference Game ID Format
**Format**: `YYYYMMDD0HHH`
- **YYYYMMDD**: Game date (e.g., 20061031 for October 31, 2006)
- **0**: Separator
- **HHH**: Home team abbreviation (e.g., MIA, LAL, BOS)

**Examples**:
- `200610310MIA` - Game on Oct 31, 2006 with Miami as home team
- `200610310LAL` - Game on Oct 31, 2006 with LA Lakers as home team

**URL Structure**:
```
https://www.basketball-reference.com/boxscores/{YYYYMMDD0HHH}.html
```

## Solution: Metadata-Based Conversion

Instead of trying to algorithmically convert NBA game IDs to basketball-reference format, we **extract metadata from the scraped data** and construct the basketball-reference ID.

### Required Metadata
From the NBA.com scraped data, we extract:
1. **GAME_ID**: NBA game ID (for tracking)
2. **GAME_DATE**: Game date in MM/DD/YYYY format
3. **HOME_TEAM_ID**: NBA team ID of the home team (identified by "vs." in MATCHUP column)

### Conversion Process

```python
def convert_to_bbref_game_id(game_date: str, home_team_id: str) -> str:
    """
    Convert game metadata to basketball-reference game ID.

    Args:
        game_date: Date in MM/DD/YYYY format (e.g., '10/31/2006')
        home_team_id: NBA team ID (e.g., '1610612748')

    Returns:
        Basketball-reference game ID (e.g., '200610310MIA')
    """
    # 1. Convert date from MM/DD/YYYY to YYYYMMDD
    from datetime import datetime
    date_obj = datetime.strptime(game_date, '%m/%d/%Y')
    date_str = date_obj.strftime('%Y%m%d')

    # 2. Convert NBA team ID to basketball-reference abbreviation
    home_team_abbrev = TEAM_ID_TO_ABBREV[home_team_id]  # e.g., 'MIA'

    # 3. Construct basketball-reference game ID
    bbref_game_id = f"{date_str}0{home_team_abbrev}"  # e.g., '200610310MIA'

    return bbref_game_id
```

### Team ID Mapping

We maintain a bidirectional mapping between NBA team IDs and basketball-reference abbreviations:

```python
# NBA Team ID → Basketball-Reference Abbreviation
TEAM_ID_TO_ABBREV = {
    '1610612737': 'ATL',  # Atlanta Hawks
    '1610612738': 'BOS',  # Boston Celtics
    '1610612741': 'CHI',  # Chicago Bulls
    '1610612747': 'LAL',  # LA Lakers
    '1610612748': 'MIA',  # Miami Heat
    # ... etc
}

# Basketball-Reference Abbreviation → NBA Team ID
TEAM_ABBREV_TO_ID = {
    'ATL': '1610612737',
    'BOS': '1610612738',
    # ... (reverse of above)
}
```

## Implementation in ValidationScraper

### Method Signature Change

**Before** (problematic):
```python
def scrape_validation_data_for_games(self, game_ids: List[str]) -> pd.DataFrame:
    # Couldn't convert game_ids alone to basketball-reference format
```

**After** (solution):
```python
def scrape_validation_data_for_games(self, game_metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Args:
        game_metadata: DataFrame with columns:
            - GAME_ID: NBA game ID
            - GAME_DATE: Game date (MM/DD/YYYY)
            - HOME_TEAM_ID: Home team NBA ID
    """
```

### Metadata Extraction in main.py

The webscraping pipeline now extracts metadata from scraped boxscore data:

```python
def scrape_validation_data(nba_scraper, newly_scraped, config, logger):
    game_metadata_list = []

    for df in newly_scraped:
        # Filter to only home teams (those with "vs." in matchup)
        home_games = df[df['MATCH UP'].str.contains('vs.', case=False, na=False)].copy()

        if not home_games.empty:
            game_metadata_list.append(
                home_games[['GAME_ID', 'GAME DATE', 'TEAM_ID']]
            )

    # Combine and deduplicate
    game_metadata = pd.concat(game_metadata_list, ignore_index=True)
    game_metadata = game_metadata.drop_duplicates(subset=['GAME_ID'])

    # Rename to standard format
    game_metadata = game_metadata.rename(columns={
        'GAME_ID': 'GAME_ID',
        'GAME DATE': 'GAME_DATE',
        'TEAM_ID': 'HOME_TEAM_ID'
    })

    # Pass to validation scraper
    nba_scraper.scrape_and_save_validation_data(game_metadata)
```

## Example Workflow

### Input Data (from NBA.com)
```
GAME_ID    | MATCH UP      | GAME DATE   | TEAM_ID
20600001   | CHI @ MIA     | 10/31/2006  | 1610612741
20600001   | MIA vs. CHI   | 10/31/2006  | 1610612748
```

### Step 1: Extract Home Team
Filter for "vs." in MATCH UP → `1610612748` (Miami Heat)

### Step 2: Build Metadata
```python
{
    'GAME_ID': '20600001',
    'GAME_DATE': '10/31/2006',
    'HOME_TEAM_ID': '1610612748'
}
```

### Step 3: Convert to Basketball-Reference ID
```python
date: '10/31/2006' → '20061031'
team: '1610612748' → 'MIA'
bbref_id: '200610310MIA'
```

### Step 4: Construct URL
```
https://www.basketball-reference.com/boxscores/200610310MIA.html
```

### Step 5: Scrape and Validate
Extract authoritative home/visitor designation from basketball-reference.com

## Advantages of This Approach

1. **No Complex Conversion Logic**: We don't need to reverse-engineer NBA's game ID algorithm
2. **Reliable**: Uses actual game date and home team from NBA data
3. **Maintainable**: Only need to keep team mappings up-to-date
4. **Flexible**: Works for all game types (regular season, playoffs, etc.)
5. **Self-Documenting**: Metadata approach makes the process transparent

## Handling Edge Cases

### Multiple Games on Same Date at Same Arena
Basketball-reference handles this by adding a suffix:
- `200610310MIA.html` - First game
- `200610311MIA.html` - Second game (doubleheader)

Our approach: We scrape the first game. If it fails, basketball-reference may return 404. This is logged and the game is skipped.

### Historical Team Relocations
The team mapping includes historical teams:
```python
TEAM_ABBREV_TO_ID = {
    'SEA': '1610612760',  # Seattle SuperSonics → OKC Thunder
    'NJN': '1610612751',  # New Jersey Nets → Brooklyn Nets
    'NOH': '1610612740',  # New Orleans Hornets → Pelicans
    # etc.
}
```

**Important**: Use the team abbreviation **at the time of the game**, not current abbreviation.

### Future Team Changes
When teams relocate or change names:
1. Update both `TEAM_ID_TO_ABBREV` and `TEAM_ABBREV_TO_ID` mappings
2. Add historical mapping for old games
3. Use current abbreviation for new games

## Testing

### Manual Test
```bash
cd src/nba_app/webscraping
python test_validation_scraper.py
```

This tests with known games from 2006 that definitely exist on basketball-reference.com.

### Expected Output
```
✓ Successfully scraped validation data for 3 games

Validation data:
     GAME_ID HOME_TEAM_ID VISITOR_TEAM_ID HOME_SCORE VISITOR_SCORE                    SOURCE                 SCRAPED_AT
0  20600001   1610612748      1610612741         66           108  basketball-reference  2025-10-05T12:34:56.789
1  20600002   1610612747      1610612756        114           106  basketball-reference  2025-10-05T12:34:58.123
2  20600003   1610612755      1610612741         92           104  basketball-reference  2025-10-05T12:35:00.456
```

## Troubleshooting

### "Unknown home team ID" Warning
**Cause**: Team ID not in `TEAM_ID_TO_ABBREV` mapping
**Fix**: Add the team ID and abbreviation to the mapping

### "Failed to load page" Warning
**Possible causes**:
1. Date format incorrect (should be MM/DD/YYYY)
2. Team abbreviation incorrect or changed
3. Game not in basketball-reference database (very rare games)
4. Network/rate limiting issues

**Debug steps**:
1. Check logs for constructed URL
2. Manually visit the URL to verify it exists
3. Check if team abbreviation matches basketball-reference's format
4. Verify date is correct

### Empty Validation Data
**Cause**: No home games found in scraped data
**Fix**: Ensure scraped data has MATCH UP column with "vs." designations

## Maintenance Checklist

- [ ] Update team mappings when franchises relocate
- [ ] Monitor basketball-reference.com for abbreviation changes
- [ ] Review failed scraping logs periodically
- [ ] Add new expansion teams to mappings
- [ ] Test with recent games after major NBA schedule changes

## Future Enhancements

### 1. Intelligent Retry for Doubleheaders
If first game ID fails, try with suffix (e.g., `200610311MIA.html`)

### 2. Date Extraction from Game ID
For special cases where GAME_DATE is missing, attempt to parse from NBA game ID structure

### 3. Caching
Cache successful conversions to avoid repeated scraping

### 4. Alternative Sources
If basketball-reference fails, fall back to other sources (ESPN, etc.)

## References

- [Basketball-Reference Box Scores](https://www.basketball-reference.com/leagues/NBA_2007_games.html)
- [NBA Stats API Documentation](https://github.com/swar/nba_api)
- Team ID reference: `/src/nba_app/webscraping/validation_scraper.py`

## Version History

- **v1.0** (2025-10-05): Initial metadata-based conversion implementation
- Replaces placeholder conversion logic from Phase 1
