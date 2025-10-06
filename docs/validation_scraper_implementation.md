# Validation Scraper Implementation - Phase 1

## Overview
This document describes the implementation of the ValidationScraper module to address home/visitor team mislabeling issues from NBA.com by scraping validation data from basketball-reference.com.

## Architecture

### Modular Design
The implementation maintains strict separation of concerns:
- **Webscraping Module**: Handles all external data collection (NBA.com + basketball-reference.com)
- **Data Processing Module**: Receives clean data, detects issues, applies corrections
- **Future: Data Validation Module**: Will orchestrate validation and correction strategies

### Components Implemented

#### 1. ValidationScraper ([validation_scraper.py](../src/nba_app/webscraping/validation_scraper.py))
**Purpose**: Scrape minimal validation dataset from basketball-reference.com

**Key Features**:
- Lightweight scraping (only essential fields: game_id, home_team, visitor_team, scores)
- Team abbreviation to NBA ID mapping
- Graceful error handling (doesn't fail pipeline if some games fail)
- Integrated with existing PageScraper infrastructure

**Methods**:
- `scrape_validation_data_for_games(game_ids)`: Scrapes validation data for list of games
- `scrape_and_save_validation_data(game_ids)`: Scrapes and saves to CSV
- `_scrape_single_game_validation(game_id)`: Scrapes one game
- `_extract_visitor_info()`: Extracts visitor team and score
- `_extract_home_info()`: Extracts home team and score

**Output Schema**:
```python
{
    'GAME_ID': str,           # NBA game ID
    'HOME_TEAM_ID': str,      # Home team NBA ID
    'VISITOR_TEAM_ID': str,   # Visitor team NBA ID
    'HOME_SCORE': str,        # Final score
    'VISITOR_SCORE': str,     # Final score
    'SOURCE': str,            # Always 'basketball-reference'
    'SCRAPED_AT': str         # ISO timestamp
}
```

#### 2. NbaScraper Updates ([nba_scraper.py](../src/nba_app/webscraping/nba_scraper.py))
**Changes**:
- Added optional `validation_scraper` parameter to `__init__`
- New method: `scrape_and_save_validation_data(game_ids)`
- Validation scraper is optional (won't break if not provided)

#### 3. DI Container Updates ([di_container.py](../src/nba_app/webscraping/di_container.py))
**Changes**:
- Added `validation_scraper` provider
- Wired `validation_scraper` into `nba_scraper`

#### 4. Main Pipeline Updates ([main.py](../src/nba_app/webscraping/main.py))
**Changes**:
- New function: `scrape_validation_data()` - extracts game IDs and triggers validation scraping
- Conditionally scrapes validation data after boxscore/matchup scraping
- Graceful degradation: pipeline continues even if validation scraping fails

#### 5. Configuration Updates ([webscraping_config.yaml](../configs/nba/webscraping_config.yaml))
**New Settings**:
```yaml
enable_validation_scraping: True  # Toggle validation data collection
validation_data_file: validation_data.csv  # Output file name
```

## Usage

### Automatic (Recommended)
Validation scraping runs automatically during the normal webscraping pipeline:

```bash
cd src/nba_app/webscraping
python main.py
```

The pipeline will:
1. Scrape boxscores from NBA.com
2. Scrape matchups from NBA.com
3. **Scrape validation data from basketball-reference.com** (if enabled)
4. Validate and concatenate data

### Manual Testing
Test the ValidationScraper independently:

```bash
cd src/nba_app/webscraping
python test_validation_scraper.py
```

### Disable Validation Scraping
Set in config:
```yaml
enable_validation_scraping: False
```

## Data Flow

```
┌─────────────────────────────────────────────────────────┐
│ WEBSCRAPING PHASE                                       │
├─────────────────────────────────────────────────────────┤
│ 1. Scrape NBA.com → games_*.csv                        │
│ 2. Scrape basketball-reference → validation_data.csv   │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ DATA PROCESSING PHASE (Future Enhancement)              │
├─────────────────────────────────────────────────────────┤
│ 3. Compare MATCHUP column vs validation data           │
│ 4. Detect mismatches (home/visitor designation)        │
│ 5. Apply corrections from validation data              │
│ 6. Log corrections for audit trail                     │
└─────────────────────────────────────────────────────────┘
```

## Next Steps (Phase 2)

### Create Data Validation Module
```
src/nba_app/data_validation/
├── __init__.py
├── issue_detector.py          # Compare primary vs validation data
├── resolution_strategies.py   # Strategy pattern for fixes
└── llm_resolver.py           # LLM-based resolution (Phase 3)
```

### Implementation Tasks
1. **Issue Detection**:
   - Load validation_data.csv
   - Compare with MATCHUP column in games_*.csv
   - Detect mismatches: both teams marked as home/visitor
   - Generate issue report

2. **Correction Strategy**:
   - Simple fix: Use validation data to correct MATCHUP
   - Update home_team column in data_processing
   - Log all corrections

3. **Integration**:
   - Add validation step in data_processing/main.py
   - Run before feature engineering
   - Output: corrected data + correction log

## Known Limitations

### Basketball-Reference Game ID Conversion
Current implementation has a placeholder for NBA game ID → basketball-reference game ID conversion.

**NBA Format**: `SSYYMGGGGG`
- SS = season type (00=preseason, 01=regular, 02=regular, 03=playoffs, 04=all-star)
- YY = season year (24 = 2024-25)
- M = month indicator
- GGGGG = sequential game number

**Basketball-Reference Format**: `YYYYMMDD0HHH`
- YYYYMMDD = game date
- 0 = separator
- HHH = home team abbreviation

**Solution Needed**:
- Option 1: Maintain mapping table (game_id → date + home_team)
- Option 2: Parse game dates from NBA.com data
- Option 3: Use basketball-reference's search API

### Team Abbreviation Mapping
Current mapping includes modern and historical teams. May need updates for:
- Future team relocations
- Name changes
- Expansion teams

### Scraping Reliability
Basketball-reference.com structure may change. Monitor for:
- CSS selector changes
- Page layout modifications
- Rate limiting

## Testing

### Unit Tests (Recommended to Add)
```python
test_validation_scraper.py:
- test_team_abbrev_to_id_mapping()
- test_scrape_single_game_validation()
- test_extract_visitor_info()
- test_extract_home_info()
- test_graceful_failure_handling()
```

### Integration Tests
```python
test_webscraping_pipeline.py:
- test_full_pipeline_with_validation()
- test_pipeline_continues_if_validation_fails()
- test_validation_data_saved_correctly()
```

## Configuration Reference

### Required Config Attributes
```python
config.enable_validation_scraping: bool
config.validation_data_file: str
config.game_id_column: str  # e.g., 'GAME_ID'
```

### Optional Optimizations
```yaml
# Future config options
validation_scraping_mode: 'all' | 'sample' | 'flagged'
validation_batch_size: 50  # Scrape in batches
validation_retry_failed: True  # Retry failed games
```

## Performance Considerations

### Current Approach
- Scrapes validation data for ALL newly scraped games
- Sequential scraping (one game at a time)
- No caching

### Future Optimizations
1. **Conditional Scraping**: Only scrape validation for flagged games
2. **Batch Processing**: Parallel requests with rate limiting
3. **Caching**: Store validation data, only scrape new games
4. **Smart Sampling**: Validate sample of games, flag errors, then validate all

## Error Handling

### Graceful Degradation
- If validation scraping fails, pipeline continues
- Errors logged but don't stop data processing
- Allows pipeline to run unattended

### Error Scenarios Handled
1. Basketball-reference page not found → Skip game, log warning
2. Unknown team abbreviation → Skip game, log warning
3. Page structure changed → Skip game, log error
4. Network timeout → Skip game, retry if configured

## Maintenance

### Regular Checks
- [ ] Verify basketball-reference.com CSS selectors still work
- [ ] Update team abbreviation mapping for new teams
- [ ] Monitor validation scraping success rate
- [ ] Review correction logs for patterns

### Logging
All actions logged with structured logging:
```python
app_logger.structured_log(logging.INFO, "Message",
    game_id=game_id,
    operation="scrape_validation",
    success=True
)
```

Check logs:
- `logs/webscraping.log` - Main pipeline
- `logs/test_validation_scraper.log` - Test runs

## Contributors
- Initial implementation: Phase 1 - Validation Dataset
- Date: 2025-10-05

## License
Same as parent project
