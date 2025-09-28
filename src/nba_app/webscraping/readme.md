<style>
  body {
    font-family: Arial, sans-serif;
    line-height: 1.6;
    color: #333;
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
    background-color: #f4f4f4;
  }
  h1 {
    color: #2c3e50;
    border-bottom: 2px solid #3498db;
    padding-bottom: 10px;
  }
  h2 {
    color: #2980b9;
    margin-top: 30px;
  }
  .step {
    background-color: #fff;
    border-left: 4px solid #3498db;
    margin-bottom: 20px;
    padding: 15px;
    border-radius: 0 5px 5px 0;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
  }
  .loop {
    margin-left: 20px;
    border-left: 2px dashed #95a5a6;
    padding-left: 20px;
  }
  .nested-loop {
    margin-left: 20px;
    border-left: 2px dotted #bdc3c7;
    padding-left: 20px;
  }
  .sub-step {
    margin: 10px 0;
  }
  .highlight {
    background-color: #e74c3c;
    color: white;
    padding: 2px 5px;
    border-radius: 3px;
  }
</style>

<h1>Data Scraping and Processing Workflow</h1>

<div class="step">
  <h2>1. Initialization</h2>
  <p>Determine which date to start scraping from</p>
</div>

<div class="step">
  <h2>2. Scrape Boxscores</h2>
  <div class="loop">
    <p><strong>For each</strong> stat_type in stat_types:</p>
    <div class="nested-loop">
      <p><strong>For each</strong> season in seasons:</p>
      <div class="nested-loop">
        <p><strong>For each</strong> sub_season in sub_seasons:</p>
        <div class="sub-step">
          <ul>
            <li>Construct URL with dates, stat_type, sub_season</li>
            <li>Go to URL</li>
            <li>Scrape data table</li>
            <li>Convert to dataframe</li>
            <li>Concat dataframe with previous sub_seasons dataframes</li>
          </ul>
        </div>
      </div>
      <p>Concat dataframe with previous seasons dataframes</p>
    </div>
    <p>Save dataframe to CSV</p>
  </div>
</div>

<div class="step">
  <h2>3. Scrape Schedule</h2>
  <ul>
    <li>Go to URL</li>
    <li>Find games for today</li>
    <li>Scrape game IDs and matchup team IDs</li>
    <li>Save game IDs to CSV</li>
    <li>Save matchups to CSV</li>
  </ul>
</div>

<div class="step">
  <h2>4. Data Validation and Integration</h2>
  <ul>
    <li>Validate the newly scraped data for consistency</li>
    <li>Concat the newly scraped data with previously saved cumulative data and save to CSVs</li>
    <li>Validate the new cumulative data for consistency</li>
  </ul>
</div>

<div class="step">
  <h2>5. <span class="highlight">Process Complete</span></h2>
</div>