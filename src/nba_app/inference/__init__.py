"""NBA Inference Module

Daily prediction pipeline for upcoming NBA games.

This module orchestrates:
1. Loading scheduled games for tomorrow
2. Feature engineering using historical data
3. Model inference with preprocessing
4. Postprocessing (calibration + conformal prediction)
5. Saving predictions for dashboard consumption
"""
