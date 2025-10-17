#!/usr/bin/env python3
"""
Quick script to analyze calibration results from saved predictions.

Usage:
    python scripts/analyze_calibration.py <predictions_file.csv>

Example:
    python scripts/analyze_calibration.py data/predictions/xgboost_val_predictions.csv
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.calibration import calibration_curve


def calculate_ece(y_true, y_pred, n_bins=10):
    """Calculate Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (y_pred > bin_lower) & (y_pred <= bin_upper)

        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_accuracy = np.mean(y_true[in_bin])
            bin_confidence = np.mean(y_pred[in_bin])
            ece += (bin_size / len(y_true)) * np.abs(bin_accuracy - bin_confidence)

    return ece


def analyze_calibration(predictions_file: str):
    """Analyze calibration from predictions file."""

    # Load predictions
    print(f"\n{'='*60}")
    print(f"Calibration Analysis: {Path(predictions_file).name}")
    print(f"{'='*60}\n")

    df = pd.read_csv(predictions_file)

    # Detect prediction columns
    uncal_col = None
    cal_col = None

    for col in df.columns:
        if 'uncalibrated' in col.lower():
            uncal_col = col
        elif 'calibrated' in col.lower() and 'adjustment' not in col.lower():
            cal_col = col

    if uncal_col is None:
        print("❌ No uncalibrated predictions found in file")
        print(f"Available columns: {df.columns.tolist()}")
        return

    print(f"✓ Found predictions:")
    print(f"  - Uncalibrated: {uncal_col}")
    if cal_col:
        print(f"  - Calibrated: {cal_col}")
    else:
        print(f"  - Calibrated: Not found (calibration was not applied)")
        print("\nTo enable calibration, set enable_calibration: true in config")
        return

    # Get target column
    target_col = 'target'
    if target_col not in df.columns:
        print(f"❌ Target column '{target_col}' not found")
        return

    y_true = df[target_col].values
    y_uncal = df[uncal_col].values
    y_cal = df[cal_col].values

    # Calculate metrics
    print(f"\n{'='*60}")
    print("METRICS COMPARISON")
    print(f"{'='*60}\n")

    # Brier Score
    brier_uncal = brier_score_loss(y_true, y_uncal)
    brier_cal = brier_score_loss(y_true, y_cal)
    brier_improvement = brier_uncal - brier_cal

    print(f"Brier Score (lower is better):")
    print(f"  Uncalibrated: {brier_uncal:.6f}")
    print(f"  Calibrated:   {brier_cal:.6f}")
    print(f"  Improvement:  {brier_improvement:.6f} {'✓' if brier_improvement > 0 else '✗'}")

    # Log Loss
    logloss_uncal = log_loss(y_true, y_uncal)
    logloss_cal = log_loss(y_true, y_cal)
    logloss_improvement = logloss_uncal - logloss_cal

    print(f"\nLog Loss (lower is better):")
    print(f"  Uncalibrated: {logloss_uncal:.6f}")
    print(f"  Calibrated:   {logloss_cal:.6f}")
    print(f"  Improvement:  {logloss_improvement:.6f} {'✓' if logloss_improvement > 0 else '✗'}")

    # ECE
    ece_uncal = calculate_ece(y_true, y_uncal)
    ece_cal = calculate_ece(y_true, y_cal)
    ece_improvement = ece_uncal - ece_cal

    print(f"\nExpected Calibration Error (lower is better):")
    print(f"  Uncalibrated: {ece_uncal:.6f}")
    print(f"  Calibrated:   {ece_cal:.6f}")
    print(f"  Improvement:  {ece_improvement:.6f} {'✓' if ece_improvement > 0 else '✗'}")

    # AUC (should stay the same)
    auc_uncal = roc_auc_score(y_true, y_uncal)
    auc_cal = roc_auc_score(y_true, y_cal)
    auc_diff = abs(auc_uncal - auc_cal)

    print(f"\nAUC (should remain similar):")
    print(f"  Uncalibrated: {auc_uncal:.6f}")
    print(f"  Calibrated:   {auc_cal:.6f}")
    print(f"  Difference:   {auc_diff:.6f} {'✓' if auc_diff < 0.01 else '⚠️ Large change!'}")

    # Probability statistics
    print(f"\n{'='*60}")
    print("PROBABILITY STATISTICS")
    print(f"{'='*60}\n")

    print(f"Uncalibrated:")
    print(f"  Mean:   {y_uncal.mean():.4f}")
    print(f"  Std:    {y_uncal.std():.4f}")
    print(f"  Min:    {y_uncal.min():.4f}")
    print(f"  Max:    {y_uncal.max():.4f}")
    print(f"  Median: {np.median(y_uncal):.4f}")

    print(f"\nCalibrated:")
    print(f"  Mean:   {y_cal.mean():.4f}")
    print(f"  Std:    {y_cal.std():.4f}")
    print(f"  Min:    {y_cal.min():.4f}")
    print(f"  Max:    {y_cal.max():.4f}")
    print(f"  Median: {np.median(y_cal):.4f}")

    # Adjustment statistics
    if 'calibration_adjustment' in [c for c in df.columns if 'adjustment' in c.lower()]:
        adj_col = [c for c in df.columns if 'calibration_adjustment' in c.lower() and 'abs' not in c.lower()][0]
        adjustment = df[adj_col].values

        print(f"\nCalibration Adjustment:")
        print(f"  Mean:     {adjustment.mean():.4f}")
        print(f"  Std:      {adjustment.std():.4f}")
        print(f"  Mean Abs: {np.abs(adjustment).mean():.4f}")
        print(f"  Max Abs:  {np.abs(adjustment).max():.4f}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}\n")

    improvements = [brier_improvement > 0, logloss_improvement > 0, ece_improvement > 0]
    improvement_count = sum(improvements)

    if improvement_count == 3:
        print("✅ Calibration IMPROVED all metrics!")
        print("   → Probabilities are more reliable after calibration")
    elif improvement_count >= 2:
        print("✓ Calibration improved most metrics")
        print("  → Calibration is working")
    elif improvement_count == 1:
        print("⚠️ Calibration improved only 1 metric")
        print("  → Try isotonic method instead of sigmoid (or vice versa)")
    else:
        print("❌ Calibration did not improve metrics")
        print("  → Model may already be well-calibrated")
        print("  → Or try different calibration method")

    if auc_diff > 0.01:
        print("\n⚠️ Warning: AUC changed significantly!")
        print("  → This is unusual and may indicate an issue")

    # Create visualization
    print(f"\n{'='*60}")
    print("Creating visualization...")
    print(f"{'='*60}\n")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Calibration curves
    ax = axes[0, 0]
    prob_true_uncal, prob_pred_uncal = calibration_curve(y_true, y_uncal, n_bins=10)
    prob_true_cal, prob_pred_cal = calibration_curve(y_true, y_cal, n_bins=10)

    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax.plot(prob_pred_uncal, prob_true_uncal, 'ro-', label='Uncalibrated')
    ax.plot(prob_pred_cal, prob_true_cal, 'go-', label='Calibrated')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title('Calibration Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Probability distributions
    ax = axes[0, 1]
    ax.hist(y_uncal, bins=20, alpha=0.6, label='Uncalibrated', color='red')
    ax.hist(y_cal, bins=20, alpha=0.6, label='Calibrated', color='green')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Frequency')
    ax.set_title('Probability Distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Uncalibrated vs Calibrated scatter
    ax = axes[1, 0]
    ax.scatter(y_uncal, y_cal, alpha=0.3, s=10)
    ax.plot([0, 1], [0, 1], 'r--', label='No change')
    ax.set_xlabel('Uncalibrated Probability')
    ax.set_ylabel('Calibrated Probability')
    ax.set_title('Uncalibrated vs Calibrated')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Metrics comparison
    ax = axes[1, 1]
    metrics = ['Brier', 'Log Loss', 'ECE']
    uncal_values = [brier_uncal, logloss_uncal, ece_uncal]
    cal_values = [brier_cal, logloss_cal, ece_cal]

    x = np.arange(len(metrics))
    width = 0.35

    ax.bar(x - width/2, uncal_values, width, label='Uncalibrated', color='red', alpha=0.7)
    ax.bar(x + width/2, cal_values, width, label='Calibrated', color='green', alpha=0.7)
    ax.set_ylabel('Score (Lower is Better)')
    ax.set_title('Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = Path(predictions_file).parent / f"{Path(predictions_file).stem}_calibration_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to: {output_path}")

    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze_calibration.py <predictions_file.csv>")
        print("\nExample:")
        print("  python scripts/analyze_calibration.py data/predictions/xgboost_val_predictions.csv")
        sys.exit(1)

    predictions_file = sys.argv[1]

    if not Path(predictions_file).exists():
        print(f"❌ File not found: {predictions_file}")
        sys.exit(1)

    analyze_calibration(predictions_file)
