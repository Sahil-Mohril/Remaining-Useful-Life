"""
test_scoring.py — Unit tests for the IEEE PHM 2012 scoring function.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluate import (
    phm_percent_error,
    phm_score_single,
    phm_challenge_score,
    rmse,
    mae,
    mape,
    r2_score,
)


class TestPhmPercentError:
    """Tests for %Er_i computation."""

    def test_perfect_prediction(self):
        """Er = 0 when prediction matches actual."""
        assert phm_percent_error(1000, 1000) == 0.0

    def test_early_prediction(self):
        """Positive Er when predicted < actual (predicted too early = safe)."""
        # Actual=1000, Predicted=800 → Er = 100*(1000-800)/1000 = 20%
        assert abs(phm_percent_error(1000, 800) - 20.0) < 0.01

    def test_late_prediction(self):
        """Negative Er when predicted > actual (predicted too late = dangerous)."""
        # Actual=1000, Predicted=1050 → Er = 100*(1000-1050)/1000 = -5%
        assert abs(phm_percent_error(1000, 1050) - (-5.0)) < 0.01

    def test_zero_actual(self):
        """Handle zero actual RUL gracefully."""
        assert phm_percent_error(0, 100) == 0.0


class TestPhmScoreSingle:
    """Tests for A_i scoring function."""

    def test_perfect_prediction(self):
        """Er=0 should give A=1.0."""
        assert abs(phm_score_single(0) - 1.0) < 1e-6

    def test_late_5_percent(self):
        """Er=-5 should give A=0.5 (definition of the exponential)."""
        assert abs(phm_score_single(-5) - 0.5) < 1e-6

    def test_early_20_percent(self):
        """Er=+20 should give A=0.5."""
        assert abs(phm_score_single(20) - 0.5) < 1e-6

    def test_late_10_percent(self):
        """Er=-10 should give A=0.25."""
        assert abs(phm_score_single(-10) - 0.25) < 1e-6

    def test_early_40_percent(self):
        """Er=+40 should give A=0.25."""
        assert abs(phm_score_single(40) - 0.25) < 1e-6

    def test_asymmetry(self):
        """Late predictions should be penalized more than early ones."""
        a_late = phm_score_single(-10)    # 10% late
        a_early = phm_score_single(10)    # 10% early
        assert a_late < a_early  # Late is worse

    def test_score_in_range(self):
        """Score should always be positive."""
        for er in np.linspace(-100, 200, 100):
            assert phm_score_single(er) > 0

    def test_score_monotonic_for_positive(self):
        """For Er > 0 (early), score should decrease as Er increases."""
        scores = [phm_score_single(er) for er in range(0, 100)]
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

    def test_score_monotonic_for_negative(self):
        """For Er < 0 (late), score should decrease as Er becomes more negative."""
        scores = [phm_score_single(er) for er in range(0, -50, -1)]
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


class TestPhmChallengeScore:
    """Tests for the overall challenge scoring."""

    def test_perfect_predictions(self):
        """Perfect predictions should give Score=1.0."""
        actuals = {"B1": 1000, "B2": 2000, "B3": 500}
        predicted = {"B1": 1000, "B2": 2000, "B3": 500}
        results = phm_challenge_score(actuals, predicted)
        assert abs(results["overall_score"] - 1.0) < 1e-6

    def test_score_with_known_values(self):
        """Test with specific known values."""
        actuals = {"B1": 1000}
        # Predicted 800 → Er = 20% → A = exp(ln(0.5) * 20/20) = 0.5
        predicted = {"B1": 800}
        results = phm_challenge_score(actuals, predicted)
        assert abs(results["per_bearing"][0]["A_i"] - 0.5) < 1e-4

    def test_regression_metrics_included(self):
        """Results should include regression metrics."""
        actuals = {"B1": 1000, "B2": 2000}
        predicted = {"B1": 900, "B2": 1800}
        results = phm_challenge_score(actuals, predicted)
        assert "regression_metrics" in results
        assert "rmse" in results["regression_metrics"]
        assert "mae" in results["regression_metrics"]


class TestRegressionMetrics:
    """Tests for standard regression metrics."""

    def test_rmse_perfect(self):
        actual = np.array([1.0, 2.0, 3.0])
        predicted = np.array([1.0, 2.0, 3.0])
        assert rmse(actual, predicted) == 0.0

    def test_rmse_known(self):
        actual = np.array([1.0, 2.0, 3.0])
        predicted = np.array([2.0, 3.0, 4.0])
        assert abs(rmse(actual, predicted) - 1.0) < 1e-6

    def test_mae_perfect(self):
        actual = np.array([1.0, 2.0, 3.0])
        predicted = np.array([1.0, 2.0, 3.0])
        assert mae(actual, predicted) == 0.0

    def test_r2_perfect(self):
        actual = np.array([1.0, 2.0, 3.0, 4.0])
        predicted = np.array([1.0, 2.0, 3.0, 4.0])
        assert abs(r2_score(actual, predicted) - 1.0) < 1e-6

    def test_mape_known(self):
        actual = np.array([100.0, 200.0])
        predicted = np.array([110.0, 180.0])
        # MAPE = mean(|10/100|, |20/200|) * 100 = mean(10, 10) = 10
        assert abs(mape(actual, predicted) - 10.0) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
