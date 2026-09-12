"""
test_data_loading.py — Unit tests for data loading and validation.
"""

import os
import sys
import pytest
import numpy as np
import tempfile
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loading import read_single_acc_file, load_bearing_data, compute_rul_labels, get_acc_files


class TestReadSingleAccFile:
    """Tests for reading individual accelerometer CSV files."""

    def _create_test_csv(self, tmpdir, filename, num_rows=2560):
        """Helper to create a test CSV file."""
        filepath = os.path.join(str(tmpdir), filename)
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            for i in range(num_rows):
                writer.writerow([9, 39, 39, 65664 + i * 39, 0.5 + 0.01 * i, -0.1 + 0.005 * i])
        return filepath

    def test_correct_row_count(self, tmp_path):
        """Test that a file with exactly 2560 rows is read correctly."""
        filepath = self._create_test_csv(tmp_path, "acc_00001.csv", 2560)
        df = read_single_acc_file(filepath, expected_rows=2560)
        assert df is not None
        assert len(df) == 2560
        assert list(df.columns) == ["hour", "minute", "second", "microsecond", "horiz_accel", "vert_accel"]

    def test_incorrect_row_count_flagged(self, tmp_path):
        """Test that a file with wrong row count is still read but flagged."""
        filepath = self._create_test_csv(tmp_path, "acc_00001.csv", 2000)
        df = read_single_acc_file(filepath, expected_rows=2560)
        assert df is not None
        assert len(df) == 2000  # Still returns data

    def test_column_values(self, tmp_path):
        """Test that column values are parsed correctly."""
        filepath = self._create_test_csv(tmp_path, "acc_00001.csv", 10)
        df = read_single_acc_file(filepath, expected_rows=10)
        assert df["hour"].iloc[0] == 9
        assert df["minute"].iloc[0] == 39
        assert abs(df["horiz_accel"].iloc[0] - 0.5) < 0.001


class TestComputeRulLabels:
    """Tests for RUL label computation."""

    def test_rul_decreasing(self):
        """RUL should be monotonically decreasing."""
        bearing_data = {"num_files": 100}
        rul = compute_rul_labels(bearing_data, recording_interval_s=10.0)
        assert len(rul) == 100
        assert rul[0] == 1000.0  # Total life = 100 * 10
        assert rul[-1] == 10.0   # Last file: 10s remaining
        assert np.all(np.diff(rul) < 0)  # Monotonically decreasing

    def test_rul_values(self):
        """Test specific RUL values."""
        bearing_data = {"num_files": 50}
        rul = compute_rul_labels(bearing_data, recording_interval_s=10.0)
        assert rul[0] == 500.0   # 50 * 10 - 0 * 10
        assert rul[25] == 250.0  # 50 * 10 - 25 * 10
        assert rul[49] == 10.0   # 50 * 10 - 49 * 10


class TestGetAccFiles:
    """Tests for file listing."""

    def test_sorted_order(self, tmp_path):
        """Files should be returned in numerical order."""
        for i in [5, 1, 10, 3, 2]:
            filepath = os.path.join(str(tmp_path), f"acc_{i:05d}.csv")
            with open(filepath, "w") as f:
                f.write("dummy")

        files = get_acc_files(str(tmp_path))
        assert len(files) == 5
        assert "acc_00001.csv" in files[0]
        assert "acc_00010.csv" in files[-1]

    def test_ignores_non_acc_files(self, tmp_path):
        """Non-acc files should be ignored."""
        for fname in ["acc_00001.csv", "temp_00001.csv", "readme.txt"]:
            with open(os.path.join(str(tmp_path), fname), "w") as f:
                f.write("dummy")

        files = get_acc_files(str(tmp_path))
        assert len(files) == 1


class TestRealDataValidation:
    """Tests against the actual dataset (run only if dataset is present)."""

    @pytest.fixture
    def learning_dir(self):
        path = os.path.join("Learning_set", "Bearing1_1")
        if not os.path.exists(path):
            pytest.skip("Learning set not found")
        return path

    def test_first_file_has_2560_rows(self, learning_dir):
        """First acc file should have 2560 rows."""
        filepath = os.path.join(learning_dir, "acc_00001.csv")
        if not os.path.exists(filepath):
            pytest.skip("File not found")
        df = read_single_acc_file(filepath)
        assert len(df) == 2560

    def test_bearing1_1_file_count(self, learning_dir):
        """Bearing1_1 should have 2803 files."""
        files = get_acc_files(learning_dir)
        assert len(files) == 2803


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
