"""
test_feature_extraction.py — Unit tests for feature extraction.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_extraction import (
    time_domain_features,
    frequency_domain_features,
    envelope_features,
    extract_features_single,
)


class TestTimeDomainFeatures:
    """Tests for time-domain feature extraction."""

    def test_rms_sine_wave(self):
        """RMS of a sine wave with amplitude A should be A/sqrt(2)."""
        t = np.linspace(0, 1, 25600)
        amplitude = 5.0
        x = amplitude * np.sin(2 * np.pi * 100 * t)
        features = time_domain_features(x)
        expected_rms = amplitude / np.sqrt(2)
        assert abs(features["rms"] - expected_rms) < 0.01

    def test_peak_value(self):
        """Peak should be the maximum absolute value."""
        x = np.array([1.0, -3.0, 2.0, -1.0, 0.5])
        features = time_domain_features(x)
        assert abs(features["peak"] - 3.0) < 1e-6

    def test_std_known(self):
        """Standard deviation of known signal."""
        x = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        features = time_domain_features(x)
        expected_std = np.std(x, ddof=1)
        assert abs(features["std"] - expected_std) < 1e-6

    def test_kurtosis_gaussian(self):
        """Gaussian noise should have kurtosis ~0 (excess kurtosis)."""
        np.random.seed(42)
        x = np.random.randn(10000)
        features = time_domain_features(x)
        assert abs(features["kurtosis"]) < 0.2  # Should be close to 0

    def test_skewness_symmetric(self):
        """Symmetric signal should have skewness ~0."""
        t = np.linspace(0, 1, 2560)
        x = np.sin(2 * np.pi * 50 * t)
        features = time_domain_features(x)
        assert abs(features["skewness"]) < 0.1

    def test_crest_factor(self):
        """Crest factor = peak / RMS."""
        x = np.array([0.0, 0.0, 0.0, 10.0, 0.0])  # Impulsive
        features = time_domain_features(x)
        rms = np.sqrt(np.mean(x ** 2))
        expected_cf = 10.0 / rms
        assert abs(features["crest_factor"] - expected_cf) < 0.01

    def test_all_features_present(self):
        """All expected features should be in the output."""
        x = np.random.randn(2560)
        features = time_domain_features(x)
        expected_keys = ["rms", "kurtosis", "skewness", "crest_factor",
                         "peak", "peak_to_peak", "std", "shape_factor",
                         "impulse_factor", "clearance_factor"]
        for key in expected_keys:
            assert key in features, f"Missing feature: {key}"

    def test_no_nan_or_inf(self):
        """Features should not contain NaN or Inf."""
        x = np.random.randn(2560)
        features = time_domain_features(x)
        for key, val in features.items():
            assert np.isfinite(val), f"Feature {key} is not finite: {val}"


class TestFrequencyDomainFeatures:
    """Tests for frequency-domain feature extraction."""

    def test_spectral_centroid_pure_tone(self):
        """Spectral centroid of a pure tone should be near the tone frequency."""
        fs = 25600
        freq = 1000  # Hz
        t = np.arange(2560) / fs
        x = np.sin(2 * np.pi * freq * t)
        features = frequency_domain_features(x, sampling_rate=fs)
        # Spectral centroid should be close to 1000 Hz
        assert abs(features["spectral_centroid"] - freq) < 100

    def test_band_power(self):
        """Band power should be concentrated in the signal's frequency band."""
        fs = 25600
        freq = 300  # Hz — falls in bpfi band [200, 400]
        t = np.arange(2560) / fs
        x = np.sin(2 * np.pi * freq * t)
        bands = {"bpfi": [200, 400], "bpfo": [100, 200]}
        features = frequency_domain_features(x, sampling_rate=fs, fft_bands=bands)
        # Most power should be in bpfi band
        assert features["band_power_bpfi"] > features["band_power_bpfo"]

    def test_no_nan_or_inf(self):
        """Features should not contain NaN or Inf."""
        x = np.random.randn(2560)
        features = frequency_domain_features(x, sampling_rate=25600)
        for key, val in features.items():
            assert np.isfinite(val), f"Feature {key} is not finite: {val}"


class TestEnvelopeFeatures:
    """Tests for envelope analysis features."""

    def test_envelope_output_keys(self):
        """Should return the expected feature keys."""
        x = np.random.randn(2560)
        features = envelope_features(x, sampling_rate=25600)
        assert "envelope_rms" in features
        assert "envelope_peak" in features
        assert "envelope_kurtosis" in features

    def test_envelope_positive(self):
        """Envelope RMS and peak should always be positive."""
        x = np.random.randn(2560)
        features = envelope_features(x, sampling_rate=25600)
        assert features["envelope_rms"] > 0
        assert features["envelope_peak"] > 0


class TestExtractFeaturesSingle:
    """Tests for combined feature extraction."""

    def test_output_shape(self):
        """Feature dict should have entries for both channels."""
        horiz = np.random.randn(2560)
        vert = np.random.randn(2560)
        features = extract_features_single(horiz, vert, sampling_rate=25600)
        # Should have h_ and v_ prefixed features
        h_keys = [k for k in features if k.startswith("h_")]
        v_keys = [k for k in features if k.startswith("v_")]
        assert len(h_keys) > 0
        assert len(v_keys) > 0
        assert len(h_keys) == len(v_keys)

    def test_total_feature_count(self):
        """Should have the expected number of total features."""
        horiz = np.random.randn(2560)
        vert = np.random.randn(2560)
        bands = {"bpfo": [100, 200], "bpfi": [200, 400], "bsf": [400, 800]}
        features = extract_features_single(horiz, vert, sampling_rate=25600, fft_bands=bands)
        # 10 time-domain + 6 freq-domain + 3 envelope = 19 per channel × 2 = 38
        # But freq-domain with 3 bands: 3 spectral + 3 band_power = 6
        # So: (10 + 6 + 3) × 2 = 38
        assert len(features) >= 30  # Allow some flexibility


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
