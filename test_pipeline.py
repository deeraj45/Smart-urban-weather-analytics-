import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_generator import generate_raw_dataset
from src.data_pipeline import clip_out_of_range, interpolate_gaps, fuse_sources, engineer_features
from src.anomaly_detection import detect_anomalies


def _sample_raw():
    return generate_raw_dataset(n_days=5, seed=1)


def test_generate_raw_dataset_shape():
    df = _sample_raw()
    assert len(df) > 0
    assert {"zone", "source", "temperature_c"}.issubset(df.columns)


def test_pipeline_removes_nans():
    df = _sample_raw()
    df = clip_out_of_range(df)
    df = interpolate_gaps(df)
    assert df["temperature_c"].isna().sum() == 0


def test_fuse_sources_reduces_to_one_row_per_zone_timestamp():
    df = _sample_raw()
    df = clip_out_of_range(df)
    df = interpolate_gaps(df)
    fused = fuse_sources(df)
    dupes = fused.duplicated(subset=["zone", "timestamp"]).sum()
    assert dupes == 0


def test_engineer_features_adds_lag_columns():
    df = _sample_raw()
    df = clip_out_of_range(df)
    df = interpolate_gaps(df)
    fused = fuse_sources(df)
    featured = engineer_features(fused)
    assert "temp_lag_1h" in featured.columns
    assert featured["temp_lag_1h"].isna().sum() == 0


def test_anomaly_detection_flags_reasonable_fraction():
    df = _sample_raw()
    df = clip_out_of_range(df)
    df = interpolate_gaps(df)
    fused = fuse_sources(df)
    featured = engineer_features(fused)
    flagged = detect_anomalies(featured, contamination=0.02)
    assert "is_anomaly" in flagged.columns
    fraction = flagged["is_anomaly"].mean()
    assert 0 < fraction < 0.1
