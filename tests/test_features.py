import pandas as pd
import statistics
import pytest

from anomaly_detection.features import build_features, save_feature_outputs
from anomaly_detection.generator import generate_sensor_data


def test_build_features_returns_expected_outputs():
    raw_df = generate_sensor_data(
        num_machines=2,
        num_timesteps=80,
        fixed_seed=295,
    )

    features_df, retention_df = build_features(raw_df)

    assert not features_df.empty
    assert not retention_df.empty


def test_build_features_has_required_original_columns():
    raw_df = generate_sensor_data(
        num_machines=2,
        num_timesteps=80,
        fixed_seed=295,
    )

    features_df, retention_df = build_features(raw_df)

    required_columns = {
        "step",
        "machine_id",
        "any_anomaly",
        "target_sensor",
        "anomaly_type",
        "temperature",
        "pressure",
        "vibration",
        "flow_rate",
        "voltage",
        "current",
    }

    assert required_columns.issubset(set(features_df.columns))


def test_build_features_has_engineered_columns():
    raw_df = generate_sensor_data(
        num_machines=2,
        num_timesteps=80,
        fixed_seed=295,
    )

    features_df, retention_df = build_features(raw_df)

    expected_engineered_columns = {
        "temperature_delta",
        "temperature_abs_delta",
        "temperature_roll_mean",
        "temperature_roll_std",
        "temperature_zscore",
        "temperature_5_step_diff",
        "temperature_10_step_diff",
        "temperature_slope_10",
        "temperature_lag_5_autocorr",
        "temperature_center_balance_10",
    }

    assert expected_engineered_columns.issubset(set(features_df.columns))


def test_build_features_drops_initial_rolling_rows():
    raw_df = generate_sensor_data(
        num_machines=2,
        num_timesteps=80,
        fixed_seed=295,
    )

    features_df, retention_df = build_features(raw_df)

    assert len(features_df) < len(raw_df)


def test_build_features_retention_report_has_expected_columns():
    raw_df = generate_sensor_data(
        num_machines=2,
        num_timesteps=80,
        fixed_seed=295,
    )

    features_df, retention_df = build_features(raw_df)

    expected_columns = {
        "anomaly_type",
        "target_sensor",
        "before_drop",
        "after_drop",
        "rows_dropped",
        "pct_kept",
    }

    assert expected_columns.issubset(set(retention_df.columns))


def test_build_features_is_reproducible_for_same_input():
    raw_df = generate_sensor_data(
        num_machines=2,
        num_timesteps=80,
        fixed_seed=295,
    )

    features_df_1, retention_df_1 = build_features(raw_df)
    features_df_2, retention_df_2 = build_features(raw_df)

    pd.testing.assert_frame_equal(features_df_1, features_df_2)
    pd.testing.assert_frame_equal(retention_df_1, retention_df_2)


def test_save_feature_outputs_writes_files(tmp_path):
    raw_df = generate_sensor_data(
        num_machines=2,
        num_timesteps=80,
        fixed_seed=295,
    )

    features_df, retention_df = build_features(raw_df)

    features_output_path = tmp_path / "test_sensor_data_features.csv"
    retention_output_path = tmp_path / "test_feature_row_retention.csv"

    save_feature_outputs(
        features_df=features_df,
        retention_df=retention_df,
        features_output_path=features_output_path,
        retention_output_path=retention_output_path,
    )

    assert features_output_path.exists()
    assert retention_output_path.exists()

    saved_features_df = pd.read_csv(features_output_path)
    saved_retention_df = pd.read_csv(retention_output_path)

    assert len(saved_features_df) == len(features_df)
    assert len(saved_retention_df) == len(retention_df)
    

def test_save_features_creates_missing_path(tmp_path):
    features_output_path = tmp_path / "features" / "run_1" / "features.csv"
    retention_output_path = tmp_path / "retention" / "run_1" / "retention.csv"
    
    features_df = pd.DataFrame({"temperature": [1, 2, 3, 4]})
    retention_df = pd.DataFrame({"rows_retained": [5, 6, 7, 8]})
    
    assert not features_output_path.parent.exists()
    assert not retention_output_path.parent.exists()
    
    save_feature_outputs(
        features_df=features_df,
        retention_df=retention_df,
        features_output_path=features_output_path,
        retention_output_path=retention_output_path,
    )
    
    assert features_output_path.exists()
    assert retention_output_path.exists()

    saved_features_df = pd.read_csv(features_output_path)
    saved_retention_df = pd.read_csv(retention_output_path)

    pd.testing.assert_frame_equal(features_df, saved_features_df)
    pd.testing.assert_frame_equal(retention_df, saved_retention_df)
    
    
def test_build_features_handles_shuffled_inputs():
    raw_df = generate_sensor_data(
        num_machines=2,
        num_timesteps=80,
        fixed_seed=295,
    )
    
    shuffled_df = raw_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    raw_features, _ = build_features(raw_df)
    shuffled_features, _ = build_features(shuffled_df)
    
    pd.testing.assert_frame_equal(
        raw_features.sort_values(by=["machine_id", "step"]).reset_index(drop=True),
        shuffled_features.sort_values(by=["machine_id", "step"]).reset_index(drop=True),
    )
    

def test_build_features_calculations_are_correct():
    pattern = [10, 15, 25, 20, 10, 15, 25, 20, 10, 30]

    test_data = pd.DataFrame({
        "machine_id": [1] * 80,
        "step": range(1, 81),
        "temperature": pattern * 8,
        "pressure": [50] * 80,
        "vibration": [2] * 80,
        "flow_rate": [1.0] * 80,
        "voltage": [120] * 80,
        "current": [10] * 80,
        "anomaly_type": ["none"] * 80,
        "target_sensor": ["none"] * 80,
    })
    
    features, _ = build_features(test_data)
    
    row = features.loc[features["step"] == 80]
    assert len(row) == 1
    row = row.iloc[0]
    
    assert row["temperature_delta"] == pytest.approx(20)
    assert row["temperature_roll_mean"] == pytest.approx(18)
    assert row["temperature_roll_std"] == pytest.approx(7.14920353)
    assert row["temperature_zscore"] == pytest .approx(1.67850865)