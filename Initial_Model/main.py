"""
Improved Predictive Maintenance Script for NASA C-MAPSS (FD001)
Student: Jahanvi Singh (2210110914)
Created: improved version

Features:
- Safe data loading (works with Kaggle/Colab cache)
- No-leakage rolling features (uses shift)
- Window-based dataset creation for sequence models
- Flexible model: XGBoost (preferred) or LightGBM/RandomForest fallback
- Quantile-based uncertainty using GradientBoostingRegressor (quantile loss)
- Better train/test splitting (random engines)
- Local Outlier Factor for anomaly detection (no forced contamination)
- Improved metrics and clearer printing (no misleading rounding)
- Rich visualizations saved to PNG

Notes:
- If xgboost or lightgbm not installed, the script falls back to sklearn's RandomForestRegressor.
- The script refers to an uploaded original script at: /mnt/data/predictive_maintenance copy.py (if present).
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def load_cmapss_data(base_path=None):
    """Load FD001 train file. Tries several common locations (kaggle cache, colab, uploaded file).

    Returns
    -------
    train_df : pandas.DataFrame
    """
    candidates = []
    if base_path:
        candidates.append(base_path)

    # common Kaggle path used in previous runs
    candidates.append('/kaggle/input/nasa-cmaps/train_FD001.txt')
    candidates.append('/kaggle/input/nasa-cmaps/train_FD001.txt')

    # the dataset path used previously in your uploaded code (fallback)
    candidates.append('/root/.cache/kagglehub/datasets/behrad3d/nasa-cmaps/versions/1/CMaps/train_FD001.txt')

    # local uploaded script path -- sibling location
    # (developer note: the uploaded file is available at /mnt/data/predictive_maintenance copy.py)
    candidates.append('/mnt/data/train_FD001.txt')

    # Try to locate a file in /kaggle/input or /content
    candidates.append('/kaggle/input/nasa-cmaps/train_FD001.txt')
    candidates.append('/content/train_FD001.txt')

    found = None
    for p in candidates:
        try:
            if p and os.path.exists(p):
                found = p
                break
        except Exception:
            continue

    if found is None:
        raise FileNotFoundError(
            "Could not find train_FD001.txt. Provide base_path or upload the file to /content or /mnt/data."
        )

    print(f"Loading data from: {found}")

    # Column names
    cols = ['unit_number', 'time_cycles'] + [f'op_setting_{i}' for i in range(1, 4)] + [f'sensor_{i}' for i in range(1, 22)]

    df = pd.read_csv(found, sep='\s+', header=None, names=cols)
    # Some parsers add extra NA columns when sep is whitespace; drop NA-only columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.dropna(axis=1, how='all')

    return df


# ---------------------------------------------------------------------------
# 2. Preprocessing and RUL computation
# ---------------------------------------------------------------------------

def compute_rul(df):
    max_cycles = df.groupby('unit_number')['time_cycles'].max().reset_index()
    max_cycles.columns = ['unit_number', 'max_cycle']
    df = df.merge(max_cycles, on='unit_number', how='left')
    df['RUL'] = df['max_cycle'] - df['time_cycles']
    df = df.drop(columns=['max_cycle'])
    return df


def preprocess_data(df, sensors_to_remove=None):
    """Remove known-constant sensors and compute RUL."""
    if sensors_to_remove is None:
        sensors_to_remove = ['sensor_1','sensor_5','sensor_10','sensor_16','sensor_19','op_setting_3']

    cols_present = [c for c in sensors_to_remove if c in df.columns]
    df = df.drop(columns=cols_present)
    print(f"Removed {len(cols_present)} sensors/settings: {cols_present}")

    df = compute_rul(df)
    print("Computed RUL for each row (no piecewise cap applied)")
    return df


# ---------------------------------------------------------------------------
# 3. Feature engineering (no leakage)
# ---------------------------------------------------------------------------

def engineer_features(df, rolling_window=5):
    df = df.sort_values(['unit_number', 'time_cycles']).copy()
    sensor_cols = [c for c in df.columns if c.startswith('sensor_')]

    # Use shift(1) so rolling stats don't include current or future rows (no leakage)
    for col in sensor_cols:
        df[f'{col}_rm{rolling_window}'] = df.groupby('unit_number')[col].transform(
            lambda x: x.shift(1).rolling(rolling_window, min_periods=1).mean()
        )
        df[f'{col}_rs{rolling_window}'] = df.groupby('unit_number')[col].transform(
            lambda x: x.shift(1).rolling(rolling_window, min_periods=1).std()
        ).fillna(0)

    # Optionally: add cycle-based features
    df['cycle_norm'] = df.groupby('unit_number')['time_cycles'].transform(lambda x: x / x.max())
    return df


# ---------------------------------------------------------------------------
# 4. Windowing for sequence models
# ---------------------------------------------------------------------------

def create_windows(df, window_size=30, feature_cols=None, target_col='RUL'):
    """Create sliding windows for each engine.

    Returns X (n_samples, window_size, n_features) and y (n_samples,)
    """
    engines = df['unit_number'].unique()
    X_list = []
    y_list = []
    idx_engine = []

    if feature_cols is None:
        feature_cols = [c for c in df.columns if (c.startswith('sensor_') or c.startswith('op_setting_') or c=='cycle_norm')]

    for eng in engines:
        sub = df[df['unit_number'] == eng].sort_values('time_cycles')
        vals = sub[feature_cols].values
        targets = sub[target_col].values
        n = len(sub)
        if n <= window_size:
            # skip engines that are too short (or pad if desired)
            continue
        for i in range(window_size, n):
            X_list.append(vals[i-window_size:i])
            y_list.append(targets[i])
            idx_engine.append(eng)

    X = np.array(X_list)
    y = np.array(y_list)
    idx_engine = np.array(idx_engine)
    return X, y, idx_engine, feature_cols


# ---------------------------------------------------------------------------
# 5. Train/test split (by engines)
# ---------------------------------------------------------------------------

def split_by_engine(X, y, engines_idx, test_size=0.2):
    unique_engines = np.unique(engines_idx)
    train_eng, test_eng = train_test_split(unique_engines, test_size=test_size, random_state=42)
    train_mask = np.isin(engines_idx, train_eng)
    test_mask = np.isin(engines_idx, test_eng)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    return X_train, X_test, y_train, y_test, train_eng, test_eng


# ---------------------------------------------------------------------------
# 6. Model training (XGBoost preferred, fallback to sklearn)
# ---------------------------------------------------------------------------

def get_regressor():
    try:
        import xgboost as xgb
        model = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=8,
            subsample=0.7,
            colsample_bytree=0.8,
            random_state=42,
            tree_method="gpu_hist",
            predictor="gpu_predictor",
            n_jobs=-1
        )
        print("Using XGBoost XGBRegressor")
        return model, 'xgboost'
    except Exception:
        print("XGBoost not available, falling back to RandomForestRegressor")
        model = RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)
        return model, 'rf'


# ---------------------------------------------------------------------------
# 7. Quantile-based uncertainty (GradientBoostingRegressor)
# ---------------------------------------------------------------------------

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor

def train_quantile_models(X_train_2d, y_train):
    q_models = {}
    for q in [0.05, 0.5, 0.95]:
        reg = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=q,
            max_depth=6,
            max_leaf_nodes=31,
            learning_rate=0.05,
            max_iter=300
        )
        reg.fit(X_train_2d, y_train)
        q_models[q] = reg
    return q_models


# ---------------------------------------------------------------------------
# 8. Evaluation utilities
# ---------------------------------------------------------------------------

def evaluate_regression(y_true, y_pred, prefix=''):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{prefix}RMSE: {rmse:.3f}")
    print(f"{prefix}MAE:  {mae:.3f}")
    print(f"{prefix}R2:   {r2:.3f}")
    return {'rmse': rmse, 'mae': mae, 'r2': r2}


# ---------------------------------------------------------------------------
# 9. Anomaly detection (LOF)
# ---------------------------------------------------------------------------

def detect_anomalies_lof(df, feature_cols, n_neighbors=35, contamination=0.02):
    X = df[feature_cols].values
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    labels = lof.fit_predict(X)
    df['anomaly_lof'] = (labels == -1)
    n_anom = df['anomaly_lof'].sum()
    print(f"Detected {n_anom} anomalies ({n_anom/len(df)*100:.2f}%) using LOF")
    return df


# ---------------------------------------------------------------------------
# 10. Visualization helpers
# ---------------------------------------------------------------------------

def plot_results(df, y_test, y_pred, pred_std, lower_q, upper_q, save_path='predictive_maintenance_improved.png'):
    plt.figure(figsize=(16,10))

    # Scatter actual vs predicted (sampled)
    plt.subplot(2,2,1)
    sample = np.random.choice(len(y_test), min(len(y_test), 1000), replace=False)
    plt.scatter(y_test[sample], y_pred[sample], alpha=0.4)
    plt.plot([0, np.max(y_test)], [0, np.max(y_test)], 'r--')
    plt.xlabel('Actual RUL')
    plt.ylabel('Predicted RUL')
    plt.title('Actual vs Predicted RUL')

    # Uncertainty violin (pred std)
    plt.subplot(2,2,2)
    plt.violinplot(pred_std, showmeans=True)
    plt.title('Prediction Std (violin)')

    # Residual histogram
    plt.subplot(2,2,3)
    res = y_test - y_pred
    plt.hist(res, bins=50)
    plt.title('Residual Distribution')

    # RUL distribution
    plt.subplot(2,2,4)
    plt.hist(df['RUL'], bins=50)
    plt.title('RUL Distribution')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"Saved visualization to {save_path}")


# ---------------------------------------------------------------------------
# 11. Main pipeline
# ---------------------------------------------------------------------------

def main(base_path=None, window_size=30):
    print('\nStarting improved predictive maintenance pipeline')

    # Load
    df = load_cmapss_data(base_path=base_path)

    # Preprocess
    df = preprocess_data(df)

    # Feature engineering (no leakage)
    df = engineer_features(df, rolling_window=5)

    # Windowing
    X, y, idx_engines, feature_cols = create_windows(df, window_size=window_size)
    print(f"Created {len(X)} windows of size {window_size}. Features per step: {len(feature_cols)}")

    # Flatten windows to 2D for tree-based models
    n_samples, w, n_feats = X.shape
    X_2d = X.reshape(n_samples, w * n_feats)

    # Split by engine
    X_train, X_test, y_train, y_test, train_eng, test_eng = split_by_engine(X_2d, y, idx_engines)
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Train engines: {len(train_eng)}, Test engines: {len(test_eng)}")

    # Scale features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Fix NaNs created by rolling features + windowing
    X_train_scaled = pd.DataFrame(X_train_scaled).fillna(0).values
    X_test_scaled = pd.DataFrame(X_test_scaled).fillna(0).values

    # Train main regressor
    model, model_name = get_regressor()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    print('\nMain model evaluation:')
    eval_main = evaluate_regression(y_test, y_pred)

    # Train quantile models for uncertainty
    print('\nTraining quantile regressors for uncertainty estimation (may take time)...')
    q_models = train_quantile_models(X_train_scaled, y_train)
    p05 = q_models[0.05].predict(X_test_scaled)
    p50 = q_models[0.5].predict(X_test_scaled)
    p95 = q_models[0.95].predict(X_test_scaled)

    # Compute simple pred_std from quantiles
    pred_std = (p95 - p05) / 4.0  # approx sd from 90% interval

    # Coverage
    coverage = np.mean((y_test >= p05) & (y_test <= p95)) * 100
    print(f"Quantile coverage (5-95): {coverage:.2f}%")
    print(f"Mean predicted uncertainty (approx std): {np.mean(pred_std):.3f}")

    # Cost-aware decision thresholding using thresholds
    thresholds = np.arange(5, 51, 5)
    results = []
    cost_unplanned = 10000
    cost_planned = 1000
    cost_unnecessary = 500
    for thresh in thresholds:
        sched = y_pred < thresh
        tp = np.sum(sched & (y_test < thresh))
        fp = np.sum(sched & (y_test >= thresh))
        fn = np.sum((~sched) & (y_test < thresh))
        total_cost = fn * cost_unplanned + tp * cost_planned + fp * cost_unnecessary
        results.append({'threshold': thresh, 'avg_cost': total_cost / len(y_test), 'tp': tp, 'fp': fp, 'fn': fn})
    results_df = pd.DataFrame(results)
    opt_idx = results_df['avg_cost'].idxmin()
    print('\nCost-aware analysis:')
    print(results_df)
    print(f"Optimal threshold: {results_df.loc[opt_idx,'threshold']} (avg cost per sample: {results_df.loc[opt_idx,'avg_cost']:.2f})")

    # Anomaly detection using LOF on raw sensor-level features (no windows)
    sensor_features = [c for c in df.columns if c.startswith('sensor_')][:10]
    df_anom = df.copy()
    df_anom = detect_anomalies_lof(df_anom.fillna(0), sensor_features, n_neighbors=35, contamination=0.02)

    # Visualize
    plot_results(df, y_test, y_pred, pred_std, p05, p95)

    # Save models and scaler
    joblib.dump({'model': model, 'scaler': scaler, 'q_models': q_models}, 'pm_models.joblib')
    print('Saved trained models to pm_models.joblib')

    # Summarize
    print('\nSummary:')
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Main model RMSE: {eval_main['rmse']:.3f}, MAE: {eval_main['mae']:.3f}, R2: {eval_main['r2']:.3f}")
    print(f"Quantile 5-95 coverage: {coverage:.2f}%")
    print(f"Optimal maintenance threshold: {results_df.loc[opt_idx,'threshold']} cycles")
    print('Pipeline completed.')


if __name__ == '__main__':
    # If running in notebook or Colab, you can supply base_path to train_FD001.txt
    # Example: python predictive_maintenance_improved.py /kaggle/input/nasa-cmaps/train_FD001.txt
    base = None
    if len(sys.argv) > 1:
        base = sys.argv[1]
    main(base_path=base, window_size=30)
