import os
import sys
import numpy as np
import pandas as pd
import joblib
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

np.random.seed(42)
os.makedirs("plots", exist_ok=True)

# Dataset Loader
def load_cmapss_data(fd, base_folder="CMaps"):
    fd = int(fd)

    train_file = f"train_FD00{fd}.txt"
    test_file  = f"test_FD00{fd}.txt"
    rul_file   = f"RUL_FD00{fd}.txt"

    search_paths = [
        os.path.join(base_folder, train_file),
        f"/root/.cache/kagglehub/datasets/behrad3d/nasa-cmaps/versions/1/CMaps/{train_file}",
        f"/content/{train_file}",
        f"/mnt/data/{train_file}",
        f"/kaggle/input/nasa-cmaps/CMaps/{train_file}"
    ]

    found_folder = None
    for p in search_paths:
        if os.path.exists(p):
            found_folder = os.path.dirname(p)
            break

    if not found_folder:
        raise FileNotFoundError(f"Could not find dataset files for FD00{fd}.")

    print(f"Using dataset FD00{fd} from: {found_folder}")

    column_names = (
        ['unit_number', 'time_cycles'] +
        [f'op_setting_{i}' for i in range(1,4)] +
        [f'sensor_{i}' for i in range(1,22)]
    )

    train_df = pd.read_csv(os.path.join(found_folder, train_file), sep=r"\s+", header=None, names=column_names)
    test_df  = pd.read_csv(os.path.join(found_folder, test_file),  sep=r"\s+", header=None, names=column_names)
    rul_df   = pd.read_csv(os.path.join(found_folder, rul_file),   sep=r"\s+", header=None)

    return train_df, test_df, rul_df

# Compute RUL
def compute_rul(df):
    max_c = df.groupby("unit_number")["time_cycles"].max().reset_index()
    max_c.columns = ["unit_number", "max_cycle"]
    df = df.merge(max_c, on="unit_number", how="left")
    df["RUL"] = df["max_cycle"] - df["time_cycles"]
    df = df.drop(columns=["max_cycle"])
    return df

# FD-Specific Preprocessing

def preprocess_for_fd(fd, df):
    drop_cols = ["sensor_1","sensor_5","sensor_10","sensor_16","sensor_19","op_setting_3"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    df = compute_rul(df)

    return df


# Rolling Features (optimized)

def engineer_features(df, win=5):
    df = df.sort_values(["unit_number","time_cycles"])

    sensors = [c for c in df.columns if c.startswith("sensor_")]
    for s in tqdm(sensors, desc="Rolling Features"):
        df[f"{s}_rm{win}"] = df.groupby("unit_number")[s].transform(
            lambda x: x.shift(1).rolling(win, min_periods=1).mean()
        )
    
    df["cycle_norm"] = df.groupby("unit_number")["time_cycles"].transform(lambda x: x / x.max())
    return df


# Windowing     

def create_windows(df, window=30):
    feature_cols = [
        c for c in df.columns
        if c.startswith("sensor_") or c.startswith("op_setting_") or c == "cycle_norm"
    ]

    X, y, units = [], [], []

    for eng in tqdm(df.unit_number.unique(), desc="Windowing Engines"):
        tmp = df[df.unit_number == eng].sort_values("time_cycles")
        if len(tmp) <= window:
            continue

        vals = tmp[feature_cols].values
        targets = tmp["RUL"].values

        for i in range(window, len(tmp)):
            # Only take LAST row of window → reduces data 30x
            X.append(vals[i])
            y.append(targets[i])
            units.append(eng)

    return np.array(X), np.array(y), np.array(units), feature_cols

# Split by Engine

def split_by_engine(X, y, engines):
    unique_eng = np.unique(engines)
    train_eng, test_eng = train_test_split(unique_eng, test_size=0.2, random_state=42)

    train_mask = np.isin(engines, train_eng)
    test_mask  = np.isin(engines, test_eng)

    return X[train_mask], X[test_mask], y[train_mask], y[test_mask]

# GPU XGBoost

def get_regressor():
    try:
        import xgboost as xgb
        print("Using GPU XGBoost")
        return xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method="gpu_hist",
            predictor="gpu_predictor"
        )
    except:
        print("XGBoost unavailable → Using RandomForest")
        return RandomForestRegressor(n_estimators=200, random_state=42)

# MAIN TRAINING

def train_on_fd(fd):
    print("\n===============================================")
    print(f"Training FD00{fd}")
    print("===============================================\n")

    train_df, test_df, rul_df = load_cmapss_data(fd)

    # Compute NASA test-set RUL
    max_cycles = test_df.groupby("unit_number")["time_cycles"].max().reset_index()
    max_cycles.columns = ["unit_number", "max_cycle"]
    test_df = test_df.merge(max_cycles, on="unit_number")

    test_df["final_rul"] = test_df["unit_number"].apply(lambda u: rul_df.iloc[int(u)-1, 0])
    test_df["RUL"] = (test_df["max_cycle"] - test_df["time_cycles"]) + test_df["final_rul"]
    test_df = test_df.drop(columns=["max_cycle","final_rul"])

    # Combine
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    full_df = preprocess_for_fd(fd, full_df)
    full_df = engineer_features(full_df, win=5)

    # Windowing
    X, y, engines, feature_cols = create_windows(full_df, window=30)

    # Scale
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = split_by_engine(X, y, engines)

    # Train model
    model = get_regressor()
    print("\nTraining model...")
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)

    print(f"\nFD00{fd} Results → RMSE={rmse:.3f}, MAE={mae:.3f}, R2={r2:.3f}")

    def plot_actual_vs_predicted(y_true, y_pred, fd_tag):
        r2 = r2_score(y_true, y_pred)

        plt.figure(figsize=(8, 6))

        plt.scatter(y_true, y_pred, alpha=0.5, label="Predicted Points")

        max_val = max(max(y_true), max(y_pred))
        plt.plot([0, max_val], [0, max_val], linestyle='--', color='red', label="Ideal Line")

        plt.title(f"{fd_tag} Actual vs Predicted RUL (R² = {r2:.3f})")
        plt.xlabel("Actual RUL")
        plt.ylabel("Predicted RUL")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()

        filepath = f"plots/{fd_tag}_actual_vs_pred.png"
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved: {filepath}")

    def plot_error_distribution(y_true, y_pred, fd_tag):
        errors = y_pred - y_true
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)

        plt.figure(figsize=(8, 6))
        plt.hist(errors, bins=40, alpha=0.7)

        plt.title(f"{fd_tag} Prediction Error Distribution")
        plt.xlabel("Prediction Error (Predicted - Actual)")
        plt.ylabel("Frequency")

        plt.axvline(0, color='black', linewidth=2, label='Zero Error')
        plt.axvline(errors.mean(), color='red', linestyle='--', label='Mean Error')

        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        
        plt.text(
            0.95, 0.95,
            f"Mean Error: {errors.mean():.2f}\nSD ~ RMSE: {rmse:.2f}",
            transform=plt.gca().transAxes,
            va='top', ha='right',
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
        )

        filepath = f"plots/{fd_tag}_error_distribution.png"
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved: {filepath}")

    # Save Models
    joblib.dump(model,  f"model_FD00{fd}.joblib")
    joblib.dump(scaler, f"scaler_FD00{fd}.joblib")
    joblib.dump(feature_cols, f"features_FD00{fd}.joblib")
    
    print(f"Saved models for FD00{fd}\n")
    
    fd_tag = f"FD00{fd}" if isinstance(fd, int) else fd

    plot_actual_vs_predicted(y_test, y_pred, fd_tag)
    plot_error_distribution(y_test, y_pred, fd_tag)

if __name__ == "__main__":

    if len(sys.argv) == 1:
        print("Usage: python main.py FD001 or python main.py all")
        sys.exit()

    arg = sys.argv[1]

    if arg.lower() == "all":
        for fd in [1,2,3,4]:
            train_on_fd(fd)
    else:
        fd = int(arg.replace("FD00","").replace("FD0","").replace("FD",""))
        train_on_fd(fd)
