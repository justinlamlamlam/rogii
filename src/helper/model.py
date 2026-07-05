# model.py

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from lightgbm import LGBMRegressor, log_evaluation, early_stopping
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold

# ==============================================================================
# CONFIGURATION AND HYPERPARAMETERS
# ==============================================================================

class ModelConfig:
    seed = 42
    n_splits = 5
    cv = GroupKFold(n_splits=n_splits)
    metric = root_mean_squared_error

def get_lgb_configs(device: str = "gpu") -> list:
    """
    Returns pre-tuned LightGBM hyperparameter configurations for model exploration.

    Args:
        device (str): System target accelerator layout ('gpu' or 'cpu'). Defaults to 'gpu'.

    Returns:
        list: Configuration dictionaries for model instantiation.
    """
    return [
        dict(
            n_estimators=10000,
            max_depth=9,
            num_leaves=63,
            subsample=0.75,
            random_state=42,
            importance_type="gain",
            boosting_type="gbdt",
            learning_rate=0.00934485794382918,
            colsample_bytree=0.39283351290380497,
            min_child_weight=0.24081152127177283,
            min_child_samples=40,
            device=device,
        ),
    ]

def get_cb_configs(device: str = "GPU") -> list:
    """
    Returns pre-tuned CatBoost hyperparameter configurations for cross-validation stacking.

    Args:
        device (str): System target accelerator layout ('GPU' or 'CPU'). Defaults to 'GPU'.

    Returns:
        list: Configuration dictionaries for model instantiation.
    """
    return [
        dict(
            iterations=8000,
            depth=7,
            l2_leaf_reg=2.0,
            min_data_in_leaf=15,
            border_count=254,
            loss_function="RMSE",
            task_type=device,
            devices="0",
            od_type="Iter",
            od_wait=300,
            verbose=0,
            learning_rate=0.020,
            random_seed=7,
        ),
        dict(
            iterations=8000,
            depth=7,
            l2_leaf_reg=2.0,
            min_data_in_leaf=15,
            border_count=254,
            loss_function="RMSE",
            task_type=device,
            devices="0",
            od_type="Iter",
            od_wait=300,
            verbose=0,
            learning_rate=0.030,
            random_seed=123,
        ),
    ]

# ==============================================================================
# MODEL STACKING TRAINING & INFERENCE PIPELINE
# ==============================================================================

def train_stack(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    features: list, 
    target_col: str = "TVT", 
    device_mode: str = "gpu"
) -> tuple:
    """
    Runs a cross-validated out-of-fold GBDT stacking structure blended via L2 Ridge regression.

    Args:
        train_df (pd.DataFrame): Training set dataframe containing learning features.
        test_df (pd.DataFrame): Inference set dataframe containing matching target structural columns.
        features (list): Strict collection list of string identifier feature names.
        target_col (str): Prediction label column identifier string. Defaults to 'TVT'.
        device_mode (str): Execution platform selector ('gpu' or 'cpu'). Defaults to 'gpu'.

    Returns:
        tuple: Contains final out-of-fold predictions, blended test matrix results, and localized evaluation logs.
    """
    X = train_df[features].values.astype(dtype=np.float32)
    y = train_df[target_col].values.astype(dtype=np.float32)
    Xt = test_df[features].values.astype(dtype=np.float32)
    g = train_df["well_id"].values
    
    oof_cols = {}
    test_cols = {}
    
    # Process LightGBM Ensemble Members
    lgb_dev = "gpu" if device_mode == "gpu" else "cpu"
    for i, params in enumerate(iterable=get_lgb_configs(device=lgb_dev)):
        name = f"lgb_{i}"
        oof = np.zeros(shape=len(train_df))
        tp = np.zeros(shape=len(test_df))
        
        for tr, va in ModelConfig.cv.split(X=X, y=y, groups=g):
            model = LGBMRegressor(**params)
            model.fit(
                X[tr], y[tr],
                eval_set=[(X[va], y[va])],
                callbacks=[early_stopping(stopping_rounds=250, verbose=False), log_evaluation(period=0)]
            )
            it = model.best_iteration_
            oof[va] = model.predict(X[va], num_iteration=it)
            tp += model.predict(Xt, num_iteration=it) / ModelConfig.n_splits
            
        oof_cols[name] = oof
        test_cols[name] = tp
        print(f"[{name}] OOF RMSE: {ModelConfig.metric(y_true=y, y_pred=oof):.4f}", flush=True)

    # Process CatBoost Ensemble Members
    cb_dev = "GPU" if device_mode == "gpu" else "CPU"
    for i, params in enumerate(iterable=get_cb_configs(device=cb_dev)):
        name = f"cb_{i}"
        oof = np.zeros(shape=len(train_df))
        tp = np.zeros(shape=len(test_df))
        
        for tr, va in ModelConfig.cv.split(X=X, y=y, groups=g):
            model = CatBoostRegressor(**params)
            model.fit(
                X[tr], y[tr],
                eval_set=[(X[va], y[va])],
                early_stopping_rounds=250
            )
            oof[va] = model.predict(X[va])
            tp += model.predict(Xt) / ModelConfig.n_splits
            
        oof_cols[name] = oof
        test_cols[name] = tp
        print(f"[{name}] OOF RMSE: {ModelConfig.metric(y_true=y, y_pred=oof):.4f}", flush=True)

    # Execute Meta-Learner Consensus Blending via Ridge Regression
    OOF = pd.DataFrame(data=oof_cols)
    TEST = pd.DataFrame(data=test_cols)
    
    ridge_meta = Ridge(alpha=1.6602834637650032, positive=True, fit_intercept=True, random_state=42)
    meta_oof = np.zeros(shape=len(train_df))
    meta_test = np.zeros(shape=len(test_df))
    
    for tr, va in ModelConfig.cv.split(X=OOF.values, y=y, groups=g):
        ridge_meta.fit(X=OOF.values[tr], y=y[tr])
        meta_oof[va] = ridge_meta.predict(X=OOF.values[va])
        meta_test += ridge_meta.predict(X=TEST.values) / ModelConfig.n_splits
        
    final_score = ModelConfig.metric(y_true=y, y_pred=meta_oof)
    print(f"--> [Meta-Ridge Stacking] Final Consolidated OOF RMSE: {final_score:.4f}", flush=True)
    
    return meta_oof, meta_test, final_score

# ==============================================================================
# INTEGRITY AUDIT AND FILE SUBMISSION EXPORT
# ==============================================================================

def gold_validate_and_write(
    test_df: pd.DataFrame, 
    predictions: np.ndarray, 
    output_path: Path = Path("submission.csv")
) -> None:
    """
    Performs dimensional constraints checks and maps evaluated outputs to disk.

    Args:
        test_df (pd.DataFrame): Validation index dataset containing original row records.
        predictions (np.ndarray): Final evaluation layout predictions array.
        output_path (Path): File destination address for storage configuration. Defaults to Path('submission.csv').

    Returns:
        None
    """
    submission = pd.DataFrame()
    submission["id"] = test_df["id"]
    submission["TVT"] = predictions
    
    # Structural Fallback Handling for Boundary Execution Errors
    if submission["TVT"].isna().sum() > 0:
        print("[Warning] Null values encountered during parsing. Executing fallback imputation...", flush=True)
        submission["TVT"] = submission["TVT"].fillna(value=test_df["last_known_tvt"])
        
    submission.to_csv(output_path, index=False)
    print(f"Successfully generated audited verification file: {output_path} (Rows: {len(submission)})", flush=True)