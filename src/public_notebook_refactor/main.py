import os
import sys
import glob
import subprocess
import warnings
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor, log_evaluation, early_stopping
from catboost import CatBoostRegressor
from sklearn.metrics import root_mean_squared_error

# Environment Setup
os.environ["ROGII_GOLD_PROFILE"] = "conservative"
warnings.filterwarnings("ignore")

kb_dir = '/kaggle/input/koolbox-offline'
if not os.path.isdir(kb_dir):
    cand = glob.glob('/kaggle/input/**/koolbox*', recursive=True)
    if cand: kb_dir = cand[0]
if os.path.isdir(kb_dir):
    whls = glob.glob(f'{kb_dir}/**/*.whl', recursive=True)
    if whls:
        for w in whls:
            subprocess.run(['pip', 'install', '--no-deps', w], check=False)
    else:
        sys.path.insert(0, kb_dir)
        for sub in os.listdir(kb_dir):
            sub_path = os.path.join(kb_dir, sub)
            if os.path.isdir(sub_path): sys.path.insert(0, sub_path)

import koolbox
from koolbox import Trainer

# Import refactored local architectural pieces
from config import *
from processing_helpers import (
    load_well_v2, tvt_from_contacts, run_pf_lik_ensemble_scales, 
    run_beam_ensemble, apply_pp
)
from dataset_builder import build_dataset
from post_processing import sg_smooth, apply_projection_post_processing
from visualization import plot_performance_metrics

def main():
    # 1. Dataset Acquisition Phase
    if (CFG.artifacts_path / "data" / "train.csv").exists():
        train_df = pd.read_csv(CFG.artifacts_path / "data" / "train.csv", low_memory=False)
    else:
        train_paths = sorted((CFG.dataset_path / "train").glob('*__horizontal_well.csv'))
        train_df = build_dataset(train_paths, is_train=True, label="train")    

    test_paths = sorted((CFG.dataset_path / "test").glob('*__horizontal_well.csv'))
    test_df = build_dataset(test_paths, is_train=False, label="test")

    features = [c for c in train_df.columns if c not in {'well','id','target'}]
    X, y, g = train_df[features], train_df['target'], train_df['well']
    X_test = test_df[features]

    # 2. Model Optimization Loop
    oof_preds, test_preds, overall_scores, fold_scores = {}, {}, {}, {}

    for i, params in enumerate(lgb_params):   
        save_path = f"models/lightgbm-{i+1}"
        if (CFG.artifacts_path / save_path).exists():
            trainer = joblib.load(list((CFG.artifacts_path / save_path).glob('*.pkl'))[0])
        else:
            trainer = Trainer(estimator=LGBMRegressor(**params), task="regression", metric=CFG.metric, cv=CFG.cv, cv_args={"groups": g}, use_early_stopping=True, verbose=True, save=True, save_path=save_path)
            trainer.fit(X, y, fit_args={"eval_metric": "rmse", "callbacks": [log_evaluation(period=250), early_stopping(stopping_rounds=250)]})
        
        oof_preds[f"lightgbm-{i+1}"] = trainer.oof_preds
        test_preds[f"lightgbm-{i+1}"] = trainer.predict(X_test)
        overall_scores[f"lightgbm-{i+1}"] = trainer.overall_score
        fold_scores[f"lightgbm-{i+1}"] = trainer.fold_scores

    for i, params in enumerate(cb_params):    
        save_path = f"models/catboost-{i+1}"
        if (CFG.artifacts_path / save_path).exists():
            trainer = joblib.load(list((CFG.artifacts_path / save_path).glob('*.pkl'))[0])
        else:
            trainer = Trainer(estimator=CatBoostRegressor(**params), task="regression", metric=CFG.metric, cv=CFG.cv, cv_args={"groups": g}, use_early_stopping=True, verbose=True, save=True, save_path=save_path)
            trainer.fit(X, y, fit_args={"verbose": 250, "early_stopping_rounds": 250, "use_best_model": True})

        oof_preds[f"catboost-{i+1}"] = trainer.oof_preds
        test_preds[f"catboost-{i+1}"] = trainer.predict(X_test)
        overall_scores[f"catboost-{i+1}"] = trainer.overall_score
        fold_scores[f"catboost-{i+1}"] = trainer.fold_scores

    # Meta-Modeling Layer
    oof_preds_df = pd.DataFrame(oof_preds)
    test_preds_df = pd.DataFrame(test_preds)

    ridge_trainer = Trainer(Ridge(**ridge_params), task="regression", metric=CFG.metric, cv=CFG.cv, cv_args={"groups": g}, verbose=True)
    ridge_trainer.fit(oof_preds_df, y)

    ridge_oof_preds = ridge_trainer.oof_preds
    ridge_test_preds = ridge_trainer.predict(test_preds_df)

    overall_scores["ridge"] = ridge_trainer.overall_score
    fold_scores["ridge"] = ridge_trainer.fold_scores

    # Post Processing Sub-ensemble pipeline
    base = train_df['last_known_tvt'].values
    ytrue = y.values + base
    pf_oof = (train_df['pf_ancc'].values - base)

    d = apply_pp(df=train_df, md=ridge_oof_preds, pd_=pf_oof, **pp_params)
    ridge_score = root_mean_squared_error(ytrue, base + d)
    overall_scores["ridge (pp)"] = ridge_score
    fold_scores["ridge (pp)"] = [ridge_score] * CFG.n_splits

    test_df2 = test_df.copy()
    pf_test = test_df2['pf_ancc'].values - test_df2['last_known_tvt'].values
    test_df2['pred'] = test_df2['last_known_tvt'].values + apply_pp(df=test_df2, md=ridge_test_preds, pd_=pf_test, **pp_params)
    test_df2 = sg_smooth(test_df2, 'pred')

    sample_sub = pd.read_csv(CFG.dataset_path / "sample_submission.csv")
    sub_1 = sample_sub[['id']].merge(test_df2[['id', 'pred']].rename(columns={'pred':'tvt'}), on='id', how='left')
    sub_1['tvt'] = sub_1['tvt'].fillna(float(train_df['last_known_tvt'].mean() + train_df['target'].mean()))

    # Direct Physical & Geometric Simulation Fallback Loop
    sample = pd.read_csv(CFG.dataset_path / 'sample_submission.csv')
    sample['well'] = sample['id'].str[:8]
    sample['row_idx'] = sample['id'].str[9:].astype(int)

    train_hw_files = sorted(glob.glob(str(CFG.dataset_path / 'train' / '*__horizontal_well.csv')))
    train_wells = [os.path.basename(f).split('__')[0] for f in train_hw_files]
    test_hw_files = sorted(glob.glob(str(CFG.dataset_path / 'test' / '*__horizontal_well.csv')))
    test_wells = [os.path.basename(f).split('__')[0] for f in test_hw_files]

    rows = []
    for wid in test_wells:
        hw_te, tw_te = load_well_v2(wid=wid, split='test')
        tvt_phys, hw_tr, tw_tr = None, None, None

        if wid in train_wells:
            try:
                hw_tr, tw_tr = load_well_v2(wid=wid, split='train')
                hw_te['TVT_input'] = hw_tr['TVT_input'].values
                tvt_phys = tvt_from_contacts(hw_tr, tw_tr)
            except: pass

        try:
            tw_ref = tw_tr if tw_tr is not None else tw_te
            pf_by_scale = run_pf_lik_ensemble_scales(hw_te, tw_ref, n_particles=500, n_seeds=128)
            tvt_pf = pf_by_scale['pf_scale_8']
        except:
            last_known = hw_te['TVT_input'].dropna()
            last_val = float(last_known.iloc[-1]) if len(last_known) > 0 else 0.0
            tvt_pf = hw_te['TVT_input'].fillna(last_val).values.astype(float)
            pf_by_scale = {f'pf_scale_{scale:g}': tvt_pf.copy() for scale in SELECTOR_SCALES}

        tvt_selector = (pf_by_scale['pf_scale_3'] + pf_by_scale['pf_scale_5'] + pf_by_scale['pf_scale_8']) / 3.0
        ws = sample[sample['well'] == wid]
        for _, row in ws.iterrows():
            ridx = int(row['row_idx'])
            tvt_val = float(tvt_phys.iloc[ridx]) if tvt_phys is not None else float(tvt_selector[ridx])
            rows.append({'id': row['id'], 'tvt': tvt_val})

    sub_2 = pd.DataFrame(rows)
    sub = sub_1.merge(sub_2, on='id', suffixes=('_1', '_2')).assign(tvt=lambda x: 0.3 * x['tvt_1'] + 0.7 * x['tvt_2'])[['id', 'tvt']]
    sub.to_csv("submission.csv", index=False)

    # 4. Final Post Processing Projection Overwrite
    apply_projection_post_processing(submission_path="submission.csv")

    # Save backup track array
    _sp45_df = pd.read_csv('submission.csv')
    _sp45_df.to_csv('sp45_projection_submission.csv', index=False)

    # Performance Visual Diagnostics
    plot_performance_metrics(fold_scores=fold_scores, overall_scores=overall_scores)

if __name__ == '__main__':
    main()