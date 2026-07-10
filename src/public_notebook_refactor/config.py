import multiprocessing
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import root_mean_squared_error

class CFG:
    dataset_path = Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction")
    artifacts_path = Path("/kaggle/input/datasets/ravaghi/wellbore-geology-prediction-artifacts")
    DATA = dataset_path # Merged fallback path from the end of code
    OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
    
    seed = 42
    n_splits = 5
    cv = GroupKFold(n_splits=n_splits)
    metric = root_mean_squared_error
    n_jobs = min(8, multiprocessing.cpu_count())
    
    # Likelihood particle filter parameters
    PF_SEEDS = 128
    PF_PARTICLES = 500
    PF_SCALES = (3., 5., 8., 12.)
    FAST = False
    N_TRAIN_WELLS = 0
    USE_GPU = "auto"
    SHOW_FIGS = True

# Signal thresholds
SELECTOR_N_EVAL_THRESHOLD = 4840.0
SELECTOR_Z_SPAN_THRESHOLDS = (136.73000000000016, 185.5133333333342)

SELECTOR_BIN_VARIANTS = {
    0: 'pf_scale_5_hold_0.2',
    1: 'pf_scale_3_hold_0.15',
    2: 'pf_scale_12_beam_0.2_hold_0.15',
    3: 'pf_scale_5_hold_0.15',
    4: 'pf_scale_5_beam_0.05_hold_0.05',
    5: 'pf_scale_12_beam_0.2_hold_0.05',
}

SELECTOR_GLOBAL_VARIANT = 'pf_scale_8_hold_0.2'
SELECTOR_SCALES = (3.0, 5.0, 8.0, 12.0)

FORMATION_COLS = ['ANCC', 'ASTNU', 'ASTNL', 'EGFDU', 'EGFDL', 'BUDA']
FORMATIONS = FORMATION_COLS

BEAM_CONFIGS = [
    (10, 20.0, 144.0, 2), (10,  8.0,  64.0, 2), ( 8, 35.0, 220.0, 1),
    (10, 14.0,  90.0, 5), (20,  4.0,  36.0, 3), (12, 12.0, 100.0, 3),
    (15, 25.0, 180.0, 2), (20, 30.0, 200.0, 2), (15, 10.0,  80.0, 4),
    (25,  6.0,  50.0, 3), (10, 40.0, 300.0, 1), (12, 18.0, 120.0, 5),
    (30,  8.0,  70.0, 2), (10, 50.0, 400.0, 0),
]

BEAMS = [
    (10, 20.0, 144.0, 2, "cons"), (10, 8.0, 64.0, 2, "loose"),
    (8, 35.0, 220.0, 1, "vcons"), (10, 14.0, 90.0, 5, "sm5"),
    (20, 4.0, 36.0, 3, "vloose"), (12, 12.0, 100.0, 3, "mid"),
    (15, 25.0, 180.0, 2, "stiff"),
]

PLANE_K = 10; DENSE_SPW = 60; DENSE_K = 20
PF_N = 600; ANCC_N = 600
PF_MOM = 0.993; PF_VN = 0.005; PF_PN = 0.01
PF_GR_SIG_MIN = 10.; PF_GR_SIG_MAX = 60.; PF_GR_SIG_DEF = 30.
PF_INIT_V_STD = 0.02; PF_INIT_SPR = 0.5; PF_RESAMP = 0.5
PF_ROUGH_P = 0.2; PF_ROUGH_V = 0.003; PF_GR_WIN = 5; PF_GR_WT = 0.3
ANCC_ALPHA = 0.998; ANCC_RN = 0.002; ANCC_PN = 0.005
ANCC_IR = 0.01; ANCC_IS = 0.3; ANCC_RP = 0.1; ANCC_RR = 0.001

ANCH_OFFS = [ -80, -40, -20, -10, -5, 0, 5, 10, 20, 40, 80 ]
BEAM_OFFS = [ -40, -20, -10, -5, -3, 0, 3, 5, 10, 20, 40 ]
SC_OFFS   = [ -30, -15, -8, -4, -2, 0, 2, 4, 8, 15, 30 ]
PF_OFFS   = [ -30, -15, -8, -4, -2, 0, 2, 4, 8, 15, 30 ]

# Modeling Parameters
lgb_params = [
    dict(boosting_type="gbdt", num_leaves=255, min_child_samples=15, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=3.0, reg_alpha=0.05, objective="regression", verbose=-1, n_jobs=-1, device_type="gpu", gpu_use_dp=False, max_bin=255, learning_rate=0.030, n_estimators=5000, seed=123),
    dict(n_jobs=-1, verbose=-1, reg_alpha=10.788188919840913, subsample=0.47437582748953966, num_leaves=64, reg_lambda=95.75401894533888, n_estimators=10000, random_state=0, boosting_type='gbdt', learning_rate=0.00934485794382918, colsample_bytree=0.39283351290380497, min_child_weight=0.24081152127177283, min_child_samples=40, device='gpu'),
    dict(n_jobs=-1, verbose=-1, reg_alpha=10.788188919840913, subsample=0.47437582748953966, num_leaves=64, reg_lambda=95.75401894533888, n_estimators=10000, random_state=29, boosting_type='gbdt', learning_rate=0.00934485794382918, colsample_bytree=0.39283351290380497, min_child_weight=0.24081152127177283, min_child_samples=40, device='gpu')
]

cb_params = [
    dict(iterations=8000, depth=7, l2_leaf_reg=2.0, min_data_in_leaf=15, border_count=254, loss_function="RMSE", task_type="GPU", devices="0", od_type="Iter", od_wait=300, verbose=0, learning_rate=0.020, random_seed=7),
    dict(iterations=8000, depth=7, l2_leaf_reg=2.0, min_data_in_leaf=15, border_count=254, loss_function="RMSE", task_type="GPU", devices="0", od_type="Iter", od_wait=300, verbose=0, learning_rate=0.030, random_seed=123)
]

ridge_params = {"random_state": 42, "alpha": 1.6602834637650032, "tol": 0.0005030247295617308, "positive": True, "fit_intercept": True}
pp_params = {'alpha': 0.985, 'tau': 120.0, 'w_pf': 0.05}