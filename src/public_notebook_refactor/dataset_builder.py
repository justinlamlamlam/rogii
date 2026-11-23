import os
import time
from pathlib import Path
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from config import *
from processing_helpers import (
    run_pf_ancc, run_pf_z, beam_search, multi_scale_ncc, 
    affine_cal, robust_slope, FormationPlaneKNN, DenseANCCImputer
)

# Initialize imputers globally within builder module context
hw_paths = sorted((CFG.dataset_path / "train").glob('*__horizontal_well.csv'))
train_wids = [p.stem.replace('__horizontal_well','') for p in hw_paths]
FI = FormationPlaneKNN(train_wids, CFG.dataset_path / "train")
DI = DenseANCCImputer(train_wids, CFG.dataset_path / "train")

def build_well(hw_path, tw_path, is_train):
    global FI, DI
    wid = Path(hw_path).stem.replace('__horizontal_well','')
    try:
        hw = pd.read_csv(hw_path); tw = pd.read_csv(tw_path).sort_values('TVT')
    except: return None
    if is_train and 'TVT' not in hw.columns: return None
    kn = hw[hw['TVT_input'].notna()]; ev = hw[hw['TVT_input'].isna()]
    if len(ev) == 0 or len(kn) < 10: return None
    if is_train and hw['TVT'].isna().all(): return None
    tw_tvt = tw['TVT'].to_numpy(np.float32); tw_gr = tw['GR'].to_numpy(np.float32)
    if len(tw_tvt) < 3: return None

    pf_a, std_a = run_pf_ancc(hw, tw_tvt, tw_gr)
    if len(pf_a) == 0: return None
    pf_z, std_z = run_pf_z(hw, tw_tvt, tw_gr)
    pf_use = pf_a.astype(np.float32); std_use = std_a.astype(np.float32)
    has_z = len(pf_z) == len(pf_a) and not np.any(np.isnan(pf_z))

    lk = kn.iloc[-1]; last_tvt = float(lk['TVT_input'])
    gr_full = hw['GR'].astype(float).interpolate(limit_direction='both').fillna(float(np.nanmean(tw_gr)))
    hgr = gr_full.iloc[ev.index[0]:].to_numpy(np.float32)
    kgr = gr_full.iloc[:len(kn)].to_numpy(np.float32)

    bpaths = {}
    for (bs, mc, es, r, tag) in BEAMS:
        bpaths[tag] = beam_search(gr_h=hgr, tw_tvt=tw_tvt, tw_gr=tw_gr, start_tvt=last_tvt, bs=bs, mc=mc, es=es, r=r)
    beam_ref = (bpaths['cons'] + bpaths['sm5']) / 2.

    ktvt = kn['TVT_input'].to_numpy(np.float32)
    sc_res, sc_ens = multi_scale_ncc(kgr, ktvt, hgr, hws=(8, 15, 25), stride=3)
    sc8, sc8s = sc_res[0]; sc15, sc15s = sc_res[1]; sc25, sc25s = sc_res[2]
    sc_cons = (sc8 + sc15 + sc25) / 3.
    sc_trust = float(np.clip(len(kn) / 200., 0., 0.6))
    hyb_ref = (1 - sc_trust) * beam_ref + sc_trust * sc_ens

    tw_at_k = np.interp(ktvt, tw_tvt, tw_gr).astype(np.float32)
    a_cal, b_cal = affine_cal(kgr, tw_at_k)
    kmd = kn['MD'].to_numpy(np.float32); kz = kn['Z'].to_numpy(np.float32)
    pfx_rmse = float(np.sqrt(np.mean((kgr - tw_at_k) ** 2)))
    slp_all = robust_slope(kmd, ktvt); slp_50 = robust_slope(kmd[-50:], ktvt[-50:])
    slp_z = robust_slope(kz, ktvt)

    swid = wid if is_train else None
    xy_ev = ev[['X', 'Y']].to_numpy(np.float64); xy_kn = kn[['X', 'Y']].to_numpy(np.float64)
    form_ev, knn_d = FI.impute(xy_ev, self_wid=swid)
    form_kn, _ = FI.impute(xy_kn, self_wid=swid)
    z_kn = kn['Z'].to_numpy(np.float32); z_ev = ev['Z'].to_numpy(np.float32)

    tvt_fs = {}; form_rmse = {}; form_list = []
    for fi2, fn in enumerate(FORMATIONS):
        b_full, b_early, b_mid, b_late, b_wls = seg_b_well(ktvt, z_kn, form_kn[:, fi2])
        tvt_f = (-z_ev + form_ev[:, fi2] + b_full).astype(np.float32)
        tvt_fw = (-z_ev + form_ev[:, fi2] + b_wls).astype(np.float32)
        tvt_f50 = (-z_ev + form_ev[:, fi2] + b_late).astype(np.float32)
        tvt_fs[f'tvtF_{fn}'] = tvt_f; tvt_fs[f'tvtFw_{fn}'] = tvt_fw
        tvt_fs[f'tvtF50_{fn}'] = tvt_f50
        tvt_fs[f'bw_{fn}'] = np.float32(b_full); tvt_fs[f'bww_{fn}'] = np.float32(b_wls)
        tvt_fs[f'bw50_{fn}'] = np.float32(b_late)
        tvt_fs[f'bw_early_{fn}'] = np.float32(b_early)
        tvt_fs[f'bw_mid_{fn}'] = np.float32(b_mid)
        form_rmse[fn] = float(np.sqrt(np.mean((ktvt - (-z_kn + form_kn[:, fi2] + b_full)) ** 2)))
        form_list.append(tvt_f)

    fs = np.stack(form_list, 1)
    form_mean_d = (fs.mean(1) - last_tvt).astype(np.float32)
    form_std_d = fs.std(1).astype(np.float32)
    form_rng_d = (fs.max(1) - fs.min(1)).astype(np.float32)

    d_ancc, d_std, d_dist = DI.impute(xy_ev, self_wid=swid)
    d_kn, d_std_kn, _ = DI.impute(xy_kn, self_wid=swid)
    b_vd = ktvt + z_kn - d_kn
    _, b_de, b_dm, b_dl, b_dw = seg_b_well(ktvt, z_kn, d_kn)
    b_d = float(np.median(b_vd))
    tvt_dense = (-z_ev + d_ancc + b_d).astype(np.float32)
    tvt_densew = (-z_ev + d_ancc + b_dw).astype(np.float32)
    tvt_dense50 = (-z_ev + d_ancc + b_dl).astype(np.float32)
    res_kn = ktvt + z_kn - d_kn
    d_rmse = float(np.sqrt(np.mean(res_kn ** 2))); d_bias = float(np.mean(res_kn)); d_nb_std = float(np.mean(d_std_kn))

    all_sigs = [pf_use] + [p for p in bpaths.values()] + [sc8, sc15, sc25, sc_ens, tvt_fs['tvtF_ANCC'], tvt_dense]
    sig_mat = np.stack(all_sigs, 1)
    sig_std = sig_mat.std(1).astype(np.float32)
    sig_mean = (sig_mat.mean(1) - last_tvt).astype(np.float32)

    gr_s = pd.Series(gr_full.values); rolls = {}
    for w in [5, 21, 51, 101]:
        r = gr_s.rolling(w, center=True, min_periods=1)
        rolls[f'grm{w}'] = r.mean().iloc[ev.index].values.astype(np.float32)
        rolls[f'grs{w}'] = r.std().fillna(0).iloc[ev.index].values.astype(np.float32)
    for lag in [1, 5, 15, 30]:
        rolls[f'glag{lag}'] = gr_s.shift(lag).bfill().iloc[ev.index].values.astype(np.float32)
        rolls[f'glead{lag}'] = gr_s.shift(-lag).ffill().iloc[ev.index].values.astype(np.float32)
    gr_d1 = gr_s.diff().fillna(0.).iloc[ev.index].values.astype(np.float32)
    gr_d2 = gr_s.diff().diff().fillna(0.).iloc[ev.index].values.astype(np.float32)
    gr_env = gr_s.rolling(21, center=True, min_periods=1).max().iloc[ev.index].values.astype(np.float32)
    gr_nrg = np.sqrt(np.maximum((gr_s ** 2).rolling(21, center=True, min_periods=1).mean(), 0.)).iloc[ev.index].values.astype(np.float32)

    hmd = ev['MD'].to_numpy(np.float32); md_since = hmd - float(lk['MD'])
    slp_b_all = (last_tvt + slp_all * md_since).astype(np.float32)
    slp_b_50 = (last_tvt + slp_50 * md_since).astype(np.float32)

    mdd = hw['MD'].diff().replace(0, np.nan)
    dzdmd = (hw['Z'].diff() / mdd).iloc[ev.index].values.astype(np.float32)
    dxdmd = (hw['X'].diff() / mdd).iloc[ev.index].values.astype(np.float32)
    dydmd = (hw['Y'].diff() / mdd).iloc[ev.index].values.astype(np.float32)

    nh = len(ev); frac = (np.arange(nh) / max(nh - 1, 1)).astype(np.float32)
    def sc(v): return np.full(nh, np.float32(v), np.float32)

    feats = {
        'well': wid, 'id': [f'{wid}_{i}' for i in ev.index],
        'last_known_tvt': sc(last_tvt),
        'pf_ancc': pf_use, 'pf_ancc_std': std_use,
        'pf_ancc_delta': (pf_use - last_tvt).astype(np.float32),
        'pf_z': (pf_z.astype(np.float32) if has_z else sc(last_tvt)),
        'pf_z_delta': ((pf_z - last_tvt).astype(np.float32) if has_z else sc(0.)),
        'pf_vs_z': ((pf_use - pf_z.astype(np.float32)) if has_z else sc(0.)),
        **{f'beam_{t}_d': (p - np.float32(last_tvt)).astype(np.float32) for t, p in bpaths.items()},
        'beam_mean_d': np.stack([(p - last_tvt) for p in bpaths.values()], 1).mean(1).astype(np.float32),
        'beam_std_d': np.stack([(p - last_tvt) for p in bpaths.values()], 1).std(1).astype(np.float32),
        'beam_med_d': np.median(np.stack([(p - last_tvt) for p in bpaths.values()], 1), 1).astype(np.float32),
        'sc8_d': (sc8 - np.float32(last_tvt)).astype(np.float32), 'sc8_sc': sc8s,
        'sc15_d': (sc15 - np.float32(last_tvt)).astype(np.float32), 'sc15_sc': sc15s,
        'sc25_d': (sc25 - np.float32(last_tvt)).astype(np.float32), 'sc25_sc': sc25s,
        'sc_cons_d': (sc_cons - np.float32(last_tvt)).astype(np.float32),
        'sc_ens_d': (sc_ens - np.float32(last_tvt)).astype(np.float32),
        'sc_trust': sc(sc_trust), 'hyb_d': (hyb_ref - np.float32(last_tvt)).astype(np.float32),
        'sig_std': sig_std, 'sig_mean_d': sig_mean,
        **tvt_fs,
        **{f'frm_rmse_{fn}': sc(form_rmse[fn]) for fn in FORMATIONS},
        'form_mean_d': form_mean_d, 'form_std_d': form_std_d, 'form_rng_d': form_rng_d,
        'spatial_ancc_d': (form_ev[:, 0] - np.float32(np.interp(last_tvt, tw_tvt, tw_gr))),
        'spatial_knn_dist': knn_d,
        'dense_ancc': d_ancc, 'dense_std': d_std, 'dense_dist': d_dist,
        'tvt_dense_d': (tvt_dense - last_tvt).astype(np.float32),
        'tvt_densew_d': (tvt_densew - last_tvt).astype(np.float32),
        'tvt_dense50_d': (tvt_dense50 - last_tvt).astype(np.float32),
        'dense_rmse': sc(d_rmse), 'dense_bias': sc(d_bias), 'dense_nb_std': sc(d_nb_std),
        'pf_vs_spatial': (pf_use - tvt_fs['tvtF_ANCC']).astype(np.float32),
        'pf_vs_dense': (pf_use - tvt_dense).astype(np.float32),
        'spatial_vs_dense': (tvt_fs['tvtF_ANCC'] - tvt_dense).astype(np.float32),
        'beam_vs_spatial': (bpaths['cons'] - tvt_fs['tvtF_ANCC']).astype(np.float32),
        'sc_vs_beam': (sc_ens - bpaths['cons']).astype(np.float32),
        'cal_a': sc(a_cal), 'cal_b': sc(b_cal),
        'pfx_rmse': sc(pfx_rmse), 'known_len': sc(len(kn)), 'eval_len': sc(nh),
        'slp_all': sc(slp_all), 'slp_50': sc(slp_50), 'slp_z': sc(slp_z),
        'slp_b_d_all': (slp_b_all - last_tvt).astype(np.float32),
        'slp_b_d_50': (slp_b_50 - last_tvt).astype(np.float32),
        'ktvt_range': sc(float(np.ptp(ktvt))), 'ktvt_std': sc(float(ktvt.std())),
        'md_since': md_since, 'frac': frac, 'frac2': frac ** 2, 'sqrt_frac': np.sqrt(frac),
        'z': z_ev,
        'dx': (ev['X'] - float(lk['X'])).to_numpy(np.float32),
        'dy': (ev['Y'] - float(lk['Y'])).to_numpy(np.float32),
        'dz': (z_ev - float(lk['Z'])).astype(np.float32),
        'dxy': np.sqrt((ev['X'] - float(lk['X'])) ** 2 + (ev['Y'] - float(lk['Y'])) ** 2).to_numpy(np.float32),
        'dzdmd': dzdmd, 'dxdmd': dxdmd, 'dydmd': dydmd,
        'gr': hgr, 'gr_d1': gr_d1, 'gr_d2': gr_d2, 'gr_env': gr_env, 'gr_nrg': gr_nrg,
        'gr_vs_tw_anc': hgr - np.float32(np.interp(last_tvt, tw_tvt, tw_gr)),
        'gr_vs_slp_all': hgr - np.interp(slp_b_all, tw_tvt, tw_gr).astype(np.float32),
        **{f'tda{int(o)}': hgr - np.float32(np.interp(last_tvt + o, tw_tvt, tw_gr)) for o in ANCH_OFFS},
        **{f'tdbc{int(o)}': hgr - np.interp(beam_ref + o, tw_tvt, tw_gr).astype(np.float32) for o in BEAM_OFFS},
        **{f'tdsc{int(o)}': hgr - np.interp(sc_ens + o, tw_tvt, tw_gr).astype(np.float32) for o in SC_OFFS},
        **{f'tdpf{int(o)}': hgr - np.interp(pf_use + o, tw_tvt, tw_gr).astype(np.float32) for o in PF_OFFS},
        'tw_range': sc(float(np.ptp(tw_tvt))), 'tw_gr_mean': sc(float(tw_gr.mean())),
    }
    for k, v in rolls.items(): feats[k] = v
    result = pd.DataFrame(feats)
    if is_train:
        if 'TVT' not in ev.columns or ev['TVT'].isna().all(): return None
        result['target'] = (ev['TVT'].to_numpy(np.float32) - np.float32(last_tvt))
    return result

def build_dataset(paths, is_train, label):
    args = [(str(p), str(p.parent / f'{p.stem.replace("__horizontal_well","")}__typewell.csv'), is_train)
            for p in paths
            if (p.parent / f'{p.stem.replace("__horizontal_well","")}__typewell.csv').exists()]
    t0 = time.time()
    res = Parallel(n_jobs=NCPU, prefer='threads', verbose=3)(
        delayed(build_well)(hp, tp, it) for hp, tp, it in args)
    parts = [r for r in res if r is not None]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()