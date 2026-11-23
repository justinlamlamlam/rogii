import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from config import CFG

def sg_smooth(df, col, sg_w=17, sg_p=3):
    df = df.copy()
    for _, g in df.groupby('well', sort=False):
        v = g[col].values
        n = len(v)
        wl = min(sg_w, n)
        if wl % 2 == 0: 
            wl -= 1
        if wl >= sg_p + 2: 
            v = savgol_filter(v, wl, sg_p)
        df.loc[g.index, col] = v
    return df

def _robfit(s, y, deg=3):
    if len(s) < deg + 2:
        return y.copy()
    c = np.polyfit(s, y, deg)
    for _ in range(4):
        r = y - np.polyval(c, s)
        sc = np.median(np.abs(r)) * 1.4826 + 1e-6
        c = np.polyfit(s, y, deg, w=1.0 / (1.0 + (r / (2.0 * sc)) ** 2))
    return np.polyval(c, s)

def apply_projection_post_processing(*, submission_path="submission.csv"):
    """
    Overwrites raw blends using robust low-order polynomial track projections.

    Args:
        submission_path (str): Target filesystem path to output file.

    Returns:
        None
    """
    try:
        _base = pd.read_csv(submission_path)
        assert set(['id','tvt']).issubset(_base.columns)
        _base['well'] = _base['id'].str[:8]
        _base['row_idx'] = _base['id'].str[9:].astype(int)
        _out = dict(zip(_base['id'].values, _base['tvt'].astype(float).values))
        _n_ok = 0
        for _wid, _g in _base.groupby('well'):
            try:
                _hw = pd.read_csv(CFG.dataset_path / 'test' / (_wid + '__horizontal_well.csv'))
                _kn = _hw[_hw['TVT_input'].notna()]
                if len(_kn) < 5:
                    continue
                _last = _kn.iloc[-1]
                _anchor = float(_last['TVT_input']) + float(_last['Z'])
                _ps = float(_last['MD']); _end = float(_hw['MD'].iloc[-1])
                _gi = _g.sort_values('row_idx')
                _ri = _gi['row_idx'].values
                _Z = _hw['Z'].values[_ri].astype(float)
                _md = _hw['MD'].values[_ri].astype(float)
                _s = (_md - _ps) / max(_end - _ps, 1e-6)
                _tvt = _gi['tvt'].values.astype(float)
                _fit = _robfit(_s, (_tvt + _Z) - _anchor, 3)
                _tvt_fit_full = (_anchor + _fit) - _Z
                _tvt_fit = 0.25 * _tvt + 0.75 * _tvt_fit_full
                if not np.all(np.isfinite(_tvt_fit)):
                    continue
                for _rid, _val in zip(_gi['id'].values, _tvt_fit):
                    _out[_rid] = float(_val)
                _n_ok += 1
            except Exception as _e:
                print('proj fallback', _wid, _e)
        print('projection applied to', _n_ok, 'wells')
        _final = _base[['id']].copy()
        _final['tvt'] = _final['id'].map(_out).astype(float)
        _final[['id','tvt']].to_csv(submission_path, index=False)
        print('wrote projected submission.csv', _final.shape)
    except Exception as _e:
        print('PROJECTION SKIPPED (kept blended submission):', _e)