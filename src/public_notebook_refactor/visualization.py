import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from config import CFG
from processing_helpers import load_well_v2

def plot_performance_metrics(*, fold_scores, overall_scores):
    """
    Draws a validation report plotting fold variance and absolute RMSE errors.

    Args:
        fold_scores (dict): Dictionary tracking baseline cross-validation fold errors.
        overall_scores (dict): Set detailing out-of-fold evaluated metrics per model structure.

    Returns:
        None
    """
    fold_scores_df = pd.DataFrame(fold_scores)
    overall_scores_df = pd.DataFrame({k: [v] for k, v in overall_scores.items()}).transpose().sort_values(by=0, ascending=True)
    order = overall_scores_df.index.tolist()

    min_score = overall_scores_df.values.flatten().min()
    max_score = overall_scores_df.values.flatten().max()
    padding = (max_score - min_score) * 0.5
    lower_limit = min_score - padding
    upper_limit = max_score + padding

    fig, axs = plt.subplots(1, 2, figsize=(15, fold_scores_df.shape[1] * 0.5))

    sns.boxplot(data=fold_scores_df, order=order, ax=axs[0], orient="h", color="grey")
    axs[0].set_title(f"Fold RMSE")
    axs[0].set_xlabel("")
    axs[0].set_ylabel("")

    barplot = sns.barplot(x=overall_scores_df.values.flatten(), y=overall_scores_df.index, ax=axs[1], color="grey")
    axs[1].set_title(f"Overall RMSE")
    axs[1].set_xlabel("")
    axs[1].set_xlim(left=lower_limit, right=upper_limit)
    axs[1].set_ylabel("")

    for i, (score, model) in enumerate(zip(overall_scores_df.values.flatten(), overall_scores_df.index)):
        color = "cyan" if "ridge" in model.lower() else "grey"
        barplot.patches[i].set_facecolor(color)
        axs[0].patches[i].set_facecolor(color) if i < len(axs[0].patches) else None
        barplot.text(score, i, round(score, 3), va="center")

    plt.tight_layout()
    plt.show()

def fig_overview(wid):
    hw, tw = load_well_v2(wid=wid, split='train')
    kn = hw[hw.TVT_input.notna()]; ev = hw[hw.TVT_input.isna()]; ps = kn.MD.iloc[-1]
    fig, ax = plt.subplots(3, 1, figsize=(12, 8.5), sharex=True)
    ax[0].plot(hw.MD, hw.Z, lw=1.2, color="#333"); ax[0].axvline(ps, color="crimson", ls="--", label="PS")
    ax[0].set_ylabel("Z / TVD (ft)"); ax[0].legend(loc="upper right")
    ax[0].set_title(f"Well {wid}: trajectory · gamma-ray · TVT target")
    ax[1].plot(kn.MD, kn.GR, lw=.7, color="steelblue", label="GR known")
    ax[1].plot(ev.MD, ev.GR, lw=.7, color="darkorange", label="GR eval"); ax[1].axvline(ps, color="crimson", ls="--")
    ax[1].set_ylabel("GR (API)"); ax[1].legend(loc="upper right")
    ax[2].plot(kn.MD, kn.TVT, lw=1.6, color="seagreen", label="TVT known (=input)")
    ax[2].plot(ev.MD, ev.TVT, lw=1.6, color="crimson", label="TVT to predict"); ax[2].axvline(ps, color="crimson", ls="--")
    ax[2].set_ylabel("TVT (ft)"); ax[2].set_xlabel("MD (ft)"); ax[2].invert_yaxis(); ax[2].legend(loc="upper right")
    for a in ax: a.grid(alpha=.25)
    plt.tight_layout(); plt.show()

def fig_correlation(wid):
    hw, tw = load_well_v2(wid=wid, split='train'); ev = hw[hw.TVT_input.isna()]
    fig, ax = plt.subplots(1, 2, figsize=(11, 6))
    ax[0].plot(tw.GR, tw.TVT, lw=1.0, color="black")
    ax[0].set_xlabel("GR (API)"); ax[0].set_ylabel("TVT (ft)"); ax[0].invert_yaxis()
    ax[0].set_title("Typewell signature: GR vs TVT")
    sc = ax[1].scatter(ev.GR, ev.TVT, s=4, c=ev.MD, cmap="viridis")
    ax[1].set_xlabel("GR (API)"); ax[1].set_ylabel("TVT (ft)"); ax[1].invert_yaxis()
    ax[1].set_title("Horizontal GR at its true TVT\nmatches the typewell signature")
    plt.colorbar(sc, ax=ax[1], label="MD (ft)")
    for a in ax: a.grid(alpha=.25)
    plt.tight_layout(); plt.show()