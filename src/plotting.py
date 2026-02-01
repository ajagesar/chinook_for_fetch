import pandas as pd
import numpy as np
from utils.helpers import ROOT, config
import yaml
import matplotlib.pyplot as plt


EVAL_LABELS = ["all_populations", "vs_ap4_cabg", "vs_ap4_noncabg"]


def main():

    # --- existing per-population plots (unchanged) --- #
    eval_labels = ["all_populations"]
    if config['splitting_method']['nice'] == True:
        eval_labels += ["vs_ap4_cabg", "vs_ap4_noncabg"]

    for eval_label in eval_labels:
        results_path = ROOT + f'data/final/{eval_label}_results.yaml'
        with open(results_path, 'r') as f:
            eval_results = yaml.safe_load(f)

        for model in eval_results:
            plot_roc(eval_results[model], model, eval_label)
            plot_precision_recall(eval_results[model], model, eval_label)
            plot_calibration(eval_results[model], model, eval_label)

    # --- incremental training performance plot --- #
    plot_incremental_performance()

    return


# ===========================================================================
# Existing plots (unchanged)
# ===========================================================================

def _save_path(model, eval_label, plot_type):
    return ROOT + f'data/final/plots/{eval_label}_{model}_{plot_type}.png'


def plot_roc(model_results, model, eval_label):
    fig, ax = plt.subplots(figsize=(8, 6))

    auc_model = model_results['auc_model']
    ax.plot(model_results['fpr'], model_results['tpr'],
            color='blue', label=f'ROC curve (AUC = {auc_model:.2f})')
    ax.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random guess')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve — {eval_label}')
    ax.legend(loc="lower right")

    fig.savefig(_save_path(model, eval_label, 'auc_plot'))
    plt.close(fig)


def plot_precision_recall(model_results, model, eval_label):
    fig, ax = plt.subplots(figsize=(8, 6))

    f1_model = model_results['f1_model']
    ax.plot(model_results['recall'], model_results['precision'],
            marker='o', color='b',
            label=f'Precision-Recall Curve (F1 = {f1_model:.2f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curve — {eval_label}')
    ax.legend(loc="lower right")

    fig.savefig(_save_path(model, eval_label, 'precision_recall_curve'))
    plt.close(fig)


def plot_calibration(model_results, model, eval_label):
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(model_results['prob_pred'], model_results['prob_true'],
            "s-", label="Calibration Curve")
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(f'Calibration Curve — {eval_label}')
    ax.legend()

    fig.savefig(_save_path(model, eval_label, 'calibration_curve'))
    plt.close(fig)


# ===========================================================================
# Incremental training performance chart
# ===========================================================================

def plot_incremental_performance():
    """Read incremental_results.yaml and produce a two-panel figure:

        Top    — AUC per model as training data grows
        Bottom — Brier score per model (lower = better)

    X-axis is training-set size (continuous).  A vertical dashed line marks
    each step boundary and is labelled with only the origin that was *added*
    at that step — no text overlap regardless of how many origins there are.
    """

    results_path = ROOT + 'data/final/incremental_results.yaml'
    with open(results_path, 'r') as f:
        incremental_results = yaml.safe_load(f)          # {model: [{step…}]}

    if not incremental_results:
        print("incremental_results.yaml is empty — skipping plot.")
        return

    # --- flatten into a tidy DataFrame --- #
    rows = []
    for model_label, steps in incremental_results.items():
        for step in steps:
            rows.append({
                'model':             model_label,
                'step_idx':          step['step_idx'],
                'origins_included':  step['origins_included'],
                'n_train':           step['n_train'],
                'auc':               step['auc'],
                'f1':                step['f1'],
                'brier_score':       step['brier_score'],
            })

    df = pd.DataFrame(rows).sort_values(['model', 'step_idx'])

    # --- step metadata (one row per step, shared across models) --- #
    ref_model  = df['model'].iloc[0]
    step_info  = (df[df['model'] == ref_model]
                  .drop_duplicates('step_idx')
                  .sort_values('step_idx')
                  .reset_index(drop=True))

    # Derive "origin added at this step" by diffing the origins_included lists.
    # Step 0 added everything in its list (the first origin); every later step
    # added exactly one new origin.
    added_at_step = []
    for i, row in step_info.iterrows():
        if i == 0:
            # first step: the single origin that starts the sequence
            added_at_step.append(row['origins_included'][0])
        else:
            prev = set(step_info.loc[i - 1, 'origins_included'])
            curr = set(row['origins_included'])
            new  = curr - prev
            added_at_step.append(new.pop() if new else '?')
    step_info['added'] = added_at_step

    # --- colour / marker maps --- #
    models     = df['model'].unique()
    cmap       = plt.cm.get_cmap('tab10', len(models))
    colors     = {m: cmap(i) for i, m in enumerate(models)}
    markers    = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
    marker_map = {m: markers[i % len(markers)] for i, m in enumerate(models)}

    # --- figure --- #
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle('Model Performance vs. Incrementally Added Data Sources',
                 fontsize=14, fontweight='bold', y=0.98)

    # ---- helper: draw vertical step lines + top labels on one axis ---- #
    def draw_step_lines(ax, label_on_top=False):
        for _, row in step_info.iterrows():
            ax.axvline(row['n_train'], color='#aaaaaa', linestyle='--',
                       linewidth=0.9, zorder=0)
            if label_on_top:
                # place the "added" label at the very top of the axes,
                # rotated so it never overlaps its neighbour
                ax.text(row['n_train'], ax.get_ylim()[1],
                        f"+ {row['added']}",
                        fontsize=7.5, color='#555555',
                        ha='center', va='bottom', rotation=30)

    # ---- Top panel: AUC ---- #
    ax = axes[0]
    for model in models:
        sub = df[df['model'] == model].sort_values('step_idx')
        ax.plot(sub['n_train'], sub['auc'],
                marker=marker_map[model], color=colors[model],
                label=model, linewidth=2, markersize=7, zorder=1)

    ax.axhline(0.5, color='grey', linestyle=':', linewidth=1, label='Random (0.5)')
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_ylim(bottom=0.4)
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax.grid(axis='y', linestyle=':', alpha=0.4)
    ax.set_title('AUC (higher = better)', fontsize=11, pad=8)

    # draw lines first, then nudge ylim up slightly so the rotated labels
    # have breathing room above the plot area
    draw_step_lines(ax, label_on_top=True)
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi + (yhi - ylo) * 0.12)   # 12 % extra headroom for labels

    # ---- Bottom panel: Brier score ---- #
    ax = axes[1]
    for model in models:
        sub = df[df['model'] == model].sort_values('step_idx')
        ax.plot(sub['n_train'], sub['brier_score'],
                marker=marker_map[model], color=colors[model],
                label=model, linewidth=2, markersize=7, zorder=1)

    ax.set_ylabel('Brier Score', fontsize=12)
    ax.set_xlabel('Training set size', fontsize=12)
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    ax.grid(axis='y', linestyle=':', alpha=0.4)
    ax.set_title('Brier Score (lower = better)', fontsize=11, pad=8)
    draw_step_lines(ax, label_on_top=False)   # lines only, no labels here

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = ROOT + 'data/final/plots/incremental_performance.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)

    print(f"\nIncremental performance plot saved to {out}\n")


if __name__ == "__main__":
    main()