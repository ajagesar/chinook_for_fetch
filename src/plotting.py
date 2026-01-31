import pandas as pd
import numpy as np
from utils.helpers import ROOT, config
import pickle
import yaml

import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import roc_curve, precision_recall_curve, auc, f1_score
from sklearn.calibration import calibration_curve
import torch


EVAL_LABELS = ["all_populations", "vs_ap4_cabg", "vs_ap4_noncabg"]


def main():

    # Determine which result files to plot based on config
    eval_labels = ["all_populations"]
    if config['splitting_method']['nice'] == True:
        eval_labels += ["vs_ap4_cabg", "vs_ap4_noncabg"]

    for eval_label in eval_labels:

        # load yaml results file
        results_path = ROOT + f'data/final/{eval_label}_results.yaml'
        with open(results_path, 'r') as f:
            eval_results = yaml.safe_load(f)

        for model in eval_results:
            plot_roc(eval_results[model], model, eval_label)
            plot_precision_recall(eval_results[model], model, eval_label)
            plot_calibration(eval_results[model], model, eval_label)

    return


def _save_path(model, eval_label, plot_type):
    return ROOT + f'data/final/plots/{eval_label}_{model}_{plot_type}.png'


def plot_roc(model_results, model, eval_label):
    fig, ax = plt.subplots(figsize=(8, 6))

    auc_model = model_results['auc_model']
    ax.plot(
        model_results['fpr'],
        model_results['tpr'],
        color='blue',
        label=f'ROC curve (AUC = {auc_model:.2f})'
    )

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
    ax.plot(
        model_results['recall'],
        model_results['precision'],
        marker='o',
        color='b',
        label=f'Precision-Recall Curve (F1 = {f1_model:.2f})'
    )

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curve — {eval_label}')
    ax.legend(loc="lower right")

    fig.savefig(_save_path(model, eval_label, 'precision_recall_curve'))
    plt.close(fig)


def plot_calibration(model_results, model, eval_label):
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        model_results['prob_pred'],
        model_results['prob_true'],
        "s-",
        label="Calibration Curve"
    )
    ax.plot(
        [0, 1],
        [0, 1],
        "k--",
        label="Perfectly calibrated"
    )

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(f'Calibration Curve — {eval_label}')
    ax.legend()

    fig.savefig(_save_path(model, eval_label, 'calibration_curve'))
    plt.close(fig)


if __name__ == "__main__":
    main()