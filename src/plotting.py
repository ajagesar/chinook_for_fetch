
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


def main():

    # load yaml results file
    with open (ROOT + 'data/final/results.yaml', 'r') as f:
        eval_results = yaml.safe_load(f)
    
    for model in eval_results:
        # plot ROC curve
        plt.cla()
        plt.figure(figsize=(8,6))

        auc_model = eval_results[model]['auc_model']
        plt.plot(
            eval_results[model]['fpr'],
            eval_results[model]['tpr'],
            color='blue',
            label=f'ROC curve (AUC = {auc_model:.2f})'
        )
        
        plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random guess')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig(ROOT + f'data/final/plots/{model}_auc_plot.png') # model label

        # Plot precision recall curve
        plt.cla()
        plt.figure(figsize=(8,6))

        f1_model = eval_results[model]['f1_model']
        plt.plot(
            eval_results[model]['recall'], # recall
            eval_results[model]['precision'], # precision
            marker='o',
            color='b',
            label=f'Precision-Recall Curve (F1-score = {f1_model:.2f})' # f1_model
        )
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower right")
        plt.savefig(ROOT + f'data/final/plots/{model}_precision_recall_curve.png') # model label

        # TODO plot calibration curve
        plt.cla()
        plt.figure(figsize=(8,6))
        
        plt.plot(
            eval_results[model]['prob_pred'],
            eval_results[model]['prob_true'],
            "s-",
            label="Calibration Curve"
        )
        plt.plot(
            [0, 1],
            [0, 1],
            "k--",
            label="Perfectly calibrated"
        )
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title("Calibration Curve")
        plt.legend()
        plt.savefig(ROOT + f'data/final/plots/{model}_calibration_curve.png')


    return


""" 
list_of_models = [
    [filename_of_model[0], model_label[1], model_load_from_pickle[2], fpr[3], tpr[4], thresholds[5], auc_model[6], precision[7], recall[8], f1_model[9]]
]
"""

if __name__=="__main__":
    main()