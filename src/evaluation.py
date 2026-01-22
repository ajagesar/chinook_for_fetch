
import pandas as pd
import numpy as np
from utils.helpers import ROOT, config
import pickle
import yaml

import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import roc_curve, precision_recall_curve, auc, f1_score
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
import torch


def main():

    # load X_test and y_test
    X_test = pd.read_csv(ROOT + 'data/intermediate/X_test_imputed.csv')
    y_test = pd.read_csv(ROOT + 'data/intermediate/y_test.csv')['in_icu_death']

    # evaluate with yaml file configs
    model_algorithms = config['model_algorithms']

    dict_of_results = {}

    for model_algorithm in model_algorithms:

        # load model
        filename = model_algorithms[model_algorithm]['filename']
        with open(ROOT + f'data/final/models/{filename}', 'rb') as file:
            model = pickle.load(file)

        # convert to tensor if working with the neural network
        if model_algorithm == 'nn':
            X_test = torch.tensor(X_test.to_numpy(), dtype=torch.float32)

        y_pred_proba = model.predict_proba(X_test)[:,1]
        y_pred = model.predict(X_test)

        # Calculate the AUC
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        auc_model = auc(fpr, tpr)

        # calculate the F1
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        f1_model = f1_score(y_test, y_pred)

        # calculate calibration values
        prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)

        results = {
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds,
                'auc_model': auc_model,
                'precision': precision,
                'recall': recall,
                'f1_model': f1_model,
                'prob_true': prob_true,
                'prob_pred': prob_pred
            }

        dict_of_results[model_algorithms[model_algorithm]['label']] = results

    # evaluate APACHE IV
    evaluate_apacheiv(X_test=X_test, y_test=y_test)

    # export list for results table
    ## convert to serializable format (np.floats are not serializable, native python floats are)
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert numpy array to list
        elif isinstance(obj, np.float64) or isinstance(obj, np.float32):
            return float(obj)  # Convert numpy float to Python float
        elif isinstance(obj, np.int64) or isinstance(obj, np.int32):
            return int(obj)  # Convert numpy int to Python int
        else:
            return obj  # Return the object as-is if no conversion is needed
        
    dict_of_results_serializable = {
        key: {k: convert_to_serializable(v) for k, v in value.items()}
        for key, value in dict_of_results.items()
    }

    with open(ROOT + 'data/final/results.yaml', 'w') as f:
        yaml.safe_dump(dict_of_results_serializable, f)

    
    """ 
    list documentation:

    list_of_models = [
        [filename_of_model[0], model_label[1], model_load_from_pickle[2], fpr[3], tpr[4], thresholds[5], auc_model[6], precision[7], recall[8], f1_model[9]]
    ]
    """

    return


def evaluate_apacheiv(X_test, y_test):

    # join nice data with test set
    apache_data = pd.read_csv(ROOT + "data/intermediate/nice/aumc")

    X_test = pd.merge(left=X_test, right=apache_data, how="left", on=["person_id", "visit_occurrence_id"])

    # evaluate logodds & probability with y_test

    # recalibrate
    apache_recalibrated = LogisticRegression(penalty='none', max_iter=10000)
    # apache_recalibrated.fit()

    return


if __name__=="__main__":
    main()