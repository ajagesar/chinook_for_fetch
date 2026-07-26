import pandas as pd
import numpy as np
from utils.helpers import ROOT, config
import pickle
import yaml

import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve, auc, f1_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
import torch
from tableone import TableOne

random_state = 42


def main():

    # load X_test and y_test
    X_test = pd.read_csv(ROOT + 'data/intermediate/X_test_imputed.csv')
    y_test = pd.read_csv(ROOT + 'data/intermediate/y_test.csv')['in_icu_death']

    get_metrics(X_test=X_test, y_test=y_test, eval_label="all_populations")

    if config['splitting_method']['nice'] == True:

        X_test_noncabg = pd.read_csv(ROOT + 'data/intermediate/X_test_noncabg.csv')
        y_test_noncabg = pd.read_csv(ROOT + 'data/intermediate/y_test_noncabg.csv')['in_icu_death']

        X_test_cabg = pd.read_csv(ROOT + 'data/intermediate/X_test_cabg.csv')
        y_test_cabg = pd.read_csv(ROOT + 'data/intermediate/y_test_cabg.csv')['in_icu_death']

        get_metrics(X_test=X_test_noncabg, y_test=y_test_noncabg, eval_label="vs_ap4_noncabg")
        get_metrics(X_test=X_test_cabg, y_test=y_test_cabg, eval_label="vs_ap4_cabg")


    return


def get_table_1():

    train = pd.read_csv(ROOT + "data/intermediate/train_data_complete.csv")
    test = pd.read_csv(ROOT + "data/intermediate/test_data_complete.csv")
    
    train['split'] = 'Train'
    test['split'] = 'Test'
    combined = pd.concat([train, test], ignore_index=True)

    columns = [
        'admitted_from_source_value', 
        'care_site_name', 
        'gender',
        'admission_origin_or_pacu',
        'admission_origin_transfer',
        'admission_origin_floor',
        'admission_origin_ctc',
        'in_icu_death',
        'height',
        'weight',
        'albumin',
        'base_excess',
        'bicarbonate',
        'bilirubin',
        'chloride',
        'creatinine',
        'crp',
        'diastolic_pressure',
        'fio2',
        'glucose',
        'heart_rate',
        'hematocrit',
        'hemoglobin',
        'ionized_calcium',
        'lactate',
        'map',
        'paco2',
        'pao2',
        'ph',
        'potassium',
        'respiratory_rate',
        'sodium',
        'systolic_pressure',
        'temperature',
        'thrombocytes',
        'troponin',
        'urea',
        'urine_output',
        'ventilated',
        'wbc',
        'aado2'
    ]
    
    categorical = [
        'admitted_from_source_value', 
        'care_site_name', 
        'gender',
        'admission_origin_or_pacu',
        'admission_origin_transfer',
        'admission_origin_floor',
        'admission_origin_ctc',
        'in_icu_death',
        'ventilated'
    ]

    groupby = 'split'
    # nonnormal = ['bmi']  # Renders as Median [IQR]

    mytable = TableOne(combined, columns=columns, categorical=categorical, 
                       groupby=groupby, pval=True)
    
    print(mytable.tabulate(tablefmt="github"))
    mytable.to_csv(ROOT + "data/final/tables/table_1.csv")
    
    return mytable


def get_metrics(X_test, y_test, eval_label):

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
            X_test_tensor = torch.tensor(X_test.to_numpy(), dtype=torch.float32)
            y_pred_proba = model.predict_proba(X_test_tensor)[:,1]
            y_pred = model.predict(X_test_tensor)
        
        else:
            y_pred_proba = model.predict_proba(X_test)[:,1]
            y_pred = model.predict(X_test)

        results = evaluate_subset(y_test, y_pred_proba, y_pred)

        dict_of_results[model_algorithms[model_algorithm]['label']] = results

    # evaluate APACHE IV
    dict_of_results['apacheiv'] = evaluate_apacheiv(eval_label)
    
    # evaluate recalibrated models
    dict_of_results['apacheiv_recal'] = evaluate_apacheiv_recalibrated(eval_label)
    dict_of_results['xgb_recal'] = evaluate_xgb_recalibrated(eval_label)

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

    with open(ROOT + f'data/final/{eval_label}_results.yaml', 'w') as f:
        yaml.safe_dump(dict_of_results_serializable, f)

    
    """ 
    list documentation:

    list_of_models = [
        [filename_of_model[0], model_label[1], model_load_from_pickle[2], fpr[3], tpr[4], thresholds[5], auc_model[6], precision[7], recall[8], f1_model[9]]
    ]
    """

    return


def evaluate_apacheiv(eval_label):

    test = pd.read_csv(ROOT + "data/intermediate/test_data_complete.csv")

    # filter to the correct population based on eval_label
    if eval_label == "vs_ap4_cabg":
        test = test.loc[test['nice_ap4_model'] == "CABG"]
    elif eval_label == "vs_ap4_noncabg":
        test = test.loc[test['nice_ap4_model'] == "NON CABG"]

    # else: eval_label == "all_populations" -> use full test set

    y_test = test['in_icu_death']
    y_pred_proba = test['nice_ap4_prob']

    results = evaluate_subset(y_test, y_pred_proba)

    # recalibrate
    apache_recalibrated = LogisticRegression(penalty='none', max_iter=10000)
    # apache_recalibrated.fit()

    return results


def evaluate_apacheiv_recalibrated(eval_label):

    test = pd.read_csv(ROOT + "data/intermediate/test_data_complete.csv")

    # filter to the correct population based on eval_label
    if eval_label == "vs_ap4_cabg":
        test = test.loc[test['nice_ap4_model'] == "CABG"]
    elif eval_label == "vs_ap4_noncabg":
        test = test.loc[test['nice_ap4_model'] == "NON CABG"]

    y = test['in_icu_death']
    y_pred_proba = test['nice_ap4_prob']

    # --- split into calibration-train and calibration-eval ---
    logit = np.log(y_pred_proba / (1 - y_pred_proba)).values.reshape(-1, 1)

    logit_cal, logit_eval, y_cal, y_eval = train_test_split(
        logit, y, test_size=0.3, random_state=random_state
    )

    # --- fit recalibration model on calibration split only ---
    apache_recalibrated = LogisticRegression(penalty=None, max_iter=10000)
    apache_recalibrated.fit(logit_cal, y_cal)

    # --- evaluate on held-out eval split ---
    y_pred_proba_recal = apache_recalibrated.predict_proba(logit_eval)[:, 1]
    results = evaluate_subset(y_eval, y_pred_proba_recal)

    return results


def evaluate_xgb_recalibrated(eval_label):

    # load X_test and y_test
    X_test = pd.read_csv(ROOT + 'data/intermediate/X_test_imputed.csv')
    y_test = pd.read_csv(ROOT + 'data/intermediate/y_test.csv')['in_icu_death']
    test = pd.read_csv(ROOT + "data/intermediate/test_data_complete.csv")

    # filter to the correct population based on eval_label
    if eval_label == "vs_ap4_cabg":
        test = test.loc[test['nice_ap4_model'] == "CABG"]
    elif eval_label == "vs_ap4_noncabg":
        test = test.loc[test['nice_ap4_model'] == "NON CABG"]

    # load model
    filename = config['model_algorithms']['xgb_unimputed']['filename']
    with open(ROOT + f'data/final/models/{filename}', 'rb') as file:
        model = pickle.load(file)

    # --- get raw XGB predicted probabilities ---
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # --- split into calibration-train and calibration-eval ---
    logit = np.log(y_pred_proba / (1 - y_pred_proba)).reshape(-1, 1)

    logit_cal, logit_eval, y_cal, y_eval = train_test_split(
        logit, y_test, test_size=0.3, random_state=random_state
    )

    # --- fit recalibration model on calibration split only ---
    xgb_recalibrated = LogisticRegression(penalty=None, max_iter=10000)
    xgb_recalibrated.fit(logit_cal, y_cal)

    # --- evaluate on held-out eval split ---
    y_pred_proba_recal = xgb_recalibrated.predict_proba(logit_eval)[:, 1]
    results = evaluate_subset(y_eval, y_pred_proba_recal)
    
    return results


def evaluate_subset(y_test, y_pred_proba, y_pred=None):
    """Compute evaluation metrics for a single subset.
    
    y_pred is optional: if not provided, it is derived from y_pred_proba
    using a 0.5 threshold (used for APACHE IV where no model.predict() is available).
    """
    if y_pred is None:
        y_pred = (y_pred_proba >= 0.5).astype(int)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    auc_model = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    f1_model = f1_score(y_test, y_pred)

    prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)

    brier_score_metric = brier_score_loss(y_test, y_pred_proba)

    # ECE and MCE computed from raw predictions to match calibration_curve's 10 bins,
    # but using bin counts for proper weighting (calibration_curve doesn't expose these).
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            # Last bin is closed on both sides: [0.9, 1.0]
            mask = (y_pred_proba >= bin_boundaries[i]) & (y_pred_proba <= bin_boundaries[i + 1])
        else:
            mask = (y_pred_proba >= bin_boundaries[i]) & (y_pred_proba < bin_boundaries[i + 1])
        
        bin_count = mask.sum()
        if bin_count == 0:
            continue
        
        bin_accuracy = y_test[mask].mean()
        bin_confidence = y_pred_proba[mask].mean()
        bin_error = abs(bin_accuracy - bin_confidence)
        
        ece += (bin_count / len(y_test)) * bin_error
        mce = max(mce, bin_error)


    return {
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'auc_model': auc_model,
        'precision': precision,
        'recall': recall,
        'f1_model': f1_model,
        'prob_true': prob_true,
        'prob_pred': prob_pred,
        'brier_score_metric': brier_score_metric,
        'ece': ece,
        'mce': mce
    }


if __name__=="__main__":
    main()