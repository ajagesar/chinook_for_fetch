import pandas as pd
import numpy as np
from utils.helpers import ROOT, config
import pickle
import yaml

import xgboost as xgb

def main():



    # train CHINOOK on each separate dataset

    # get overdispersion

    # create funnel plot

    return



def train_xgb_unimputed(X_train, y_train, random_state=42):

    # Load best hyperparameters
    with open(ROOT + 'data/final/optuna_best_parameters/xgb_unimputed_params.yaml', 'r') as f:
        best_params = yaml.safe_load(f)
    
    # # Use all available cores for XGBoost
    # best_params['n_jobs'] = -1
    # best_params['tree_method'] = 'hist'  # Faster histogram-based algorithm
    
    # Fit model
    xgb_model = xgb.XGBClassifier(  
        **best_params, 
        objective="binary:logistic", 
        enable_categorical=True, 
        random_state=random_state,
        feature_names=list(X_train.columns)
    )
    
    xgb_model.fit(X_train, y_train)
    
    # Save model
    filename = ROOT + 'data/final/models/xgb_unimputed.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(xgb_model, file)
    
    return xgb_model


if __name__=="__main__":
    main()