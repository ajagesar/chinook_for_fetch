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

    # # Load best hyperparameters
    # with open(ROOT + 'data/final/optuna_best_parameters/xgb_unimputed_params.yaml', 'r') as f:
    #     best_params = yaml.safe_load(f)
    
    # # Use all available cores for XGBoost
    # best_params['n_jobs'] = -1
    # best_params['tree_method'] = 'hist'  # Faster histogram-based algorithm
    
    # Fit model
    xgb_model = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
              gamma=0, importance_type=None,
              interaction_constraints='', learning_rate=0.1, max_delta_step=0,
              max_depth=2, min_child_weight=15,
              monotone_constraints='()', n_estimators=100, n_jobs=16,
              num_parallel_tree=1, predictor='auto', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)
    
    xgb_model.fit(X_train, y_train)
    
    # Save model
    filename = ROOT + 'data/final/models/xgb_unimputed.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(xgb_model, file)
    
    return xgb_model


if __name__=="__main__":
    main()