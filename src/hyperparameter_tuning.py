from utils.helpers import ROOT, config
import pandas as pd
import yaml
import optuna
import os
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score
from sklearn import svm
import xgboost as xgb
from utils.torchneuralnet import PyTorchNN
from skorch import NeuralNetBinaryClassifier
import torch

def main():

    # set config
    model_algorithms = config['model_algorithms']
    n_trials = config['optuna_trials']
    random_state = config['random_state']
    n_jobs = config['optuna_jobs'] # Use all cores by default, or specify in config

    # Create directories for persistent storage
    os.makedirs(ROOT + 'data/final/optuna_studies', exist_ok=True)
    os.makedirs(ROOT + 'data/final/optuna_best_parameters', exist_ok=True)

    # load train set
    if config['oversample'] is True:
        X_train = pd.read_csv(ROOT + 'data/intermediate/X_train_smote.csv')
        y_train = pd.read_csv(ROOT + 'data/intermediate/y_train_smote.csv')['in_icu_death']
    else:
        X_train = pd.read_csv(ROOT + 'data/intermediate/X_train_imputed.csv')
        y_train = pd.read_csv(ROOT + 'data/intermediate/y_train.csv')['in_icu_death']

    # load validation set if to be used
    if config['use_validation_set'] is True:
        X_validate = pd.read_csv(ROOT + 'data/intermediate/X_validate.csv')
        y_validate = pd.read_csv(ROOT + 'data/intermediate/y_validate.csv')

        

    # iterate over all model algorithms
    for model_key in model_algorithms:

        # create subfunction to work with optuna
        def objective(trial):

            # # loop over all model algorithms
            # for model in config:
            # set hyperparameter tuning code for xgboost
            if model_key == 'xgb':

                # make shortcut 
                para_sc = config['model_algorithms'][model_key]['hyperparameter_range']

                # set parameters # TODO add all hyperparameters
                params = {
                    para_sc['max_depth']['label']: trial.suggest_int(para_sc['max_depth']['label'], para_sc['max_depth']['min'],para_sc['max_depth']['max']), 
                    para_sc['min_child_weight']['label']: trial.suggest_int(para_sc['min_child_weight']['label'], para_sc['min_child_weight']['min'], para_sc['min_child_weight']['max']),
                    para_sc['subsample']['label']: trial.suggest_float(para_sc['subsample']['label'], para_sc['subsample']['min'], para_sc['subsample']['max']),
                    para_sc['colsample_bytree']['label']: trial.suggest_float(para_sc['colsample_bytree']['label'], para_sc['colsample_bytree']['min'], para_sc['colsample_bytree']['max']),
                    para_sc['reg_lambda']['label']: trial.suggest_float(para_sc['reg_lambda']['label'], para_sc['reg_lambda']['min'], para_sc['reg_lambda']['max'], log=para_sc['reg_lambda']['log']),
                    para_sc['learning_rate']['label']: trial.suggest_float(para_sc['learning_rate']['label'], para_sc['learning_rate']['min'], para_sc['learning_rate']['max'])
                }

                # initiate model
                model = xgb.XGBClassifier(**params, objective="binary:logistic", enable_categorical=True, random_state=random_state, tree_method='hist')

                # TODO make XGBoost work with sklearn's cross_val_score API
                # # apply 5-fold crossvalidation -> does not work with sklearn API
                # scores = cross_val_score(model, X_train_imputed, y_train, cv=5, scoring='f1')

                dtrain = xgb.DMatrix(X_train, label=y_train)
                # dvalidate = xgb.DMatric(X_validate, label=y_validate)

                if config['use_cross_validation'] is True:
                    scores = xgb.cv(params, dtrain, nfold=5, stratified=True, early_stopping_rounds=40, num_boost_round=10000, seed=random_state, metrics='auc')
                    return scores.values[-1][2]
                    
                elif config['use_validation_set'] is True:
                    model.fit(X_train, y_train)
                    y_pred = model.predict_proba(X_validate)[:,1]
                    return roc_auc_score(y_validate, y_pred)

                else:
                    return 'error, no method specified to tune hyperparameters'

                # scores = xgb.cv(params, dtrain, nfold=5, stratified=True, early_stopping_rounds=40, num_boost_round=10000, seed=random_state, metrics='auc')
                
                # return  scores.values[-1][2] # order or results: train-auc-mean, train-auc-std, test-auc-mean, test-auc-std -> get test-auc-mean[2] from latest iteration[-1]
            
            if model_key == 'logreg':

                # make shortcut
                para_sc = config['model_algorithms'][model_key]['hyperparameter_range']
                # para_sc['penalty'].append(None) # add as hyperparameter

                print("X_train shape:", X_train.shape)
                print("y_train shape:", y_train.shape)

                # set parameters
                params = {
                    'penalty': trial.suggest_categorical('penalty', para_sc['penalty']),
                    'solver': trial.suggest_categorical('solver', para_sc['solver']),
                    'C': trial.suggest_float('C', para_sc['C']['min'], para_sc['C']['max'], log=para_sc['C']['log']),
                    'max_iter': trial.suggest_int('max_iter', para_sc['max_iter']['min'], para_sc['max_iter']['max'])
                }

                # initiate model
                model = LogisticRegression(**params, random_state=random_state, n_jobs=-1)

                if config['use_cross_validation'] is True:
                    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
                    return scores.mean()
                
                elif config['use_validation_set'] is True:
                    model.fit(X_train, y_train)
                    y_pred = model.predict_proba(X_validate)[:,1]
                    return roc_auc_score(y_validate, y_pred)

                else:
                    return 'error, no method specified to tune hyperparameters'

            
            if model_key == 'svm':

                # make shortcut
                para_sc = config['model_algorithms'][model_key]['hyperparameter_range']

                # set parameters to search by optuna
                params = {
                    'C': trial.suggest_float('C', para_sc['C']['min'], para_sc['C']['max'], log=para_sc['C']['log']),
                    'gamma': trial.suggest_float('gamma', para_sc['gamma']['min'], para_sc['gamma']['max'], log=para_sc['gamma']['log']),
                    'kernel': trial.suggest_categorical('kernel', para_sc['kernel'])
                }

                # initiate model
                model = svm.SVC(**params, random_state=random_state, probability=True)
            
                if config['use_cross_validation'] is True:
                    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
                    return scores.mean()
                    
                elif config['use_validation_set'] is True:
                    y_pred = model.predict_proba(X_validate)[:,1]
                    return roc_auc_score(y_validate, y_pred)

                else:
                    return 'error, no method specified to tune hyperparameters'
            
            if model_key == 'knn':
                
                # make shortcut
                para_sc = config['model_algorithms'][model_key]['hyperparameter_range']

                # set parameters to search by optuna
                params = {
                    'n_neighbors': trial.suggest_int('n_neighbors', para_sc['n_neighbors']['min'], para_sc['n_neighbors']['max']),
                    'weights': trial.suggest_categorical('weights', para_sc['weights']),
                    'metric': trial.suggest_categorical('metric', para_sc['metric'])
                }

                # initiate model
                model = KNeighborsClassifier(**params, n_jobs=-1)

                if config['use_cross_validation'] is True:
                    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
                    return scores.mean()
                    
                elif config['use_validation_set'] is True:
                    model.fit(X_train, y_train)
                    y_pred = model.predict_proba(X_validate)[:,1]
                    return roc_auc_score(y_validate, y_pred)

                else:
                    return 'error, no method specified to tune hyperparameters'
            
            if model_key == 'naivebayes':

                # make shortcut
                para_sc = config['model_algorithms'][model_key]['hyperparameter_range']

                # set parameters to search by optuna
                params = {
                    'var_smoothing': trial.suggest_loguniform('var_smoothing', para_sc['var_smoothing']['min'], para_sc['var_smoothing']['max'])
                }

                # initiate model
                model = GaussianNB(**params)

                if config['use_cross_validation'] is True:
                    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
                    return scores.mean()
                    
                elif config['use_validation_set'] is True:
                    model.fit(X_train, y_train)
                    y_pred = model.predict_proba(X_validate)[:,1]
                    return roc_auc_score(y_validate, y_pred)

                else:
                    return 'error, no method specified to tune hyperparameters'
            
            if model_key == 'nn':
                
                # make shortcut
                para_sc = config['model_algorithms'][model_key]['hyperparameter_range']

                # set parameters to search by optuna
                params = {
                    'hidden_layers': [trial.suggest_int(f'layer_{i}', para_sc['hidden_layers']['min'], para_sc['hidden_layers']['max']) for i in range(trial.suggest_int('num_layers', para_sc['num_layers']['min'], para_sc['num_layers']['max']))],
                    'lr': trial.suggest_float('lr', para_sc['lr']['min'], para_sc['lr']['max'], log=True),
                    'max_epochs': trial.suggest_int('epochs', para_sc['epochs']['min'], para_sc['epochs']['max']),
                    'batch_size': trial.suggest_categorical('batch_size', para_sc['batch_size'])
                }

                # params['hidden_layers'] = tuple(params['hidden_layers']) # make tuple from hidden layers
                # explicitly define hidden_layers and drop it from dictionary
                hidden_layers = params['hidden_layers']
                params.pop('hidden_layers')

                # model = TorchNeuralNet(**params, random_state=random_state) # set breakpoint to find out why the model is identified as a regressor instead of a classifier
                # create skorch wrapper
                model = NeuralNetBinaryClassifier(
                    PyTorchNN(input_dim=X_train.shape[1], hidden_layers=hidden_layers),
                    criterion=torch.nn.BCEWithLogitsLoss,
                    optimizer=torch.optim.Adam,
                    **params
                )

                X_train_tensor = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
                y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32)

                if config['use_cross_validation'] is True:
                    scores = cross_val_score(model, X_train_tensor, y_train_tensor, cv=5, scoring='roc_auc')
                    return scores.mean()
                    
                elif config['use_validation_set'] is True:

                    X_validate_tensor = torch.tensor(X_validate.to_numpy(), dtype=torch.float32)
                    y_validate_tensor = torch.tensor(y_validate.to_numpy(), dtype=torch.float32)

                    model.fit(X_train, y_train)
                    y_pred = model.predict_proba(X_validate_tensor)[:,1]
                    return roc_auc_score(y_validate, y_pred)

                else:
                    return 'error, no method specified to tune hyperparameters'
                               
        
        # Create optuna study with persistent storage and parallel execution
        sampler = optuna.samplers.TPESampler()
        
        # Create unique study name and storage path
        study_name = f'{model_key}_optimization'
        storage_name = f'sqlite:///{ROOT}data/final/optuna_studies/{model_key}_study.db'
        
        # Load existing study or create new one
        optuna_load_study = config['optuna_load_study']

        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            direction='maximize', 
            sampler=sampler,
            load_if_exists=optuna_load_study  # Resume from existing study if available
        )
        
        # Check if study has existing trials
        n_completed_trials = len(study.trials)
        n_remaining_trials = max(0, n_trials - n_completed_trials)
        
        print(f"\n{'='*60}")
        print(f"Model: {model_key}")
        print(f"Completed trials: {n_completed_trials}/{n_trials}")
        print(f"Remaining trials: {n_remaining_trials}")
        print(f"Using {n_jobs} parallel jobs")
        print(f"{'='*60}\n")

        # Only optimize if there are remaining trials
        if n_remaining_trials > 0:
            # Run optimization with parallel execution
            study.optimize(
                objective, 
                n_trials=n_remaining_trials, 
                n_jobs=n_jobs,  # Parallel trials
                show_progress_bar=True
            )
        else:
            print(f"Study for {model_key} already completed {n_trials} trials. Skipping optimization.")

        # Export best parameters
        best_params = study.best_params
        with open(ROOT + f'data/final/optuna_best_parameters/{model_key}_params.yaml', 'w') as f:
            yaml.dump(best_params, f, default_flow_style=False)

        # Export all trials to CSV for analysis
        study_df = study.trials_dataframe()
        study_df.to_csv(ROOT + f'data/final/optuna_studies/{model_key}_all_trials.csv', index=False)

        # Print best hyperparameters and score
        print(f"\nBest hyperparameters for {model_key}: {study.best_params}")
        print(f"Best score: {study.best_value:.4f}")
        print(f"Total completed trials: {len(study.trials)}\n")


if __name__=="__main__":
    main()