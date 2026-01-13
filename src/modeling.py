from utils.helpers import ROOT, config
from utils.torchneuralnet import PyTorchNN
import pandas as pd
import pickle
import yaml
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from skorch import NeuralNetBinaryClassifier
import torch
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from functools import partial

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting model training pipeline...")
    
    # Load train set with progress indicator
    logger.info("Loading training data...")
    if config['oversample']:
        X_train = pd.read_csv(ROOT + 'data/intermediate/X_train_smote.csv')
        y_train = pd.read_csv(ROOT + 'data/intermediate/y_train_smote.csv')['in_icu_death']
    else:
        X_train = pd.read_csv(ROOT + 'data/intermediate/X_train_imputed.csv')
        y_train = pd.read_csv(ROOT + 'data/intermediate/y_train.csv')['in_icu_death']
    
    logger.info(f"Data loaded: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    
    random_state = config['random_state']
    
    # Define training tasks
    training_tasks = [
        ('XGBoost', train_xgb_model, {'X_train': X_train, 'y_train': y_train, 'random_state': random_state}),
        ('Logistic Regression', train_lr_model, {'X_train': X_train, 'y_train': y_train, 'random_state': random_state}),
        ('SVM', train_svm_model, {'X_train': X_train, 'y_train': y_train, 'random_state': random_state}),
        ('KNN', train_knn_model, {'X_train': X_train, 'y_train': y_train}),
        ('Naive Bayes', train_naivebayes_model, {'X_train': X_train, 'y_train': y_train}),
    ]
    
    # Train models in parallel (excluding NN which needs special handling)
    logger.info("Training models in parallel...")
    models = {}
    
    with ProcessPoolExecutor(max_workers=min(5, 32)) as executor:
        # Submit all tasks
        future_to_model = {
            executor.submit(train_model_wrapper, name, func, kwargs): name 
            for name, func, kwargs in training_tasks
        }
        
        # Process completed tasks with progress bar
        with tqdm(total=len(training_tasks), desc="Training models") as pbar:
            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    models[model_name] = future.result()
                    logger.info(f"✓ {model_name} training completed")
                except Exception as e:
                    logger.error(f"✗ {model_name} training failed: {str(e)}")
                pbar.update(1)
    
    # Train neural network separately (PyTorch doesn't pickle well across processes)
    logger.info("Training Neural Network...")
    try:
        models['Neural Network'] = train_nn_model(X_train=X_train, y_train=y_train)
        logger.info("✓ Neural Network training completed")
    except Exception as e:
        logger.error(f"✗ Neural Network training failed: {str(e)}")
    
    # Train APACHE IV recalibrated model
    logger.info("Training APACHE IV recalibrated model...")
    try:
        models['APACHE IV Recalibrated'] = train_apiv_recal_model(X_train=X_train, y_train=y_train)
        logger.info("✓ APACHE IV Recalibrated training completed")
    except Exception as e:
        logger.error(f"✗ APACHE IV Recalibrated training failed: {str(e)}")
    
    logger.info("All model training completed!")
    return models


def train_model_wrapper(name, func, kwargs):
    """Wrapper function for parallel execution"""
    return func(**kwargs)


def train_xgb_model(X_train, y_train, random_state=42):
    # Load best hyperparameters
    with open(ROOT + 'data/final/optuna_best_parameters/xgb_params.yaml', 'r') as f:
        best_params = yaml.safe_load(f)
    
    # Use all available cores for XGBoost
    best_params['n_jobs'] = -1
    best_params['tree_method'] = 'hist'  # Faster histogram-based algorithm
    
    # Fit model
    xgb_model = xgb.XGBClassifier(
        **best_params, 
        objective="binary:logistic", 
        enable_categorical=True, 
        random_state=random_state
    )
    xgb_model.fit(X_train, y_train)
    
    # Save model
    filename = ROOT + 'data/final/models/xgb_model.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(xgb_model, file)
    
    return xgb_model


def train_lr_model(X_train, y_train, random_state):
    with open(ROOT + 'data/final/optuna_best_parameters/logreg_params.yaml', 'r') as f:
        best_params = yaml.safe_load(f)
    
    # Enable parallel computation
    lr_model = LogisticRegression(**best_params, random_state=random_state, n_jobs=-1)
    lr_model.fit(X_train, y_train)
    
    # Save model
    filename = ROOT + 'data/final/models/logreg_model.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(lr_model, file)
    
    return lr_model


def train_svm_model(X_train, y_train, random_state):
    with open(ROOT + 'data/final/optuna_best_parameters/svm_params.yaml', 'r') as f:
        best_params = yaml.safe_load(f)
    
    # SVM doesn't have n_jobs for fit, but setting cache_size can help
    best_params['cache_size'] = 2000  # Increase cache for your 128GB RAM
    
    svm_model = svm.SVC(**best_params, probability=True, random_state=random_state)
    svm_model.fit(X_train, y_train)
    
    # Save model
    filename = ROOT + 'data/final/models/svm_model.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(svm_model, file)
    
    return svm_model


def train_knn_model(X_train, y_train):
    with open(ROOT + 'data/final/optuna_best_parameters/knn_params.yaml', 'r') as f:
        best_params = yaml.safe_load(f)
    
    # Enable parallel computation
    knn_model = KNeighborsClassifier(**best_params, n_jobs=-1)
    knn_model.fit(X_train, y_train)
    
    # Save model
    filename = ROOT + 'data/final/models/knn_model.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(knn_model, file)
    
    return knn_model


def train_naivebayes_model(X_train, y_train):
    with open(ROOT + 'data/final/optuna_best_parameters/naivebayes_params.yaml', 'r') as f:
        best_params = yaml.safe_load(f)
    
    naivebayes_model = GaussianNB(**best_params)
    naivebayes_model.fit(X_train, y_train)
    
    # Save model
    filename = ROOT + 'data/final/models/naivebayes_model.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(naivebayes_model, file)
    
    return naivebayes_model


def train_nn_model(X_train, y_train):
    with open(ROOT + 'data/final/optuna_best_parameters/nn_params.yaml', 'r') as f:
        best_params = yaml.safe_load(f)
    
    # Extract hidden layers configuration
    num_layers = best_params['num_layers']
    hidden_layers = tuple(best_params[f'layer_{i}'] for i in range(num_layers))
    
    best_params = {k: v for k, v in best_params.items() if 'layer' not in k}
    best_params['max_epochs'] = best_params.pop('epochs')
    
    # Set PyTorch to use multiple threads
    torch.set_num_threads(32)
    
    # Prepare data once
    X_train_tensor = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
    
    # Create model with callbacks for progress
    callbacks = [
        ('progress', tqdm(desc='NN Training', total=best_params['max_epochs'], leave=False))
    ]
    
    nn_model = NeuralNetBinaryClassifier(
        PyTorchNN(input_dim=X_train.shape[1], hidden_layers=hidden_layers),
        criterion=torch.nn.BCEWithLogitsLoss,
        optimizer=torch.optim.Adam,
        callbacks=callbacks,
        **best_params
    )
    
    nn_model.fit(X_train_tensor, y_train_tensor)
    
    # Save model
    filename = ROOT + 'data/final/models/nn_model.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(nn_model, file)
    
    return nn_model


def train_apiv_recal_model(X_train, y_train):
    logger.info("APACHE IV recalibration training starts now.")
    
    # TODO: Implement APACHE IV recalibration
    # logodds calculation and logistic regression training
    
    return None


if __name__ == "__main__":
    main()