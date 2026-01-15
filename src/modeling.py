from utils.helpers import ROOT, config
from utils.torchneuralnet import PyTorchNN
import pandas as pd
import numpy as np
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
from skorch.callbacks import Callback
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
        # ('SVM', train_svm_model, {'X_train': X_train, 'y_train': y_train, 'random_state': random_state}), -> does not converge
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
    max_epochs = best_params.pop('epochs')
    best_params['max_epochs'] = max_epochs
    
    # Calculate optimal batch size for your hardware
    n_samples = len(X_train)
    n_features = X_train.shape[1]
    
    # Estimate memory per sample (features + hidden layers + gradients)
    total_neurons = n_features + sum(hidden_layers) + 1
    bytes_per_sample = total_neurons * 4 * 3  # float32 * (forward + backward + optimizer)
    
    # Use 80% of available RAM for batch, leave 20% for OS/other processes
    available_ram_gb = 128 * 0.8
    available_ram_bytes = available_ram_gb * 1024**3
    
    # Calculate theoretical max batch size
    max_batch_size = int(available_ram_bytes / bytes_per_sample)
    
    # Practical constraints for optimal training
    # 1. Use power of 2 for better hardware utilization
    # 2. Cap at reasonable values for training dynamics
    # 3. Ensure at least 10 batches per epoch for proper gradient updates
    min_batches_per_epoch = 10
    max_practical_batch = min(
        n_samples // min_batches_per_epoch,  # At least 10 batches
        16384,  # Cap at 16K for training stability
        max_batch_size
    )
    
    # Round down to nearest power of 2
    optimal_batch_size = 2 ** int(np.log2(max_practical_batch))
    
    # Override batch_size if not already optimal
    if 'batch_size' not in best_params or best_params['batch_size'] < optimal_batch_size:
        best_params['batch_size'] = optimal_batch_size
        logger.info(f"Optimized batch_size for 128GB RAM: {optimal_batch_size}")
    
    # Set PyTorch to use multiple threads
    torch.set_num_threads(32)
    
    logger.info(f"NN Architecture: {X_train.shape[1]} → {' → '.join(map(str, hidden_layers))} → 1")
    logger.info(f"Training for {max_epochs} epochs with batch_size={best_params.get('batch_size', 128)}")
    
    # Prepare data once
    X_train_tensor = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
    
    # Create model with detailed callbacks
    callbacks = [
        ('progress', DetailedProgressCallback()),
    ]
    
    nn_model = NeuralNetBinaryClassifier(
        PyTorchNN(input_dim=X_train.shape[1], hidden_layers=hidden_layers),
        criterion=torch.nn.BCEWithLogitsLoss,
        optimizer=torch.optim.Adam,
        callbacks=callbacks,
        verbose=0,  # Suppress default skorch output
        **best_params
    )
    
    logger.info("Starting neural network training...")
    nn_model.fit(X_train_tensor, y_train_tensor)
    
    # Save using torch.save instead of pickle to avoid callback serialization issues
    filename = ROOT + 'data/final/models/nn_model.pkl'
    
    # Save the complete model state including the wrapper
    save_dict = {
        'module_state_dict': nn_model.module_.state_dict(),
        'input_dim': X_train.shape[1],
        'hidden_layers': hidden_layers,
        'best_params': best_params,
        'criterion': torch.nn.BCEWithLogitsLoss,
        'optimizer': torch.optim.Adam
    }
    
    torch.save(save_dict, filename)
    logger.info(f"Model saved to {filename}")
    
    # Also save a version compatible with direct loading (optional)
    # This creates a new model without callbacks that can be pickled
    nn_model_clean = NeuralNetBinaryClassifier(
        PyTorchNN(input_dim=X_train.shape[1], hidden_layers=hidden_layers),
        criterion=torch.nn.BCEWithLogitsLoss,
        optimizer=torch.optim.Adam,
        verbose=0,
        **best_params
    )
    nn_model_clean.initialize()
    nn_model_clean.module_.load_state_dict(nn_model.module_.state_dict())
    
    # Save the clean version with pickle
    filename_clean = ROOT + 'data/final/models/nn_model_clean.pkl'
    with open(filename_clean, 'wb') as file:
        pickle.dump(nn_model_clean, file)
    logger.info(f"Clean model (no callbacks) saved to {filename_clean}")
    
    return nn_model


def train_apiv_recal_model(X_train, y_train):
    logger.info("APACHE IV recalibration training starts now.")
    
    # TODO: Implement APACHE IV recalibration
    # logodds calculation and logistic regression training
    
    return None

### --- helper class & functions --- ###

# Create custom callback for detailed progress
class DetailedProgressCallback(Callback):
    def __init__(self):
        self.pbar = None
        self.best_loss = float('inf')
        
    def initialize(self):
        self.best_loss = float('inf')
        return self
        
    def on_train_begin(self, net, X=None, y=None, **kwargs):
        self.pbar = tqdm(total=net.max_epochs, desc='NN Epochs', unit='epoch')
        
    def on_epoch_end(self, net, dataset_train=None, dataset_valid=None, **kwargs):
        # Get current epoch info
        current_epoch = len(net.history)
        train_loss = net.history[-1]['train_loss']
        valid_loss = net.history[-1].get('valid_loss', None)
        
        # Calculate improvement
        improvement = ""
        if train_loss < self.best_loss:
            improvement = f"↓ (best: {self.best_loss:.4f}→{train_loss:.4f})"
            self.best_loss = train_loss
        
        # Update progress bar with detailed info
        postfix = {
            'train_loss': f'{train_loss:.4f}',
        }
        if valid_loss is not None:
            postfix['valid_loss'] = f'{valid_loss:.4f}'
        
        self.pbar.set_postfix(postfix)
        self.pbar.update(1)
        
        # Log every 10 epochs or on last epoch
        if current_epoch % 10 == 0 or current_epoch == net.max_epochs:
            logger.info(f"Epoch {current_epoch}/{net.max_epochs} - Loss: {train_loss:.4f} {improvement}")
    
    def on_train_end(self, net, X=None, y=None, **kwargs):
        if self.pbar:
            self.pbar.close()
        logger.info(f"NN Training completed - Final loss: {self.best_loss:.4f}")

if __name__ == "__main__":
    main()