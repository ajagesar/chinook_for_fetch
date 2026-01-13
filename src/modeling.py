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


def main():

    # load train set
    if config['oversample'] == True:
        X_train = pd.read_csv(ROOT + 'data/intermediate/X_train_smote.csv')
        y_train = pd.read_csv(ROOT + 'data/intermediate/y_train_smote.csv')['in_icu_death']
    else:
        X_train = pd.read_csv(ROOT + 'data/intermediate/X_train_imputed.csv')
        y_train = pd.read_csv(ROOT + 'data/intermediate/y_train.csv')['in_icu_death']

    # TODO load APACHE IV data

    random_state = config['random_state']

    # xgb_model = xgb.XGBClassifier(objective="binary:logistic", enable_categorical=True, random_state=42, n_estimators=50, eval_metric=["auc"])
    # xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

    # train xgb_model
    xgb_model = train_xgb_model(X_train=X_train, y_train=y_train, random_state=42)

    # train lr model
    lr_model = train_lr_model(X_train=X_train, y_train=y_train, random_state=random_state)

    # train svm model
    svm_model = train_svm_model(X_train=X_train, y_train=y_train, random_state=random_state)

    # train knn model
    knn_model = train_knn_model(X_train=X_train, y_train=y_train)

    # train naive bayes model
    naivebayes_model = train_naivebayes_model(X_train=X_train, y_train=y_train)

    # train neural network
    nn_model = train_nn_model(X_train=X_train, y_train=y_train)

    # train recalibrated APACHE IV model
    apiv_recal_model = train_apiv_recal_model(X_train=X_train, y_train=y_train)

    return


def train_xgb_model(X_train, y_train, random_state=42):
    # load best hyperparameters
    with open(ROOT + 'data/final/optuna_best_parameters/xgb_params.yaml', 'r') as f:
        best_params = yaml.safe_load(f)

    # fit model
    xgb_model = xgb.XGBClassifier(**best_params, objective="binary:logistic", enable_categorical=True, random_state=random_state)
    xgb_model.fit(X_train, y_train)

    # save model
    filename = ROOT + 'data/final/models/xgb_model.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(xgb_model, file)

    return xgb_model


def train_lr_model(X_train, y_train, random_state):
    
    with open(ROOT + 'data/final/optuna_best_parameters/logreg_params.yaml', 'r') as f:
        best_params = yaml.safe_load(f)

    # fit model
    lr_model = LogisticRegression(**best_params, random_state=random_state)
    lr_model.fit(X_train, y_train)

    # save model
    filename = ROOT + 'data/final/models/logreg_model.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(lr_model, file)
    
    return lr_model


def train_svm_model(X_train, y_train, random_state):

    with open(ROOT + 'data/final/optuna_best_parameters/svm_params.yaml', 'r') as f:
        best_params = yaml.safe_load(f)

    # fit model
    svm_model = svm.SVC(**best_params, probability=True, random_state=random_state)
    svm_model.fit(X_train, y_train)

    # save model
    filename = ROOT + 'data/final/models/svm_model.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(svm_model, file)
    
    return svm_model


def train_knn_model(X_train, y_train):
    
    with open(ROOT + 'data/final/optuna_best_parameters/knn_params.yaml', 'r') as f:
        best_params = yaml.safe_load(f)

    # fit model
    knn_model = KNeighborsClassifier(**best_params)
    knn_model.fit(X_train, y_train)

    # save model
    filename = ROOT + 'data/final/models/knn_model.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(knn_model, file)

    return knn_model


def train_naivebayes_model(X_train, y_train):
    
    with open(ROOT + 'data/final/optuna_best_parameters/naivebayes_params.yaml', 'r') as f:
        best_params = yaml.safe_load(f)

    # fit model
    naivebayes_model = GaussianNB(**best_params)
    naivebayes_model.fit(X_train, y_train)

    # save model
    filename = ROOT + 'data/final/models/naivebayes_model.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(naivebayes_model, file)

    return naivebayes_model


def train_nn_model(X_train, y_train):
    
    with open(ROOT + 'data/final/optuna_best_parameters/nn_params.yaml', 'r') as f:
        best_params = yaml.safe_load(f)

    # explicitly define hidden_layers and drop it from dictionary
    num_layers = best_params['num_layers']
    hidden_layers = tuple(best_params[f'layer_{i}'] for i in range(num_layers))
    
    best_params = {k: v for k, v in best_params.items() if 'layer' not in k}

    # rename epochs to max_epochs
    best_params['max_epochs'] = best_params.pop('epochs')

    # fit model
    nn_model = NeuralNetBinaryClassifier(
                        PyTorchNN(input_dim=X_train.shape[1], hidden_layers=hidden_layers),
                        criterion=torch.nn.BCEWithLogitsLoss,
                        optimizer=torch.optim.Adam,
                        **best_params
                    )
    
    X_train_tensor = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32)

    nn_model.fit(X_train_tensor, y_train_tensor)

    # save model
    filename = ROOT + 'data/final/models/nn_model.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(nn_model, file)

    return nn_model


def train_apiv_recal_model(X_train, y_train):
    
    # calculate original APACHE IV logodds on train data set for CABG and non-CABG patients
    print("script has run until apache iv recalibrated training")
    print("Apache IV recalibration training starts now.")

    # logodds = calculate_hos_mort_non_cabg(X_train, result='logodds') -> does not work, columns removed for split. Will have to this calculation differently in the pipeline

    # train basic logistic regression model with logodds as independent variable

    
    return


if __name__=="__main__":
    main()