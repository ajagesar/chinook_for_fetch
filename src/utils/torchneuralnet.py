import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from skorch import NeuralNetBinaryClassifier

class PyTorchNN(nn.Module):
    def __init__(self, input_dim, hidden_layers):
        super().__init__()
        layers = []

        prev_dim = input_dim

        for hidden_units in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_units))
            layers.append(nn.ReLU())
            prev_dim = hidden_units

        layers.append(nn.Linear(prev_dim, 1)) # use output dimension of 1 because of binary classification, which gives a logit that can be converted to probability with sigmoid function

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

