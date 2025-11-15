import torch
import torch.nn as nn


class RegressionNet(nn.Module):
    def __init__(self, n_covariates):
        super(RegressionNet, self).__init__()
        self.first_layer = nn.Linear(n_covariates + 1, 64)
        self.first_activation_layer = nn.ReLU()
        self.second_layer = nn.Linear(64, 64)
        self.second_activation_layer = nn.ReLU()
        self.output_layer = nn.Linear(64, 1)

    def forward(self, _input):
        hidden_state = self.first_layer(_input)
        hidden_state = self.first_activation_layer(hidden_state)
        hidden_state = self.second_layer(hidden_state)
        hidden_state = self.second_activation_layer(hidden_state)
        output = self.output_layer(hidden_state)
        return output


class RegressionLoss(nn.MSELoss):
    pass


class RieszNet(nn.Module):
    def __init__(self, n_covariates):
        super(RieszNet, self).__init__()
        self.first_layer = nn.Linear(n_covariates + 1, 64)
        self.first_activation_layer = nn.ReLU()
        self.second_layer = nn.Linear(64, 64)
        self.second_activation_layer = nn.ReLU()
        self.output_layer = nn.Linear(64, 1)
        self.epsilon = nn.Parameter(torch.zeros(1))

    def forward(self, _input):
        hidden_state = self.first_layer(_input)
        hidden_state = self.first_activation_layer(hidden_state)
        hidden_state = self.second_layer(hidden_state)
        hidden_state = self.second_activation_layer(hidden_state)
        output = self.output_layer(hidden_state)
        return output


class RieszLoss(nn.Module):
    @staticmethod
    def forward(
        actual_predictions, full_treatment_predictions, no_treament_predictions
    ):
        square_term = actual_predictions**2
        plugin_term = full_treatment_predictions - no_treament_predictions
        return torch.mean(square_term + plugin_term)
