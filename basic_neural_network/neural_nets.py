import torch
import torch.nn as nn


class MLPNet(nn.Module):
    def __init__(self, n_covariates):
        super(MLPNet, self).__init__()
        self.first_layer = nn.Linear(n_covariates + 1, 32)
        self.first_activation_layer = nn.ReLU()
        self.second_layer = nn.Linear(32, 32)
        self.second_activation_layer = nn.ReLU()
        self.output_layer = nn.Linear(32, 1)

    def forward(self, covariates, treatments):
        _input = torch.cat([covariates, treatments], dim=1)
        hidden_state = self.first_layer(_input)
        hidden_state = self.first_activation_layer(hidden_state)
        hidden_state = self.second_layer(hidden_state)
        hidden_state = self.second_activation_layer(hidden_state)
        output = self.output_layer(hidden_state)
        return output


class RegressionLoss(nn.MSELoss):
    pass


class RieszLoss(nn.Module):

    def __init__(self, lambda_l2=0.01):
        super(RieszLoss, self).__init__()
        self.lambda_l2 = lambda_l2

    def forward(
        self, actual_predictions, full_treatment_predictions, no_treatment_predictions
    ):
        square_term = actual_predictions**2
        plugin_term = full_treatment_predictions - no_treatment_predictions
        return torch.mean(square_term - 2 * plugin_term)
