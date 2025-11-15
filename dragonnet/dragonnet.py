import torch
import torch.nn as nn


class MLPNet(nn.Module):
    def __init__(self, n_covariates):
        super(MLPNet, self).__init__()
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
