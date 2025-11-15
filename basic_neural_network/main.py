import torch
from torch import nn
from torch import optim
import numpy as np

from basic_neural_network.neural_nets import RegressionNet, RegressionLoss
from basic_neural_network.generate_data import generate_data

n = 1000
n_covariates = 4
average_treatment_effect = 3
n_epochs = 1000

covariates, treatments, outcomes = generate_data(
    n, n_covariates, average_treatment_effect
)
X = torch.cat([covariates, treatments], dim=1)
regression_net = RegressionNet(n_covariates)
regression_criterion = RegressionLoss()  # squared error loss
regression_optimizer = optim.Adam(regression_net.parameters(), lr=1e-3)

for epoch in range(n_epochs):
    regression_optimizer.zero_grad()
    predicted_outcomes = regression_net(X)
    regression_loss = regression_criterion(predicted_outcomes, outcomes)
    regression_loss.backward()
    regression_optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss = {regression_loss.item():.4f}")

regression_net.eval()
no_treatment_counterfactuals = torch.cat([covariates, torch.zeros_like(treatments)], dim=1)
treatment_counterfactuals = torch.cat([covariates, torch.ones_like(treatments)], dim=1)
with torch.no_grad():
    Y0 = regression_net(no_treatment_counterfactuals).numpy()
    Y1 = regression_net(treatment_counterfactuals).numpy()
    print(np.mean(Y1 - Y0))
