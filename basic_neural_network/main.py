import torch
from torch import optim
import numpy as np

from basic_neural_network.neural_nets import (
    RegressionNet,
    RegressionLoss,
    RieszNet,
    RieszLoss,
)
from basic_neural_network.generate_data import generate_data

n = 1000
n_covariates = 100
average_treatment_effect = 3
n_epochs = 100
l2_lambda = 0.1

covariates, treatments, outcomes = generate_data(
    n, n_covariates, average_treatment_effect
)
X = torch.cat([covariates, treatments], dim=1)
X_no_treatment = torch.cat([covariates, torch.zeros_like(treatments)], dim=1)
X_full_treatment = torch.cat([covariates, torch.ones_like(treatments)], dim=1)

regression_net = RegressionNet(n_covariates)
regression_criterion = RegressionLoss()  # squared error loss
regression_optimizer = optim.Adam(regression_net.parameters(), lr=1e-3, weight_decay=l2_lambda)

riesz_net = RieszNet(n_covariates)
riesz_criterion = RieszLoss()
riesz_optimizer = optim.Adam(riesz_net.parameters(), lr=1e-3, weight_decay=l2_lambda)


for epoch in range(n_epochs):
    regression_optimizer.zero_grad()
    predicted_outcomes = regression_net(X)
    regression_loss = regression_criterion(predicted_outcomes, outcomes)
    regression_loss.backward()
    regression_optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss = {regression_loss.item():.4f}")

    riesz_optimizer.zero_grad()
    X_combined = torch.cat([X, X_no_treatment, X_full_treatment], dim=0)
    all_outputs = riesz_net(X_combined)
    actual_riesz, no_treatment_riesz, full_treatment_riesz = torch.split(
        all_outputs, n, dim=0
    )
    riesz_loss = riesz_criterion(actual_riesz, no_treatment_riesz, full_treatment_riesz)
    riesz_loss.backward()
    riesz_optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss = {riesz_loss.item():.4f}")

regression_net.eval()
riesz_net.eval()
with torch.no_grad():
    Y0 = regression_net(X_no_treatment).numpy()
    Y1 = regression_net(X_full_treatment).numpy()
    residual = outcomes.numpy() - regression_net(X).numpy()
    riesz = riesz_net(X).numpy()

    print(np.mean(Y1 - Y0))
    print(np.mean(Y1 - Y0 + residual * riesz))
