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
n_epochs = 1000
l2_lambda = 0.3

covariates, treatments, outcomes = generate_data(
    n, n_covariates, average_treatment_effect
)
regression_net = RegressionNet(n_covariates)
regression_criterion = RegressionLoss()  # squared error loss
regression_optimizer = optim.Adam(
    regression_net.parameters(), lr=1e-3, weight_decay=l2_lambda
)

riesz_net = RieszNet(n_covariates)
riesz_criterion = RieszLoss()
riesz_optimizer = optim.Adam(riesz_net.parameters(), lr=1e-3, weight_decay=l2_lambda)


for epoch in range(n_epochs):
    regression_optimizer.zero_grad()
    predicted_outcomes = regression_net(covariates, treatments)
    regression_loss = regression_criterion(predicted_outcomes, outcomes)
    regression_loss.backward()
    regression_optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss = {regression_loss.item():.4f}")

    riesz_optimizer.zero_grad()
    actual_riesz = riesz_net(covariates, treatments)
    no_treatment_riesz = riesz_net(covariates, torch.zeros_like(treatments))
    full_treatment_riesz = riesz_net(covariates, torch.ones_like(treatments))
    riesz_loss = riesz_criterion(actual_riesz, full_treatment_riesz, no_treatment_riesz)
    riesz_loss.backward()
    riesz_optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss = {riesz_loss.item():.4f}")


regression_net.eval()
riesz_net.eval()
with torch.no_grad():
    Y0 = regression_net(covariates, torch.zeros_like(treatments)).numpy()
    Y1 = regression_net(covariates, torch.ones_like(treatments)).numpy()
    residual = outcomes.numpy() - regression_net(covariates, treatments).numpy()
    riesz = riesz_net(covariates, treatments).numpy()

    print(np.mean(Y1 - Y0))
    print(np.mean(Y1 - Y0 + residual * riesz))
