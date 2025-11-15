import torch
from torch import nn
from torch import optim
import numpy as np

from dragonnet.dragonnet import MLPNet
from dragonnet.generate_data import generate_data

n = 1000
n_covariates = 10
average_treatment_effect = 3
n_epochs = 1000

covariates, treatments, outcomes = generate_data(
    n, n_covariates, average_treatment_effect
)
X = torch.cat([covariates, treatments], dim=1)
model = MLPNet(n_covariates)
criterion = nn.MSELoss()  # squared error loss
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(n_epochs):
    optimizer.zero_grad()

    # Forward pass
    preds = model(X)

    # Compute squared error loss
    loss = criterion(preds, outcomes)

    # Backprop + update
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss = {loss.item():.4f}")

model.eval()
no_treatment_counterfactuals = torch.cat([covariates, torch.zeros_like(treatments)], dim=1)
treatment_counterfactuals = torch.cat([covariates, torch.ones_like(treatments)], dim=1)
with torch.no_grad():
    Y0 = model(no_treatment_counterfactuals).numpy()
    Y1 = model(treatment_counterfactuals).numpy()
    print(np.mean(Y1 - Y0))
