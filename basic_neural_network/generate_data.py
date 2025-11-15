import numpy as np
import torch


def generate_data(n=1000, n_covariates=10, average_treatment_effect=3):
    covariates, treatments, outcomes = _generate_data(
        n, n_covariates, average_treatment_effect
    )
    return (
        torch.from_numpy(covariates).float(),
        torch.from_numpy(treatments).float().unsqueeze(1),
        torch.from_numpy(outcomes).float().unsqueeze(1),
    )


def _generate_data(n=1000, n_covariates=10, average_treatment_effect=3):
    covariates = np.random.normal(0, 1, (n, n_covariates))
    probits = covariates[:, 0] + covariates[:, 1] + covariates[:, 2]
    treatment_probabilities = 1 / (1 + np.exp(-probits))
    treatments = np.random.binomial(1, treatment_probabilities)
    outcomes = (
        average_treatment_effect * treatments
        + covariates[:, 0]
        + covariates[:, 1]
        + covariates[:, 2]
        + np.random.normal(0, 1, n)
    )
    return covariates, treatments, outcomes


