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
    probits = _calculate_probits(covariates)
    treatment_probabilities = 1 / (1 + np.exp(-probits))
    treatments = np.random.binomial(1, treatment_probabilities)
    outcomes = _calculate_outcomes(covariates, average_treatment_effect, treatments)
    return covariates, treatments, outcomes


def _calculate_probits(covariates):
    return covariates[:, 1] + covariates[:, 2] + covariates[:, 3]


def _calculate_outcomes(covariates, theta, average_treatment_effect):
    return (
        theta * average_treatment_effect
        + np.sin(covariates[:, 1])
        + np.cos(covariates[:, 2])
        + covariates[:, 4]
    )
