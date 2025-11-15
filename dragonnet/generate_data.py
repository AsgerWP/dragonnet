import numpy as np


def _generate_data(n=1000, p=10, average_treatment_effect=3):
    covariates = np.random.normal(0, 1, (n, p))
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
        + covariates[:, 1]
        + covariates[:, 2]
        + covariates[:, 4]
    )
