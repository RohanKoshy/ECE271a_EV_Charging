"""Nonnegative Gaussian demand with exact and right-censored observations."""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, truncnorm


@dataclass(frozen=True)
class GaussianDemand:
    mean: float
    sd: float

    def __post_init__(self):
        if not np.isfinite([self.mean, self.sd]).all() or self.mean < 0 or self.sd <= 0:
            raise ValueError("mean must be nonnegative and sd positive, both finite")

    def moments(self):
        return truncnorm.stats(-self.mean / self.sd, np.inf, loc=self.mean,
                               scale=self.sd, moments="mv")

    def sample(self, size, rng):
        return truncnorm.rvs(-self.mean / self.sd, np.inf, loc=self.mean,
                             scale=self.sd, size=size, random_state=rng)

    def quantile(self, probability, lower=0.0):
        return truncnorm.ppf(probability, (np.asarray(lower) - self.mean) / self.sd,
                             np.inf, loc=self.mean, scale=self.sd)

    def shortfall(self, delivered, lower=0.0):
        d, lower = np.broadcast_arrays(np.asarray(delivered, float), np.asarray(lower, float))
        a = (lower - self.mean) / self.sd
        z = (np.maximum(d, lower) - self.mean) / self.sd
        denominator = norm.logsf(a)
        survival = np.exp(norm.logsf(z) - denominator)
        density = np.exp(norm.logpdf(z) - denominator)
        value = self.sd * density + (self.mean - np.maximum(d, lower)) * survival
        return value + np.maximum(lower - d, 0), -survival


class GaussianEstimator:
    def __init__(self, initial_mean=12.0, initial_sd=6.0, min_sd=0.25, prior_weight=0.1):
        self.initial = GaussianDemand(initial_mean, initial_sd)
        if not np.isfinite([min_sd, prior_weight]).all() or min_sd <= 0 or prior_weight < 0:
            raise ValueError("min_sd must be positive and prior_weight nonnegative")
        self.model = self.initial
        self.min_sd = min_sd
        self.prior_weight = prior_weight
        self.records = {}
        self.fit_success = True

    def observe(self, key, delivered_kwh, complete):
        value = float(delivered_kwh)
        if not np.isfinite(value) or value < 0:
            raise ValueError("delivered energy must be finite and nonnegative")
        old = self.records.get(key)
        if old is not None and (value < old[0] - 1e-8 or (old[1] and (not complete or abs(value - old[0]) > 1e-8))):
            raise ValueError("session observations must be cumulative and completion final")
        if complete or value > 0:
            self.records[key] = (value, bool(complete))

    def fit(self):
        if not self.records:
            return self.model
        values, exact = map(np.asarray, zip(*self.records.values()))
        exact = exact.astype(bool)

        def objective(parameters):
            mean, log_sd = parameters
            sd = np.exp(log_sd)
            normalizer = norm.logsf(-mean / sd)
            log_likelihood = np.where(
                exact, norm.logpdf((values - mean) / sd) - log_sd,
                norm.logsf((values - mean) / sd)) - normalizer
            prior = ((mean - self.initial.mean) / self.initial.sd) ** 2
            prior += (log_sd - np.log(self.initial.sd)) ** 2
            return -log_likelihood.sum() + self.prior_weight * prior

        result = minimize(objective, [self.model.mean, np.log(self.model.sd)],
                          method="L-BFGS-B", bounds=[(0, None), (np.log(self.min_sd), None)])
        self.fit_success = bool(result.success and np.isfinite(result.fun))
        if self.fit_success:
            self.model = GaussianDemand(float(result.x[0]), float(np.exp(result.x[1])))
        return self.model
