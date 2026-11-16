"""Hidden mixtures of nonnegative Gaussian session demands."""
import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import logsumexp, softmax
from scipy.stats import norm, truncnorm

from evopt.learning.gaussian import GaussianDemand, GaussianEstimator


class GaussianMixtureDemand:
    def __init__(self, components, weights):
        self.components = tuple(c if isinstance(c, GaussianDemand) else GaussianDemand(**c)
                                for c in components)
        self.weights = np.asarray(weights, float)
        if (not self.components or self.weights.shape != (len(self.components),)
                or not np.isfinite(self.weights).all() or (self.weights <= 0).any()
                or not np.isclose(self.weights.sum(), 1.)):
            raise ValueError("provide one positive weight per component, summing to one")
        self.weights = self.weights / self.weights.sum()
        self.mean = float(sum(w * c.mean for w, c in zip(self.weights, self.components)))
        self.sd = float(np.sqrt(sum(w * (c.sd ** 2 + (c.mean - self.mean) ** 2)
                                    for w, c in zip(self.weights, self.components))))

    def sample_with_types(self, size, rng):
        types = rng.choice(len(self.components), size=size, p=self.weights)
        demand = np.empty(size)
        for j, component in enumerate(self.components):
            mask = types == j
            demand[mask] = component.sample(int(mask.sum()), rng)
        return demand, types

    def sample(self, size, rng):
        return self.sample_with_types(size, rng)[0]

    def conditional_weights(self, lower):
        lower = np.asarray(lower, float)
        logs = np.stack([np.log(w) + norm.logsf((lower - c.mean) / c.sd)
                         - norm.logsf(-c.mean / c.sd)
                         for w, c in zip(self.weights, self.components)], axis=-1)
        return np.exp(logs - logsumexp(logs, axis=-1, keepdims=True))

    def shortfall(self, delivered, lower=0.0):
        delivered, lower = np.broadcast_arrays(np.asarray(delivered, float), np.asarray(lower, float))
        weights = self.conditional_weights(lower)
        values, slopes = zip(*(c.shortfall(delivered, lower) for c in self.components))
        return (np.sum(weights * np.stack(values, axis=-1), axis=-1),
                np.sum(weights * np.stack(slopes, axis=-1), axis=-1))

    def quantile(self, probability, lower=0.0):
        if not 0 < probability < 1:
            raise ValueError("probability must lie strictly between zero and one")
        lower = np.asarray(lower, float)
        result = np.empty_like(lower)
        for index in np.ndindex(lower.shape):
            bound = float(lower[index])
            upper = max(float(c.quantile(probability, bound)) for c in self.components)

            def residual(value):
                _, slope = self.shortfall(value, bound)
                return 1. + float(slope) - probability

            result[index] = brentq(residual, bound, upper + 1e-8)
        return result

    def moments(self):
        moments = [truncnorm.stats(-c.mean / c.sd, np.inf, loc=c.mean, scale=c.sd,
                                   moments="mv") for c in self.components]
        mean = sum(w * m for w, (m, v) in zip(self.weights, moments))
        variance = sum(w * (v + (m - mean) ** 2) for w, (m, v) in zip(self.weights, moments))
        return float(mean), float(variance)


class GaussianMixtureEstimator(GaussianEstimator):
    def __init__(self, initial_components, initial_weights, min_sd=0.5, prior_weight=0.1):
        super().__init__(min_sd=min_sd, prior_weight=prior_weight)
        order = np.argsort([c["mean"] for c in initial_components])
        self.initial = GaussianMixtureDemand([initial_components[j] for j in order],
                                             np.asarray(initial_weights)[order])
        self.model = self.initial

    def fit(self):
        if not self.records:
            return self.model
        values, exact = map(np.asarray, zip(*self.records.values()))
        exact = exact.astype(bool)
        n = len(self.initial.components)
        initial_means = np.array([c.mean for c in self.initial.components])
        initial_sd = np.array([c.sd for c in self.initial.components])

        def unpack(parameters):
            return parameters[:n], np.exp(parameters[n:2*n]), softmax(np.r_[parameters[2*n:], 0.])

        def objective(parameters):
            means, sd, weights = unpack(parameters)
            z = (values[:, None] - means) / sd
            logs = np.where(exact[:, None], norm.logpdf(z) - np.log(sd), norm.logsf(z))
            logs -= norm.logsf(-means / sd)
            likelihood = logsumexp(logs + np.log(weights), axis=1).sum()
            prior = np.sum(((means - initial_means) / initial_sd) ** 2)
            prior += np.sum(np.log(sd / initial_sd) ** 2)
            prior += np.sum(np.log(weights / self.initial.weights) ** 2)
            return -likelihood + self.prior_weight * prior

        current = self.model
        start = np.r_[[c.mean for c in current.components],
                      np.log([c.sd for c in current.components]),
                      np.log(current.weights[:-1] / current.weights[-1])]
        bounds = [(0., None)] * n + [(np.log(self.min_sd), np.log(1e4))] * n + [(-15., 15.)] * (n - 1)
        result = minimize(objective, start, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": 300})
        self.fit_success = bool(result.success and np.isfinite(result.fun))
        if self.fit_success:
            means, sd, weights = unpack(result.x)
            order = np.argsort(means)
            self.model = GaussianMixtureDemand(
                [GaussianDemand(float(means[j]), float(sd[j])) for j in order], weights[order])
        return self.model
