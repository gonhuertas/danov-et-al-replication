"""Year-on-year mixed-frequency DFM for the Danov et al. reconstruction.

Statsmodels provides the Kalman smoother, EM state-equation updates, and news.
The subclass replaces its quarter-on-quarter [1,2,3,2,1] aggregation with the
paper's year-on-year [1,1,1] restriction (equations 9 and 14-15).
"""
import numpy as np
from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ


class YearOnYearDFM(DynamicFactorMQ):
    weights = np.array([1.0, 1.0, 1.0, 0.0, 0.0])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.idiosyncratic_ar1:
            raise ValueError("This specification requires AR(1) idiosyncratic components.")
        for q, indices in enumerate(self._s["idio_ar_Q_ix"]):
            self["design", self.k_endog_M + q, indices] = self.weights

    def clone(self, endog, k_endog_monthly=None, endog_quarterly=None, **kwargs):
        if k_endog_monthly is None and endog_quarterly is None:
            k_endog_monthly = self.k_endog_M
        return super().clone(endog, k_endog_monthly=k_endog_monthly,
                             endog_quarterly=endog_quarterly, **kwargs)

    def update(self, params, **kwargs):
        params = self.handle_params(params, transformed=kwargs.get("transformed", True),
                                    includes_fixed=kwargs.get("includes_fixed", False))
        kwargs["transformed"] = True
        super().update(params, **kwargs)
        cursor = sum(len(self._s.endog_factor_iloc[i]) for i in range(self.k_endog_M))
        loadings = params[self._p["loadings"]]
        for i in range(self.k_endog_M, self.k_endog):
            iloc = self._s.endog_factor_iloc[i]
            indices = self._s["factors_L1_5_ix"][:, iloc]
            self["design", i, indices.ravel()] = (
                self.weights[:, None] * loadings[cursor:cursor + len(iloc)]
            ).ravel()
            cursor += len(iloc)
        return params

    @property
    def start_params(self):
        # Fill only inside the training vintage; no unseen observations enter PCA.
        import pandas as pd
        monthly = pd.DataFrame(self.endog[:, :self.k_endog_M])
        filled = monthly.interpolate(limit_direction="both").fillna(0).to_numpy()
        u, singular, _ = np.linalg.svd(filled, full_matrices=False)
        k = self.k_factors
        factors = u[:, :k] * singular[:k] / np.sqrt(self.k_endog_M)
        params = np.zeros(self.k_params)
        loadings, residuals = [], []
        for i in range(self.k_endog):
            iloc = self._s.endog_factor_iloc[i]
            regressors = factors[:, iloc].copy()
            if i >= self.k_endog_M:
                base = regressors.copy()
                for lag in (1, 2):
                    regressors[lag:] += base[:-lag]
            valid = np.isfinite(self.endog[:, i])
            beta = np.linalg.lstsq(regressors[valid], self.endog[valid, i], rcond=None)[0]
            residual = np.full(self.nobs, np.nan)
            residual[valid] = self.endog[valid, i] - regressors[valid] @ beta
            loadings.extend(beta)
            residuals.append(residual)
        params[self._p["loadings"]] = loadings
        ar = np.linalg.lstsq(factors[:-1], factors[1:], rcond=None)[0].T
        radius = max(abs(np.linalg.eigvals(ar)))
        if radius >= 0.98:
            ar *= 0.98 / radius
        innovations = factors[1:] - factors[:-1] @ ar.T
        covariance = innovations.T @ innovations / len(innovations)
        chol = np.linalg.cholesky(covariance + np.eye(k) * 1e-6)
        params[self._p["factor_ar"]] = ar.ravel()
        params[self._p["factor_cov"]] = chol[np.tril_indices(k)]
        rho, variance = [], []
        for i, residual in enumerate(residuals):
            valid = np.isfinite(residual[1:]) & np.isfinite(residual[:-1])
            if i < self.k_endog_M and valid.sum() > 3:
                lag, current = residual[:-1][valid], residual[1:][valid]
                coefficient = float(np.clip(lag @ current / max(lag @ lag, 1e-8), -0.95, 0.95))
                var = np.mean((current - coefficient * lag) ** 2)
            else:
                coefficient, var = 0.0, np.nanvar(residual) / 3
            rho.append(coefficient)
            variance.append(max(float(var), 1e-4))
        params[self._p["idiosyncratic_ar1"]] = rho
        params[self._p["idiosyncratic_var"]] = variance
        return params

    def _em_maximization_obs_missing(self, res, Eaa, a, compute_H=False):
        if compute_H:
            raise ValueError("Measurement variance is zero; AR(1) noise is in the state.")
        state = self._s
        loading_matrix = np.zeros((self.k_endog, state.k_states_factors))
        means = a[..., 0]
        for i in range(self.k_endog):
            iloc = state.endog_factor_iloc[i]
            observed = np.isfinite(self.endog[:, i])
            selector = np.zeros((self.k_states, len(iloc)))
            noise = np.zeros(self.k_states)
            if i < self.k_endog_M:
                current = state["factors_L1"][iloc]
                selector[current, np.arange(len(iloc))] = 1
                noise[state.k_states_factors + i] = 1
            else:
                indices = state["factors_L1_5_ix"][:, iloc]
                for lag in range(3):
                    selector[indices[lag], np.arange(len(iloc))] = 1
                noise[state["idio_ar_Q_ix"][i - self.k_endog_M, :3]] = 1
            moments = Eaa[observed].sum(axis=0)
            lhs = selector.T @ moments @ selector
            rhs = (self.endog[observed, i] @ means[observed] - noise @ moments) @ selector
            beta = np.linalg.solve(lhs + np.eye(len(iloc)) * 1e-10, rhs)
            loading_matrix[i] = (selector @ beta)[:state.k_states_factors]
        return loading_matrix, np.full((self.k_endog, self.k_endog), np.nan)


def make_model(data, factors, standardize=True):
    names = [f"f{i + 1}" for i in range(factors)]
    return YearOnYearDFM(
        data, k_endog_monthly=len(data.columns) - 1, factors=names,
        factor_orders={tuple(names): 1}, idiosyncratic_ar1=True,
        standardize=standardize,
    )
