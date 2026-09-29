"""Case-cluster logit inference shared by the two specification-curve producers."""
import numpy as np
import statsmodels.api as sm


class ClusterInferenceError(RuntimeError):
    """The requested covariance or convergence contract was not satisfied."""


def fit_case_cluster_logit(y, X, clusters, *, maxiter=100):
    """Fit ordinary logit coefficients with case-cluster sandwich covariance.

    A covariance failure must never fall back to nonrobust or HC1 inference.
    The finite-sample correction and normal-reference inference are explicit.
    """
    try:
        cluster_ids, cluster_idx = np.unique(clusters, return_inverse=True)
        if len(cluster_ids) < 2:
            raise ClusterInferenceError('Case-cluster inference requires at least two clusters')
        result = sm.Logit(y, X).fit(
            disp=0, maxiter=maxiter, method='newton', warn_convergence=False,
            cov_type='cluster',
            cov_kwds={'groups': cluster_idx, 'use_correction': True},
            use_t=False,
        )
        if not result.mle_retvals.get('converged', False):
            raise ClusterInferenceError('Case-cluster logit optimizer did not converge')
        if result.cov_type != 'cluster' or result.use_t:
            raise ClusterInferenceError('Requested cluster covariance / normal inference was not applied')
        if result.cov_kwds.get('use_correction') is not True:
            raise ClusterInferenceError('Requested finite-sample covariance correction was not applied')
        covariance = np.asarray(result.cov_params())
        if (covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]
                or not np.all(np.isfinite(covariance)) or np.any(np.diag(covariance) <= 0)):
            raise ClusterInferenceError('Case-cluster covariance is invalid or has nonpositive variance')
    except ClusterInferenceError:
        raise
    except Exception as exc:
        raise ClusterInferenceError('Case-cluster logit fit or covariance failed') from exc
    return result
