import numpy as np

def irls_fit(x, y, order=2, max_iter=50, tol=1e-6, eps=1e-6):
    X = np.vstack([x**i for i in range(order, -1, -1)]).T
    beta = np.linalg.lstsq(X, y, rcond=None)[0]

    for _ in range(max_iter):
        residuals = y - X @ beta
        weights = 1.0 / (np.abs(residuals) + eps)
        W = np.diag(weights)
        beta_new = np.linalg.lstsq(W @ X, W @ y, rcond=None)[0]
        if np.linalg.norm(beta_new - beta) < tol:
            break
        beta = beta_new
    return beta
