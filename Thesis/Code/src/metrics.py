import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict:
    """Compute MAE, RMSE, R-squared and Pearson correlation."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    yt = y_true - y_true.mean()
    yp = y_pred - y_pred.mean()
    pearson = float(
        np.sum(yt * yp) / (np.sqrt(np.sum(yt**2)) * np.sqrt(np.sum(yp**2)) + 1e-8)
    )

    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(root_mean_squared_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "Pearson": pearson,
    }
