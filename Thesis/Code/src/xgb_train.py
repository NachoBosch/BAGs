from __future__ import annotations

import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error


def build_feats(
    *,
    Z: np.ndarray | None = None,
    T1: np.ndarray | None = None,
    sex: np.ndarray | None = None,
    diag: np.ndarray | None = None,
) -> np.ndarray:
    """Concatenate feature blocks in a fixed order. All arrays must share row count."""
    parts: list[np.ndarray] = []

    for arr, is_1d in [(Z, False), (T1, False), (sex, True), (diag, True)]:
        if arr is not None:
            a = np.asarray(arr, dtype=np.float32)
            if is_1d:
                a = a.reshape(-1, 1)
            parts.append(a)

    if not parts:
        raise ValueError("At least one feature block must be provided.")
    return np.concatenate(parts, axis=1)


def build_feature_names(
    *,
    z_dim: int = 0,
    t1_dim: int = 0,
    use_sex: bool = False,
    use_diag: bool = False,
) -> list[str]:
    """Feature name list matching the column order produced by build_feats."""
    names: list[str] = []
    names += [f"z_mu_{i:03d}" for i in range(z_dim)]
    names += [f"t1_{i:04d}" for i in range(t1_dim)]
    if use_sex:
        names.append("sex")
    if use_diag:
        names.append("diag")
    return names


def clean_xy(X, y):
    """Sanitize features (NaN/inf -> 0) and drop rows where y is not finite."""
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(y)
    return X[mask], y[mask]


def train_xgb(Xtr, ytr, params: dict) -> XGBRegressor:
    """Fit an XGBRegressor with the given parameters."""
    model = XGBRegressor(**params)
    model.fit(Xtr, ytr, verbose=False)
    return model


def eval_xgb(model: XGBRegressor, Xte, yte) -> dict:
    """Predict and return MAE plus raw predictions."""
    pred = model.predict(Xte)
    return {"mae": float(mean_absolute_error(yte, pred)), "pred": pred}
