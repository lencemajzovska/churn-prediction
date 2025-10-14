"""Enhetstester för model.py
(träning, kalibrering, feature importance).
"""

import pytest
import numpy as np
import pandas as pd
import warnings

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.model import (
    split_data,
    train_and_evaluate,
    evaluate_model,
    calibrate_model,
    feature_importance,
)

@pytest.fixture(scope="module")
def toy_data_small():
    X = pd.DataFrame({
        "feat1": [0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.8],
        "feat2": [1, 0, 1, 0, 1, 0, 1, 0],
    })
    y = pd.Series([0, 1, 0, 1, 1, 0, 0, 1], name="churned")
    ids = pd.Series(range(100, 108), name="customer_id")
    return X, y, ids


@pytest.fixture(scope="module")
def toy_data_big():
    X_arr, y_arr = make_classification(
        n_samples=200, n_features=6, n_informative=4, n_redundant=0,
        n_clusters_per_class=1, weights=[0.7, 0.3], random_state=42
    )
    X = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(X_arr.shape[1])])
    y = pd.Series(y_arr, name="churned")
    ids = pd.Series(np.arange(1, len(y)+1), name="customer_id")
    return X, y, ids


def test_split_data(toy_data_small):
    X, y, ids = toy_data_small
    X_tr, X_te, y_tr, y_te, ids_tr, ids_te, feats = split_data(
        X, y, ids, random_state=42, test_size=0.25
    )
    assert len(X_tr) + len(X_te) == len(X)
    assert ids_te.is_unique
    assert feats == list(X.columns)
    assert list(X_tr.columns) == feats == list(X_te.columns)


def test_evaluate_model(toy_data_small):
    X, y, _ = toy_data_small
    mdl = LogisticRegression(max_iter=200, solver="liblinear")
    res = evaluate_model("LogReg", mdl, X, y, X, y)
    for k in ["name", "model", "proba", "auc", "f1_05", "best_thr", "f1_best", "precision_at_k"]:
        assert k in res
    assert isinstance(res["proba"], np.ndarray)
    assert res["proba"].shape[0] == len(y)
    assert 0.0 <= res["f1_05"] <= 1.0
    assert 0.0 <= res["f1_best"] <= 1.0
    assert (0.0 <= res["auc"] <= 1.0) or np.isnan(res["auc"])


def test_model_selection(toy_data_big):
    X, y, _ = toy_data_big
    best_model, best_name, cv_df = train_and_evaluate(X, y, random_state=42)
    assert best_model is not None
    assert isinstance(best_name, str) and len(best_name) > 0
    assert not cv_df.empty
    assert {"model", "auc_mean", "auc_std", "folds"}.issubset(cv_df.columns)
    assert cv_df.iloc[0]["model"] == best_name


def test_calibration(toy_data_big):
    X, y, _ = toy_data_big
    base = LogisticRegression(max_iter=200, solver="liblinear")
    calib, used = calibrate_model(base, X, y, method="isotonic")
    assert used in {"isotonic", "sigmoid", "none"}
    assert hasattr(calib, "predict_proba")
    proba = calib.predict_proba(X)
    assert proba.shape == (len(X), 2)


def test_calibration_single_class():
    X = pd.DataFrame({"f": [0.1, 0.2, 0.3, 0.4]})
    y = pd.Series([0, 0, 0, 0], name="churned")
    base = LogisticRegression(max_iter=200, solver="liblinear")
    calib, used = calibrate_model(base, X, y, method="isotonic")
    assert used == "none"
    calib.fit(X, y)
    assert hasattr(calib, "predict_proba")


def _assert_importance_df(df, allow_std_optional: bool = True):
    assert isinstance(df, pd.DataFrame)
    assert "feature" in df.columns
    assert "importance" in df.columns
    if not allow_std_optional:
        assert "importance_std" in df.columns
    assert not df.empty
    assert (df["importance"] >= 0).all()


def test_feature_importance_permutation(toy_data_big):
    X, y, _ = toy_data_big
    lr = LogisticRegression(max_iter=200, solver="liblinear").fit(X, y)
    imp = feature_importance(lr, X, y, list(X.columns), perm_repeats=3)
    _assert_importance_df(imp, allow_std_optional=False)


def test_feature_importance_shap(toy_data_big):
    X, y, _ = toy_data_big
    rf = RandomForestClassifier(n_estimators=50, random_state=42).fit(X, y)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*the 'data' parameter is deprecated.*")
    imp = feature_importance(rf, X, y, list(X.columns), perm_repeats=2)
    assert "feature" in imp.columns and "importance" in imp.columns
    assert not imp.empty