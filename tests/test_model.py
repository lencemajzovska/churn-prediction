import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from model import (
    best_f1_threshold,
    precision_at_k,
    evaluate_model,
    get_models,
    train_compare,
    select_best,
    cross_validate_auc,
    calibrate_model,
)


@pytest.fixture
def sample_data():
    """Skapar ett enkelt dataset för tester."""
    X = pd.DataFrame({
        "feat1": [0.1, 0.3, 0.5, 0.7],
        "feat2": [1, 0, 1, 0],
    })
    y = pd.Series([0, 1, 0, 1], name="churned")
    ids = pd.Series([10, 20, 30, 40], name="customer_id")
    return X, y, ids


def test_best_f1_threshold():
    """Kontrollerar att best_f1_threshold returnerar tröskel och F1 inom [0,1]."""
    y_true = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.4, 0.6, 0.9])
    thr, f1 = best_f1_threshold(y_true, proba)
    assert 0 <= thr <= 1
    assert 0 <= f1 <= 1


def test_precision_at_k():
    """Kontrollerar att precision_at_k returnerar värde mellan 0 och 1."""
    y_true = [0, 1, 0, 1, 1]
    proba = [0.2, 0.8, 0.1, 0.7, 0.9]
    prec = precision_at_k(y_true, proba, k=0.4)
    assert 0 <= prec <= 1


def test_evaluate_model(sample_data):
    """Kontrollerar att evaluate_model returnerar alla förväntade mått och sannolikheter."""
    X, y, _ = sample_data
    model = LogisticRegression(max_iter=50, solver="liblinear")  # ✨ la till solver
    res = evaluate_model("LogReg", model, X, y, X, y)
    assert "auc" in res
    assert "f1_best" in res
    assert isinstance(res["proba"], np.ndarray)
    assert res["proba"].shape[0] == len(y)
    assert 0 <= res["f1_best"] <= 1  # ✨ extra check
    assert 0 <= res["auc"] <= 1 or np.isnan(res["auc"])  # ✨ extra check


def test_get_models():
    """Kontrollerar att get_models returnerar minst en LogReg-modell."""
    models = get_models(random_state=42, scale_pos_weight=1.0)
    assert isinstance(models, list)
    assert any("LogReg" in name for name, _ in models)


def test_train_compare(sample_data):
    """Kontrollerar att train_compare returnerar resultat, tabell och proba_df."""
    X, y, ids = sample_data
    results, compare_df, proba_df, y_test, ids_test = train_compare(
        X, y, ids, random_state=42, test_size=0.5
    )
    assert isinstance(results, list)
    assert not compare_df.empty
    assert not proba_df.empty
    assert len(y_test) == len(ids_test)
    # ✨ kolla att viktiga kolumner finns
    assert set(["model", "auc", "f1_05", "best_thr", "f1_best"]).issubset(compare_df.columns)


def test_select_best(sample_data):
    """Kontrollerar att select_best väljer modell med högst AUC."""
    X, y, ids = sample_data
    results, _, _, _, _ = train_compare(X, y, ids, random_state=42, test_size=0.5)
    best = select_best(results)
    assert isinstance(best, dict)
    assert "auc" in best
    assert "name" in best


def test_cross_validate_auc(sample_data):
    """Kontrollerar att cross_validate_auc returnerar tabell med auc_mean och auc_std."""
    X, y, _ = sample_data
    models = get_models(random_state=42, scale_pos_weight=1.0)
    df = cross_validate_auc(models, X, y, random_state=42, n_splits=2)
    assert "auc_mean" in df.columns
    assert "auc_std" in df.columns
    assert not df.empty
    assert df["auc_mean"].between(0, 1).all()  # ✨ extra check


def test_calibrate_model_isotonic_or_fallback(sample_data):
    """Kontrollerar att kalibrering fungerar (isotonic eller fallback)."""
    X, y, _ = sample_data
    base_model = LogisticRegression(max_iter=50, solver="liblinear")  # ✨ la till solver
    calib = calibrate_model(base_model, X, y, method="isotonic")
    # Oavsett metod ska modellen kunna ge predict_proba
    assert hasattr(calib, "predict_proba")
    preds = calib.predict_proba(X)
    assert preds.shape[0] == len(X)


def test_calibrate_model_tiny_dataset():
    """Kontrollerar att ett extremt litet dataset inte kraschar (sigmoid eller okalibrerad)."""
    X = pd.DataFrame({"feat1": [0.1, 0.9]})
    y = pd.Series([0, 1], name="churned")
    base_model = LogisticRegression(max_iter=50, solver="liblinear")  # ✨ la till solver

    calib = calibrate_model(base_model, X, y, method="isotonic")
    # Ska alltid gå att använda för prediktion
    if hasattr(calib, "predict_proba"):
        preds = calib.predict_proba(X)
        assert preds.shape[0] == len(X)
    else:
        preds = calib.predict(X)
        assert preds.shape[0] == len(X)
