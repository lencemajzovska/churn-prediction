"""
Modellträning:

- train/test-split
- val av modell
- cross-validation
- utvärdering
- kalibrering av sannolikheter
- feature importance
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

import shap
from typing import List, Tuple

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.dummy import DummyClassifier

from xgboost import XGBClassifier

log = logging.getLogger(__name__)


# === Train/Test-split ===
def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    ids: pd.Series,
    random_state: int = 42,
    test_size: float = 0.25,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series, List[str]]:
    """
    Delar upp data i train/test med stratifiering (bevarar churn-fördelning).
    """
    # Inputvalidering
    if X.empty or y.empty:
        raise ValueError("split_data() received empty X or y.")
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows.")

    # Stratifiering = samma churn-fördelning i train/test
    X_tr, X_te, y_tr, y_te, ids_tr, ids_te = train_test_split(
        X, y, ids,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    assert ids_te.is_unique, "ids_test contains duplicates."

    # Säkerställ samma featureordning i train/test
    feature_names = X.columns.tolist()
    X_tr = X_tr.loc[:, feature_names]
    X_te = X_te.loc[:, feature_names]

    log.debug(
        "Train/Test split completed — Train: %s, Test: %s (churn rate: %.1f%%)",
        len(y_tr), len(y_te), float(y.mean() * 100),
    )
    return X_tr, X_te, y_tr, y_te, ids_tr, ids_te, feature_names


# === Modelluppsättning ===
def _get_models(random_state: int, scale_pos_weight: float) -> list[tuple[str, object]]:
    """Definierar modeller som utvärderas."""
    return [
        (
            "LogReg (baseline)",
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=1000,
                            class_weight="balanced",
                            solver="liblinear",
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
        ),
        (
            "RandomForest",
            RandomForestClassifier(
                n_estimators=300,
                max_features="sqrt",
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=random_state,
            ),
        ),
        (
            "XGBoost",
            XGBClassifier(
                n_estimators=400,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                min_child_weight=1.0,
                reg_lambda=1.0,
                scale_pos_weight=scale_pos_weight,
                eval_metric="auc",
                tree_method="hist",
                n_jobs=-1,
                random_state=random_state,
            ),
        ),
    ]


# === Hjälpfunktioner ===
def _predict_proba_safely(model, X: pd.DataFrame) -> np.ndarray:
    """Hämtar sannolikheter från modellen oavsett modelltyp."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        if scores.ndim > 1:
            scores = scores.ravel()
        proba = MinMaxScaler().fit_transform(scores.reshape(-1, 1)).ravel()
    else:
        # Fallback: prediktion som sannolikhet
        proba = model.predict(X).astype(float)
    return np.nan_to_num(proba, nan=0.0, posinf=1.0, neginf=0.0)


# === Utvärdering ===
def evaluate_model(
    name: str,
    model,
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_te: pd.DataFrame,
    y_te: pd.Series,
    k: float = 0.10,
    thr: float = 0.50,
) -> dict:
    """Tränar modell och utvärderar prestanda på testdata."""
    model.fit(X_tr, y_tr)
    proba = _predict_proba_safely(model, X_te)

    # Standardmått för churnmodell
    auc_val = roc_auc_score(y_te, proba) if len(np.unique(y_te)) > 1 else float("nan")
    pred_05 = (proba >= thr).astype(int)
    f1_05 = f1_score(y_te, pred_05)

    # Bästa F1 baserat på threshold
    precisions, recalls, thresholds = precision_recall_curve(y_te, proba)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    i = int(np.argmax(f1s))
    best_thr = float(thresholds[i]) if i < len(thresholds) else thr
    f1_best = float(f1s[i])

    # Precision@K (för top-prioritering)
    n = max(1, int(len(proba) * k))
    idx_top = np.argsort(-proba)[:n]
    prec_at_k = float(np.take(y_te.to_numpy(), idx_top).sum() / n)

    log.debug(
        "%s — AUC: %.3f | F1@0.50: %.3f | Best F1: %.3f | Precision@K: %.3f",
        name, auc_val, f1_05, f1_best, prec_at_k,
    )

    return {
        "name": name,
        "model": model,
        "proba": proba,
        "auc": auc_val,
        "f1_05": f1_05,
        "best_thr": best_thr,
        "f1_best": f1_best,
        "precision_at_k": prec_at_k,
    }


# === Cross-validation och modellval ===
def train_and_evaluate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> tuple[object, str, pd.DataFrame]:
    """Väljer bästa modell med cross-validation (AUC)."""

    # Hantera obalanserad data genom viktning
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    spw = (neg / max(pos, 1)) if pos else 1.0
    log.debug("Calculated scale_pos_weight = %.2f (neg=%s, pos=%s)", spw, neg, pos)

    models = _get_models(random_state, spw)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    cv_results = []
    for name, model in models:
        aucs = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="roc_auc",
            n_jobs=1
        )
        cv_results.append({
            "model": name,
            "auc_mean": aucs.mean(),
            "auc_std": aucs.std(),
            "folds": cv.get_n_splits()
        })
        log.debug("%s | CV AUC = %.3f ± %.3f", name, aucs.mean(), aucs.std())

    cv_df = (
        pd.DataFrame(cv_results)
        .round(3)
        .sort_values("auc_mean", ascending=False)
        .reset_index(drop=True)
    )
    best_name = str(cv_df.iloc[0]["model"])
    best_model = next(m for n, m in models if n == best_name)

    log.debug("Best model selected by CV: %s", best_name)
    return best_model, best_name, cv_df


# === Kalibrering av modell ===
def calibrate_model(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: str = "isotonic",
) -> tuple[object, str]:
    """
    Kalibrerar sannolikhetsutmatning från modellen.
    Använder isotonic eller sigmoid som fallback.
    """
    if y_train.nunique() < 2:
        # Hantera edge-case: bara en klass
        dummy = DummyClassifier(strategy="constant", constant=y_train.iloc[0])
        dummy.fit(X_train, y_train)
        return dummy, "none"

    try:
        # Isotonic ger ofta bäst kalibrering
        calib = CalibratedClassifierCV(estimator=clone(model), cv=5, method=method)
        calib.fit(X_train, y_train)
        return calib, method

    except ValueError:
        # Fallback om isotonic inte stöds
        calib = CalibratedClassifierCV(estimator=clone(model), cv=5, method="sigmoid")
        calib.fit(X_train, y_train)
        return calib, "sigmoid"


# === Feature importance ===
def feature_importance(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: List[str],
    perm_repeats: int = 5,
) -> pd.DataFrame:
    """
    Beräknar feature importance för den tränade modellen.
    - SHAP används för trädmodeller (RandomForest, XGBoost)
    - Permutation Importance används för övriga modeller eller som fallback
    """

    # Hanterar modell inuti Pipeline/CalibratedClassifierCV
    def _unwrap_estimator(m):
        if isinstance(m, Pipeline):
            return m.named_steps.get("clf", m)
        if isinstance(m, CalibratedClassifierCV):
            return getattr(m, "estimator", m)
        return m

    est = _unwrap_estimator(model)
    is_tree = isinstance(est, (RandomForestClassifier, XGBClassifier))

    if is_tree:
        try:
            # SHAP för tolkning av trädmodeller
            explainer = shap.TreeExplainer(est)
            shap_values = explainer.shap_values(X_test)

            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                shap_values = (
                    shap_values[:, :, 1]
                    if shap_values.shape[2] == 2
                    else shap_values.mean(axis=2)
                )

            importance = np.abs(shap_values).mean(axis=0)
            importance_df = pd.DataFrame({"feature": feature_names, "importance": importance})
            method = "SHAP"

        except Exception:
            # Fallback till Permutation Importance
            perm = permutation_importance(
                model,
                X_test,
                y_test,
                n_repeats=perm_repeats,
                random_state=42,
                n_jobs=-1,
            )
            importance_df = pd.DataFrame(
                {
                    "feature": feature_names,
                    "importance": perm.importances_mean,
                    "importance_std": perm.importances_std,
                }
            )
            method = "Permutation (fallback)"
    else:
        perm = permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=perm_repeats,
            random_state=42,
            n_jobs=-1,
        )
        importance_df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": perm.importances_mean,
                "importance_std": perm.importances_std,
            }
        )
        method = "Permutation"

    importance_df = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)
    log.debug("Feature importance calculated using %s (%s features).", method, len(importance_df))

    return importance_df