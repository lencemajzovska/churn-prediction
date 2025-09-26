from __future__ import annotations
import logging
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier

log = logging.getLogger(__name__)


def best_f1_threshold(y_true, proba, grid=None) -> Tuple[float, float]:
    """
    Beräkna vilket tröskelvärde (0–1) som ger högst F1-score.
    Returnerar både tröskeln och motsvarande F1.
    """
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)
    f1s = [f1_score(y_true, (proba >= t).astype(int)) for t in grid]
    i = int(np.argmax(f1s))
    return float(grid[i]), float(f1s[i])


def precision_at_k(y_true, proba, k: float = 0.1) -> float:
    """
    Precision bland de k% kunder med högst predicerad risk.
    Nyttigt för att mäta hur bra modellen identifierar toppkandidater
    för churn (t.ex. för riktade kampanjer).
    """
    y_true = pd.Series(y_true).reset_index(drop=True)
    proba = pd.Series(proba).reset_index(drop=True)

    n = max(1, int(len(proba) * k))
    if n == 1 and k < 1 / len(proba):
        log.warning("precision_at_k: k=%.2f gav <1 sample, justerat till 1.", k)

    idx = np.argsort(-proba.values)[:n]
    return float((y_true.iloc[idx].sum()) / n)


def evaluate_model(
    name: str,
    model,
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_te: pd.DataFrame,
    y_te: pd.Series,
    k: float = 0.1,
    thr: float = 0.50
) -> Dict[str, Any]:
    """
    Tränar en modell och returnerar utvärderingsmått.

    Returnerar en dict med:
    - name: modellnamn
    - model: det tränade modellobjektet
    - proba: predikterade sannolikheter
    - auc: ROC AUC
    - f1_05: F1-score vid tröskel 0.5
    - best_thr: tröskel som gav bäst F1-score
    - f1_best: bästa F1-score
    - precision_at_k: precision bland topp-k kunder
    """
    model.fit(X_tr, y_tr)

    # Prediktera sannolikheter (fallback om metoden saknas)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_te)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_te).reshape(-1, 1)
        proba = MinMaxScaler().fit_transform(scores).ravel()
    else:
        proba = model.predict(X_te).astype(float)

    # Hantera ev. NaN/inf
    proba = np.nan_to_num(proba, nan=0.0, posinf=1.0, neginf=0.0)

    # AUC
    auc_val = roc_auc_score(y_te, proba) if len(np.unique(y_te)) > 1 else float("nan")

    # F1-score vid tröskel 0.5
    pred = (proba >= thr).astype(int)
    f1_05 = f1_score(y_te, pred)

    # Bästa threshold enligt F1
    precisions, recalls, thresholds = precision_recall_curve(y_te, proba)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    i = np.argmax(f1s)
    best_thr = float(thresholds[i]) if i < len(thresholds) else thr
    f1_best = float(f1s[i])

    # Precision@K
    prec_at_k = precision_at_k(y_te, proba, k=k)

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


def get_models(random_state: int, scale_pos_weight: float) -> List[Tuple[str, object]]:
    """
    Skapar en lista med modeller som ska jämföras.
    """
    log.info("Initierar kandidater: LogReg, RandomForest, XGBoost")
    return [
        (
            "LogReg (baseline)",
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=random_state
                )),
            ]),
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


def train_compare(
    X: pd.DataFrame,
    y: pd.Series,
    ids: pd.Series,
    random_state: int = 42,
    test_size: float = 0.25
) -> Tuple[List[dict], pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Tränar och jämför flera modeller på ett test-set.

    Returnerar:
      - results: lista med dicts från evaluate_model
      - compare_df: tabell med AUC/F1 för modellerna
      - proba_df: DataFrame med sannolikheter per kund
      - y_test: faktiska churn-labels för test-set
      - ids_test: kund-ID:n för test-set
    """
    log.info("Startar train/test-split (test_size=%s)", test_size)

    # Dela upp data i train/test
    X_tr, X_te, y_tr, y_te, ids_tr, ids_te = train_test_split(
        X, y, ids,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Beräkna class weight för obalanserad data (används i XGBoost)
    pos = int(y_tr.sum())
    neg = int(len(y_tr) - pos)
    spw = (neg / max(pos, 1)) if pos else 1.0
    log.info(
        "Training data: %s kunder (%s pos, %s neg). scale_pos_weight=%.2f",
        len(y_tr), pos, neg, spw
    )

    models = get_models(random_state, spw)

    results: List[dict] = []
    proba_frames = []

    # Träna och utvärdera varje modell
    for name, mdl in models:
        res = evaluate_model(name, mdl, X_tr, y_tr, X_te, y_te, k=0.10, thr=0.50)
        results.append(res)

        # Spara sannolikheter per kund (för BI/analys)
        proba_frames.append(pd.DataFrame({
            "customer_id": ids_te,
            f"proba_{name}": res["proba"],
        }))

    # Sammanfattande jämförelsetabell
    compare_df = pd.DataFrame({
        "model":     [r["name"] for r in results],
        "auc":       [r["auc"] for r in results],
        "f1_05":     [r["f1_05"] for r in results],
        "best_thr":  [r["best_thr"] for r in results],
        "f1_best":   [r["f1_best"] for r in results],
    }).round(3).sort_values(["auc", "f1_best"], ascending=False)

    # Kombinera sannolikheter per modell i en tabell
    proba_df = pd.concat(proba_frames, axis=1)
    proba_df = proba_df.loc[:, ~proba_df.columns.duplicated()]
    log.info("Jämförelse klar. %s modeller utvärderade.", len(results))

    return results, compare_df, proba_df, y_te, ids_te


def select_best(results: List[dict]) -> dict:
    """
    Väljer bästa modell baserat på högsta AUC.
    """
    best = max(results, key=lambda r: r["auc"])
    log.info("Vald bästa modell: %s (AUC=%.3f)", best["name"], best["auc"])
    return best


def cross_validate_auc(
    models: List[Tuple[str, object]],
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
    n_splits: int = 5
) -> pd.DataFrame:
    """
    Kör cross-validation på flera modeller.
    Returnerar en tabell med AUC (medelvärde och standardavvikelse).
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows = []
    for name, mdl in models:
        log.info("CV för %s (%s folds)", name, n_splits)
        aucs = cross_val_score(mdl, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        rows.append({
            "model": name,
            "auc_mean": float(aucs.mean()),
            "auc_std": float(aucs.std()),
            "n": len(aucs),
        })
    log.info("Cross-validation klar för %s modeller", len(models))
    return pd.DataFrame(rows).round(3).sort_values("auc_mean", ascending=False)


def calibrate_model(base_model, X_train, y_train, method: str = "isotonic"):
    """
    Kalibrerar en modell för att förbättra sannolikhetsskattningarna.

    Försöker först träna kalibreringen med den metod som anges i argumentet
    (t.ex. "isotonic" eller "sigmoid").
    Om det misslyckas → fallback till sigmoid.
    Om även det misslyckas → returnerar en okalibrerad modell.

    Parametrar
    ----------
    base_model : sklearn-modell
        Basmodell att kalibrera.
    X_train : pd.DataFrame
        Träningsdata.
    y_train : pd.Series
        Labels.
    method : str, default="isotonic"
        Första metod som försöks ("isotonic" eller "sigmoid").

    Returns
    -------
    model : sklearn-modell
        Kalibrerad modell (eller okalibrerad om båda metoderna misslyckas).
    """
    try:
        log.info("Kalibrerar modell (%s)", method)

        n_samples = len(y_train)
        n_classes = y_train.nunique()

        # Dynamiskt cv
        max_cv = min(5, n_samples // n_classes)
        if max_cv < 2:
            raise ValueError("För få samples för CV-kalibrering")

        calib = CalibratedClassifierCV(
            estimator=clone(base_model),
            cv=max_cv,
            method=method
        )
        calib.fit(X_train, y_train)
        return calib

    except Exception as e:
        log.warning("Isotonic misslyckades (%s); försöker sigmoid", e)

        try:
            fallback = CalibratedClassifierCV(
                estimator=clone(base_model),
                cv=2,
                method="sigmoid"
            )
            fallback.fit(X_train, y_train)
            return fallback
        except Exception as e2:
            log.error("Även sigmoid misslyckades (%s); returnerar okalibrerad modell", e2)
            safe_model = clone(base_model)
            safe_model.fit(X_train, y_train)
            return safe_model
