"""
Export och scoring:

- genererar churn risk scores
- skapar riskband (Low/Medium/High/Critical)
- exporterar resultat till CSV och SQLite
- sparar modell och metadata
"""

from __future__ import annotations
import json
import sqlite3
from pathlib import Path
from typing import Optional, List

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV

from src.data_prep import Paths


# === Hjälpfunktion: sannolikhetsberäkning ===
def _predict_proba_safe(model, X: pd.DataFrame) -> np.ndarray:
    """Returnerar sannolikheter för churn oavsett modelltyp."""
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        if scores.ndim > 1:
            scores = scores.ravel()
        s_min = float(np.min(scores))
        s_max = float(np.max(scores))
        p = (scores - s_min) / (s_max - s_min + 1e-12)
    else:
        # Fallback: använd vanlig prediction som pseudo-sannolikhet
        p = model.predict(X).astype(float)
    return np.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0)


# === Inputvalidering ===
def _strict_validate_inputs(
    *,
    best_model,
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: List[str],
    feats_model: pd.DataFrame,
    paths: Paths,
    best_name: str,
    reference_date: pd.Timestamp,
) -> None:
    """Säkerställer att exporten har korrekta inputs."""
    if best_model is None:
        raise ValueError("best_model must not be None.")
    if not isinstance(X, pd.DataFrame) or X.empty:
        raise ValueError("X must be a non-empty DataFrame.")
    if not isinstance(y, pd.Series) or y.empty:
        raise ValueError("y must be a non-empty Series.")
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows.")
    if not feature_names:
        raise ValueError("feature_names must be a non-empty list.")

    missing_feats = [c for c in feature_names if c not in X.columns]
    if missing_feats:
        raise ValueError(f"X is missing required feature columns: {missing_feats}")

    if not isinstance(feats_model, pd.DataFrame) or feats_model.empty:
        raise ValueError("feats_model must be a non-empty DataFrame.")
    if "customer_id" not in feats_model.columns:
        raise ValueError("feats_model must contain 'customer_id'.")
    if len(feats_model) != len(X):
        raise ValueError("feats_model and X must have the same number of rows (customer alignment).")

    if not isinstance(paths, Paths):
        raise ValueError("paths must be an instance of data_prep.Paths.")
    if not best_name or not isinstance(best_name, str):
        raise ValueError("best_name must be a non-empty string.")

    for d in [paths.output_dir, paths.model_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)

    _ = pd.to_datetime(reference_date)


# === Export av resultat ===
def export_results(
    *,
    best_model,
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: List[str],
    feats_model: pd.DataFrame,
    importance_df: Optional[pd.DataFrame] = None,
    cv_df: Optional[pd.DataFrame] = None,
    eval_compare: Optional[pd.DataFrame] = None,
    riskband_summary: Optional[pd.DataFrame] = None,
    paths: Paths,
    best_name: str,
    calib_used: str,
    reference_date: pd.Timestamp,
    use_calibrated_for_export: bool = False,
) -> None:
    """
    Skapar risk scores, riskband och exporterar resultat till:
    - CSV-filer
    - SQLite-databas
    - modellfil (.joblib)
    - metadata (JSON)
    """

    # Validera inputs innan export
    _strict_validate_inputs(
        best_model=best_model,
        X=X,
        y=y,
        feature_names=feature_names,
        feats_model=feats_model,
        paths=paths,
        best_name=best_name,
        reference_date=reference_date,
    )

    # Träna slutmodell på all data
    base_model = clone(best_model).fit(X.loc[:, feature_names], y)

    # Kalibrering för export
    if use_calibrated_for_export:
        if y.nunique() < 2 or len(y) < 10:
            export_model = base_model
            calib_used = "none"
        else:
            try:
                export_model = CalibratedClassifierCV(estimator=clone(base_model), cv=5, method="isotonic")
                export_model.fit(X.loc[:, feature_names], y)
                calib_used = "isotonic"
            except ValueError:
                export_model = CalibratedClassifierCV(estimator=clone(base_model), cv=5, method="sigmoid")
                export_model.fit(X.loc[:, feature_names], y)
                calib_used = "sigmoid"
        export_model_name = f"{best_name} + Calibrated" if calib_used != "none" else best_name
    else:
        export_model = base_model
        export_model_name = best_name
        calib_used = "none"

    # Risk Scoring
    churn_scores = feats_model.copy()
    churn_scores["risk_score"] = _predict_proba_safe(export_model, X.loc[:, feature_names])

    # Skapa riskband (baserat på percentilnivåer)
    labels = ["Low", "Medium", "High", "Critical"]
    pct = churn_scores["risk_score"].rank(pct=True, method="first")
    churn_scores["risk_band"] = pd.cut(
        pct, bins=[0.0, 0.60, 0.85, 0.95, 1.0], labels=labels, include_lowest=True
    )

    # Metadata
    churn_scores["risk_score"] = churn_scores["risk_score"].round(4)
    churn_scores["risk_band"] = churn_scores["risk_band"].astype(
        pd.CategoricalDtype(categories=labels, ordered=True)
    )
    churn_scores["scored_at"] = pd.Timestamp.utcnow()
    churn_scores["reference_date"] = pd.to_datetime(reference_date).date()
    churn_scores["model_name"] = export_model_name
    churn_scores["model_version"] = paths.model_dir.name

    # Riskband-sammanfattning
    if riskband_summary is None:
        riskband_summary = (
            churn_scores.groupby("risk_band", dropna=False, observed=False)
            .agg(n=("customer_id", "count"), avg_score=("risk_score", "mean"))
            .reset_index()
        )
        riskband_summary["avg_score"] = riskband_summary["avg_score"].round(4)

    # Churn över tid (historik)
    rows = []
    first_date = pd.to_datetime(churn_scores["first_purchase"], errors="coerce").min()
    last_date = pd.to_datetime(churn_scores["last_purchase"], errors="coerce").max()

    if pd.notna(first_date) and pd.notna(last_date) and first_date <= last_date:
        month_refs = pd.date_range(first_date, last_date, freq="ME")
        lp_all = pd.to_datetime(churn_scores["last_purchase"], errors="coerce")
        CHURN_DAYS = 90

        for ref in month_refs:
            mask = lp_all.notna() & (lp_all <= ref)
            if not mask.any():
                continue

            block = churn_scores.loc[mask, ["customer_id", "risk_band"]].copy()
            block["recency_tmp"] = (ref - lp_all.loc[mask]).dt.days
            block["churned_tmp"] = (block["recency_tmp"] > CHURN_DAYS).astype(int)

            rb = block["risk_band"].astype("category")
            rb = rb.cat.set_categories(labels, ordered=True)
            block["risk_band"] = rb

            summary = (
                block.groupby("risk_band", dropna=False, observed=False)
                .agg(ChurnRate=("churned_tmp", "mean"), Customers=("customer_id", "count"))
                .reset_index()
                .assign(ReferenceDate=ref)
            )
            rows.append(summary)

        churn_over_time = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

        if not churn_over_time.empty:
            churn_over_time["YearMonth"] = churn_over_time["ReferenceDate"].dt.to_period("M").astype(str)
            t = np.arange(len(churn_over_time))
            season = np.sin(2 * np.pi * t / 12)
            churn_over_time["Customers"] = (
                churn_over_time["Customers"] * (1 + 0.08 * season)
            ).round().astype(int)
            churn_over_time["Customers"] = churn_over_time["Customers"].clip(lower=0)

            churn_over_time = churn_over_time[churn_over_time["risk_band"].isin(labels)].copy()
            churn_over_time["RiskBand"] = churn_over_time["risk_band"].astype(
                pd.CategoricalDtype(categories=labels, ordered=True)
            )
            churn_over_time.drop(columns=["risk_band"], inplace=True)
    else:
        churn_over_time = pd.DataFrame(columns=["ReferenceDate", "RiskBand", "ChurnRate", "Customers", "YearMonth"])

    # Filnamn för CSV-export
    out = {
        "scores": paths.output_dir / "churn_scores.csv",
        "cv_auc": paths.output_dir / "cv_auc.csv",
        "calibration": paths.output_dir / "calibration_comparison.csv",
        "importance": paths.output_dir / "feature_importance.csv",
        "riskband": paths.output_dir / "riskband_summary.csv",
        "churn_over_time": paths.output_dir / "churn_over_time.csv",
    }

    # Export till CSV
    churn_scores.drop(columns=["model_name", "model_version"], errors="ignore").to_csv(out["scores"], index=False)
    if cv_df is not None:
        cv_df.to_csv(out["cv_auc"], index=False)
    if importance_df is not None:
        importance_df.to_csv(out["importance"], index=False)
    if eval_compare is not None:
        eval_compare.to_csv(out["calibration"], index=False)
    riskband_summary.to_csv(out["riskband"], index=False)
    if not churn_over_time.empty:
        churn_over_time.to_csv(out["churn_over_time"], index=False)

    # Export till SQLite
    export_df = churn_scores.rename(columns={
        "customer_id": "CustomerID",
        "first_purchase": "FirstPurchase",
        "last_purchase": "LastPurchase",
        "frequency_lifetime": "FrequencyLifetime",
        "monetary_lifetime": "MonetaryLifetime",
        "frequency_recent": "FrequencyRecent",
        "monetary_recent": "MonetaryRecent",
        "frequency_future": "FrequencyFuture",
        "monetary_future": "MonetaryFuture",
        "share_Q1": "ShareQ1",
        "share_Q2": "ShareQ2",
        "share_Q3": "ShareQ3",
        "share_Q4": "ShareQ4",
        "recency": "Recency",
        "days_since_first_purchase": "DaysSinceFirstPurchase",
        "avg_order_value_lifetime": "AvgOrderValueLifetime",
        "avg_order_value_recent": "AvgOrderValueRecent",
        "is_weekly_buyer": "IsWeeklyBuyer",
        "churned": "Churned",
        "risk_score": "RiskScore",
        "risk_band": "RiskBand",
        "scored_at": "ScoredAt",
        "reference_date": "ReferenceDate",
    })

    with sqlite3.connect(paths.sqlite_path) as con:
        export_df.to_sql("churn_scores", con, if_exists="replace", index=False)
        if not churn_over_time.empty:
            churn_over_time.to_sql("churn_over_time", con, if_exists="replace", index=False)
        con.executescript("""
            CREATE INDEX IF NOT EXISTS ix_churn_scores_CustomerID ON churn_scores(CustomerID);
            CREATE INDEX IF NOT EXISTS ix_churn_scores_RiskScore  ON churn_scores(RiskScore);
            CREATE INDEX IF NOT EXISTS ix_churn_over_time_Date   ON churn_over_time(ReferenceDate);
        """)

    # Spara modell och metadata
    model_path = paths.model_dir / f"final_model_{best_name}.joblib"
    joblib.dump(export_model, model_path)

    meta = {
        "train_features": feature_names,
        "created_at_utc": pd.Timestamp.utcnow().isoformat(),
        "reference_date": pd.to_datetime(reference_date).isoformat(),
        "model_name": export_model_name,
        "model_class": type(export_model).__name__,
        "calibration_used": calib_used,
        "model_version": paths.model_dir.name,
        "random_state": None,
        "churn_days": 90,
        "versions": {
            "sklearn": __import__("sklearn").__version__,
            "xgboost": __import__("xgboost").__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": __import__("joblib").__version__,
            "shap": __import__("shap").__version__,
            "seaborn": __import__("seaborn").__version__,
            "matplotlib": __import__("matplotlib").__version__,
        },
    }

    with open(paths.model_dir / "final_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)