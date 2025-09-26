import json
import logging
import sqlite3
from pathlib import Path
from typing import Union

import joblib
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV

log = logging.getLogger(__name__)


def train_full_and_export(
    best_model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    feats: pd.DataFrame,
    reference_date: pd.Timestamp,
    paths,
    use_calibrated: bool = False,
    model_name_hint: str = "best",
    random_state: int = 42
) -> pd.DataFrame:
    """
    Träna om den bästa modellen på hela datamängden och exportera resultat.

    Skapar:
      - Kundrisker som CSV och SQLite (inklusive index)
      - Modell som `.joblib`
      - Metadata som `.json`

    Parametrar
    ----------
    best_model : BaseEstimator
        En tränad sklearn-modell att använda som bas.
    X : pd.DataFrame
        Feature-matris.
    y : pd.Series
        Labels (binär churn).
    feats : pd.DataFrame
        Feature-matris med kund-ID för export.
    reference_date : pd.Timestamp
        Referensdatum för scoring.
    paths : Namespace-liknande objekt
        Innehåller sökvägar för export (export_csv, sqlite_db, model_dir).
    use_calibrated : bool, default=False
        Om sann, försök kalibrera modellen (isotonic → sigmoid → fallback).
    model_name_hint : str, default="best"
        Basnamn som används i exporterade filer.
    random_state : int, default=42
        Slumpfrö för reproducerbarhet.

    Returns
    -------
    feats_export : pd.DataFrame
        DataFrame med risk_scores, risk_band och metadatafält.
    """

    def _fit_calibrated(base_model: BaseEstimator, X, y, method: str, cv: int) -> CalibratedClassifierCV:
        """Träna kalibrerad modell med vald metod och cv."""
        m = CalibratedClassifierCV(estimator=clone(base_model), cv=cv, method=method)
        m.fit(X, y)
        return m

    # Träna om basmodellen på hela datasetet
    base_model_full = clone(best_model)
    base_model_full.fit(X, y)

    export_model: Union[BaseEstimator, CalibratedClassifierCV] = base_model_full
    export_name = model_name_hint

    if use_calibrated:
        try:
            n_samples = len(y)
            n_classes = y.nunique()
            max_cv = min(5, n_samples // n_classes)
            if max_cv < 2:
                raise ValueError(f"För få samples för CV (samples={n_samples}, classes={n_classes})")

            export_model = _fit_calibrated(base_model_full, X, y, "isotonic", max_cv)
            export_name = f"{model_name_hint}+Calibrated(isotonic)"
            log.info("Kalibrering lyckades med isotonic (cv=%d)", max_cv)

        except Exception as e:
            log.warning("Isotonic misslyckades (%s); försöker sigmoid", e)
            try:
                export_model = _fit_calibrated(base_model_full, X, y, "sigmoid", cv=2)
                export_name = f"{model_name_hint}+Calibrated(sigmoid)"
                log.info("Kalibrering lyckades med sigmoid (cv=2)")
            except Exception as e2:
                log.error("Även sigmoid misslyckades (%s); kör okalibrerad modell", e2)
                export_model = base_model_full
                export_name = f"{model_name_hint}+Uncalibrated"

    # Prediktera riskscore för alla kunder
    feats_export = feats.copy()
    feats_export["risk_score"] = export_model.predict_proba(X)[:, 1]

    # Dela in kunder i fyra risknivåer (percentiler)
    labels = ["Low", "Med", "High", "Critical"]
    pct = feats_export["risk_score"].rank(pct=True, method="first")
    feats_export["risk_band"] = pd.cut(
        pct, bins=[0, 0.25, 0.50, 0.75, 1.0],
        labels=labels, include_lowest=True, ordered=True
    )

    # Lägg till metadatafält i DataFrame
    feats_export["risk_score"] = feats_export["risk_score"].round(4)
    feats_export["model_name"] = export_name
    feats_export["model_version"] = "v1"
    feats_export["scored_at"] = pd.Timestamp.utcnow().isoformat()
    feats_export["reference_date"] = reference_date.isoformat()

    # Exportera till CSV
    feats_export.to_csv(paths.export_csv, index=False)

    # Exportera till SQLite
    with sqlite3.connect(paths.sqlite_db) as con:
        feats_export.to_sql("churn_scores", con, if_exists="replace", index=False)
        con.executescript("""
        CREATE INDEX IF NOT EXISTS ix_churn_scores_customer_id ON churn_scores(customer_id);
        CREATE INDEX IF NOT EXISTS ix_churn_scores_risk_score   ON churn_scores(risk_score);
        CREATE INDEX IF NOT EXISTS ix_churn_scores_model_name   ON churn_scores(model_name);
        """)

    # Exportera tränad modell
    model_path = paths.model_dir / f"final_model_{export_name}.joblib"
    joblib.dump(export_model, model_path)

    # Metadata
    meta = {
        "train_cols": list(X.columns),
        "created_at_utc": pd.Timestamp.utcnow().isoformat(),
        "reference_date": reference_date.isoformat(),
        "best_model_name": export_name,
        "random_state": random_state,
        "sklearn_version": __import__("sklearn").__version__,
        "xgboost_version": __import__("xgboost").__version__,
        "pandas_version": pd.__version__,
        "numpy_version": __import__("numpy").__version__,
    }

    # Standardfil som testerna förväntar sig
    meta_path = paths.model_dir / "final_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Extra fil med modellnamn (för spårbarhet)
    meta_path_named = paths.model_dir / f"final_meta_{export_name}.json"
    with open(meta_path_named, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    log.info("Modell tränad och exporterad som %s", export_name)
    return feats_export
