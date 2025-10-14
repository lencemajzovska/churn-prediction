"""
Pipeline:

- inläsning av data
- feature engineering
- modellval och utvärdering
- kalibrering
- export av resultat
"""

from __future__ import annotations
import sys
import logging
import warnings
import pandas as pd
from pathlib import Path
from log_config import setup_logging
from src.data_prep import Paths, ModelConfig, APIConfig, load_orders, clean_orders, prepare_features
from src.model import split_data, train_and_evaluate, evaluate_model, calibrate_model, feature_importance
from src.export import export_results

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

warnings.filterwarnings("ignore", category=FutureWarning)
log = logging.getLogger(__name__)


def main() -> None:
    """Kör hela churn-pipelinen steg för steg."""
    setup_logging()
    log.info("Pipeline started")

    # Konfiguration
    cfg = ModelConfig()
    api = APIConfig()
    paths = Paths.create()

    # Datainläsning
    df_raw, source = load_orders(paths, api)
    if df_raw is None or df_raw.empty:
        log.error("No input data available.")
        raise RuntimeError("No input data available.")

    log.info(
        f"Data loaded ({len(df_raw):,} rows, "
        f"{df_raw['CustomerID'].nunique() if 'CustomerID' in df_raw.columns else df_raw['customer_id'].nunique():,} customers)"
    )

    # Datastädning
    df = clean_orders(df_raw)
    if df.empty:
        log.error("Cleaned dataset is empty.")
        raise RuntimeError("Cleaned dataset is empty.")

    log.info(f"Cleaned dataset ready ({len(df):,} rows, {df['customer_id'].nunique():,} customers)")

    # Feature engineering
    X, y, ids, feats_model, reference_date, feature_names = prepare_features(df, cfg)
    if X.empty or y.empty:
        log.error("Feature matrix is empty.")
        raise RuntimeError("Feature matrix is empty.")

    log.info(f"Features generated ({len(ids):,} customers, {len(feature_names)} features)")

    # Train/Test-split
    X_train, X_test, y_train, y_test, ids_train, ids_test, feature_names = split_data(
        X, y, ids, random_state=cfg.random_state, test_size=0.25
    )
    log.info("Train/Test split completed")

    # Modellval
    best_model, best_name, cv_df = train_and_evaluate(X_train, y_train, random_state=cfg.random_state)

    for model, auc, std in zip(cv_df["model"], cv_df["auc_mean"], cv_df["auc_std"]):
        log.info(f"{model} | CV AUC = {auc:.3f} ± {std:.3f}")

    log.info(f"Model selected: {best_name}")

    # Utvärdering + kalibrering
    res_uncal = evaluate_model(best_name, best_model, X_train, y_train, X_test, y_test)
    calib_model, calib_used = calibrate_model(best_model, X_train, y_train)
    res_cal = evaluate_model(f"{best_name} (Calibrated)", calib_model, X_train, y_train, X_test, y_test)

    log.info(f"Calibration applied: {calib_used}")

    # Feature importance
    importance_df = feature_importance(
        calib_model, X_test, y_test, feature_names, perm_repeats=cfg.perm_repeats
    )

    # Resultatsammanställning
    eval_compare = (
        pd.DataFrame([res_uncal, res_cal])[["name", "auc", "f1_05", "f1_best", "best_thr", "precision_at_k"]]
        .rename(
            columns={
                "name": "Model",
                "auc": "AUC",
                "f1_05": "F1@0.50",
                "f1_best": "BestF1",
                "best_thr": "BestThr",
                "precision_at_k": "Precision@10",
            }
        )
        .round(3)
    )

    # Export av resultat
    export_results(
        best_model=best_model,
        X=X,
        y=y,
        feature_names=feature_names,
        feats_model=feats_model,
        importance_df=importance_df,
        cv_df=cv_df,
        eval_compare=eval_compare,
        riskband_summary=None,
        paths=paths,
        best_name=best_name,
        calib_used=calib_used,
        reference_date=reference_date,
        use_calibrated_for_export=False,
    )

    log.info("Export completed")
    log.info("Pipeline finished successfully")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.getLogger("pipeline").exception("Pipeline failed: %s", e)
        raise