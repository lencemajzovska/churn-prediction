import logging
from pathlib import Path

from data_prep import make_paths, make_api_config, load_orders, clean_orders, build_features
from model import train_compare, select_best
from export import train_full_and_export
from log_config import setup_logging


log = logging.getLogger(__name__)


def run_pipeline(project_root: Path):
    """
    Kör hela churn-pipelinen:
    - Dataladdning
    - Feature engineering
    - Modellträning
    - Val av bästa modell
    - Export av resultat
    """
    # Skapa paths- och API-konfiguration
    paths = make_paths(project_root)
    api_cfg = make_api_config()

    # Ladda och städa orderdata (API → CSV/XLSX → dummy)
    df, source = load_orders(paths, api_cfg)
    df = clean_orders(df)
    log.info("Källdata: %s | rader=%s | kunder=%s", source, len(df), df['customer_id'].nunique())

    # Bygg features och label för modellträning
    X, y, ids, feats_model, reference_date, feature_names = build_features(
        df, churn_days=90
    )

    # Träna och jämför kandidater
    results, compare_df, proba_df, y_test, ids_test = train_compare(
        X, y, ids, random_state=42, test_size=0.25
    )

    # Välj bästa modellen baserat på AUC
    best = select_best(results)
    log.info(
        "Bästa modell: %s (AUC=%.3f, F1@0.5=%.3f, F1_best=%.3f)",
        best["name"], best["auc"], best["f1_05"], best["f1_best"]
    )


    # Träna om bästa modellen på full dataset och exportera resultat
    train_full_and_export(
        best["model"],
        X,
        y,
        feats_model,
        reference_date,
        paths,
        use_calibrated=False,
        model_name_hint=best["name"],
        random_state=42,
    )



if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    setup_logging(project_root)

    log.info("Startar churn-pipeline")
    run_pipeline(project_root)
    log.info("Pipeline klar")