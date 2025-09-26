import json
import sqlite3
import pandas as pd
import pytest

from sklearn.linear_model import LogisticRegression

from export import train_full_and_export
from data_prep import make_paths


@pytest.fixture
def sample_data(tmp_path):
    """Fixture: liten testdataset med två klasser och paths för export."""
    paths = make_paths(tmp_path)

    X = pd.DataFrame({
        "feat1": [0.1, 0.3, 0.5, 0.7, 0.9, 1.1],
        "feat2": [1, 0, 1, 0, 1, 0],
    })
    y = pd.Series([0, 1, 0, 1, 0, 1], name="churned")
    ids = pd.Series(range(100, 106), name="customer_id")

    feats = X.copy()
    feats["customer_id"] = ids
    reference_date = pd.Timestamp("2024-01-01")

    return X, y, feats, reference_date, paths


def _check_export_files(paths):
    """Verifierar att alla exportfiler har skapats och innehåller rätt metadata."""
    # CSV
    assert paths.export_csv.exists(), "Export-CSV saknas"

    # SQLite
    assert paths.sqlite_db.exists(), "SQLite-fil saknas"
    with sqlite3.connect(paths.sqlite_db) as con:
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", con)
        assert "churn_scores" in tables["name"].tolist(), "Tabellen churn_scores saknas i SQLite"

    # Modell
    model_files = list(paths.model_dir.glob("final_model_*.joblib"))
    assert len(model_files) == 1, "Exakt en model-fil förväntas"

    # Metadata
    meta_files = list(paths.model_dir.glob("final_meta_*.json"))
    assert len(meta_files) == 1, "Exakt en metadata-fil förväntas"
    with open(meta_files[0], encoding="utf-8") as f:
        meta = json.load(f)
    for key in ["train_cols", "best_model_name", "sklearn_version"]:
        assert key in meta, f"Metadata saknar nyckel: {key}"


def test_train_full_and_export_creates_files(sample_data):
    """Export utan kalibrering ska skapa alla filer och kolumner."""
    X, y, feats, reference_date, paths = sample_data
    model = LogisticRegression(max_iter=200)

    feats_export = train_full_and_export(
        best_model=model,
        X=X,
        y=y,
        feats=feats,
        reference_date=reference_date,
        paths=paths,
        use_calibrated=False,
        model_name_hint="LogRegTest",
        random_state=123,
    )

    # DataFrame-kontroller
    assert "risk_score" in feats_export.columns, "risk_score saknas i export"
    assert "risk_band" in feats_export.columns, "risk_band saknas i export"
    assert set(feats_export["risk_band"].unique()).issubset(
        {"Low", "Med", "High", "Critical"}
    ), "risk_band innehåller ogiltiga värden"

    # Kontrollera filer
    _check_export_files(paths)


def test_train_full_and_export_with_calibration(sample_data):
    """Export med kalibrering ska använda isotonic eller sigmoid och skapa korrekta filer."""
    X, y, feats, reference_date, paths = sample_data
    model = LogisticRegression(max_iter=200)

    feats_export = train_full_and_export(
        best_model=model,
        X=X,
        y=y,
        feats=feats,
        reference_date=reference_date,
        paths=paths,
        use_calibrated=True,
        model_name_hint="LogRegCalib",
        random_state=123,
    )

    # Modellnamn ska indikera kalibrering (isotonic eller sigmoid)
    model_names = feats_export["model_name"].unique().tolist()
    assert any(
        "isotonic" in name.lower() or "sigmoid" in name.lower()
        for name in model_names
    ), f"Modelnamn saknar kalibreringsindikation: {model_names}"

    # Risk_scores ska ligga mellan 0–1
    assert "risk_score" in feats_export.columns, "risk_score saknas"
    assert feats_export["risk_score"].between(0, 1).all(), "risk_score ligger utanför [0, 1]"

    _check_export_files(paths)


def test_train_full_and_export_fallback_sigmoid(tmp_path):
    """Vid extremt liten dataset ska fallback till sigmoid eller okalibrerad användas."""
    paths = make_paths(tmp_path)

    X = pd.DataFrame({"feat1": [0.1, 0.9]})
    y = pd.Series([0, 1], name="churned")
    feats = X.copy()
    feats["customer_id"] = [1, 2]
    reference_date = pd.Timestamp("2024-01-01")

    model = LogisticRegression(max_iter=200)

    feats_export = train_full_and_export(
        best_model=model,
        X=X,
        y=y,
        feats=feats,
        reference_date=reference_date,
        paths=paths,
        use_calibrated=True,
        model_name_hint="LogRegFallback",
        random_state=123,
    )

    # Modellnamn ska indikera fallback (sigmoid eller okalibrerad)
    model_names = feats_export["model_name"].unique().tolist()
    assert any(
        "sigmoid" in name.lower() or "uncalibrated" in name.lower()
        for name in model_names
    ), f"Modelnamn saknar fallback-indikation: {model_names}"

    # Risk_scores ska ligga mellan 0–1
    assert "risk_score" in feats_export.columns, "risk_score saknas"
    assert feats_export["risk_score"].between(0, 1).all(), "risk_score ligger utanför [0, 1]"

    _check_export_files(paths)
