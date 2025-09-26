import pytest
import pandas as pd
import numpy as np

from data_prep import (
    make_paths, make_api_config, to_df,
    clean_orders, compute_rfm, build_features, Paths, APIConfig
)


@pytest.fixture
def tmp_project(tmp_path):
    """Skapar temporära kataloger för test."""
    return make_paths(tmp_path)


def test_make_paths(tmp_project):
    """Testar att make_paths skapar kataloger och paths-objektet."""
    paths = tmp_project
    assert isinstance(paths, Paths), "make_paths returnerar fel typ"
    # Kataloger ska finnas
    assert paths.data_dir.exists(), "data_dir saknas"
    assert paths.model_dir.exists(), "model_dir saknas"
    assert paths.images_dir.exists(), "images_dir saknas"
    # Viktiga attribut
    assert paths.export_csv is not None, "export_csv saknas"
    assert paths.sqlite_db is not None, "sqlite_db saknas"
    assert paths.dummy_file is not None, "dummy_file saknas"


def test_make_api_config(monkeypatch):
    """Testar att APIConfig byggs korrekt från miljövariabler."""
    monkeypatch.setenv("RUN_API", "true")
    monkeypatch.setenv("ORDERS_API_BASE", "http://fake.api")
    monkeypatch.setenv("ORDERS_API_KEY", "abc123")

    cfg = make_api_config()
    assert isinstance(cfg, APIConfig), "make_api_config returnerar fel typ"
    assert cfg.run_api is True, "run_api borde vara True"
    assert cfg.base_url == "http://fake.api"
    assert cfg.api_key == "abc123"


def test_make_api_config_defaults(monkeypatch):
    """Testar att APIConfig får defaultvärden när miljövariabler saknas."""
    monkeypatch.delenv("RUN_API", raising=False)
    monkeypatch.delenv("ORDERS_API_BASE", raising=False)
    monkeypatch.delenv("ORDERS_API_KEY", raising=False)

    cfg = make_api_config()
    assert isinstance(cfg, APIConfig)
    assert cfg.run_api is False
    assert cfg.base_url is None
    assert cfg.api_key is None


def test_to_df_normalization():
    """Testar att API-records normaliseras korrekt."""
    records = [
        {"customer.id": 1, "total": 100, "created_at": "2024-01-01T12:00:00Z"},
        {"customer.id": 2, "total": 50, "created_at": "2024-01-02T12:00:00Z"},
    ]
    df = to_df(records, "created_at")
    assert list(df.columns) == ["customer_id", "order_date", "amount"]
    assert df["customer_id"].tolist() == [1, 2]
    assert (df["amount"] >= 0).all(), "amount ska vara icke-negativ"
    assert pd.api.types.is_datetime64_any_dtype(df["order_date"]), "order_date ska vara datetime"


def test_to_df_empty():
    """Testar att to_df hanterar tom input."""
    df = to_df([], "created_at")
    assert isinstance(df, pd.DataFrame)
    assert df.empty, "DataFrame borde vara tom"
    assert list(df.columns) == ["customer_id", "order_date", "amount"]


def test_clean_orders():
    """Testar att clean_orders rensar NaN och feltyper."""
    df = pd.DataFrame({
        "customer_id": [1, 2, None],
        "order_date": ["2024-01-01", None, "2024-01-03"],
        "amount": [100, None, 50],
    })
    cleaned = clean_orders(df)
    assert "amount" in cleaned.columns, "Kolumnen 'amount' saknas"
    assert cleaned["amount"].isna().sum() == 0, "NaN borde vara rensade i amount"
    assert cleaned["customer_id"].isna().sum() == 0, "NaN borde vara rensade i customer_id"
    assert pd.api.types.is_datetime64_any_dtype(cleaned["order_date"]), "order_date borde vara datetime"
    assert len(cleaned) < len(df), "clean_orders borde droppa rader"
    assert pd.api.types.is_numeric_dtype(cleaned["amount"]), "amount borde vara numerisk"


def test_compute_rfm():
    """Testar att compute_rfm skapar RFM-score och segment."""
    df = pd.DataFrame({
        "customer_id": [1, 1, 2],
        "order_date": pd.to_datetime(["2024-01-01", "2024-01-10", "2024-01-05"]),
        "amount": [100, 200, 50],
    })
    rfm = compute_rfm(df)
    assert "R_score" in rfm.columns
    assert "F_score" in rfm.columns
    assert "M_score" in rfm.columns
    assert "segment" in rfm.columns
    assert rfm["customer_id"].nunique() == 2
    assert not rfm["segment"].isna().any(), "segment får inte vara NaN"
    for col in ["R_score", "F_score", "M_score"]:
        assert rfm[col].between(1, 5).all(), f"{col} borde ligga mellan 1–5"
    assert rfm["RFM_sum"].between(3, 15).all(), "RFM_sum borde ligga 3–15"


def test_build_features():
    """Testar att build_features skapar features och label korrekt för churn-modellen."""
    df = pd.DataFrame({
        "customer_id": [1, 1, 2, 2],
        "order_date": pd.to_datetime(["2024-01-01", "2024-04-01", "2024-02-01", "2024-03-01"]),
        "amount": [100, 200, 50, 75],
    })
    X, y, ids, feats_model, reference_date, feature_names = build_features(df, churn_days=90)

    assert isinstance(X, pd.DataFrame), "X borde vara DataFrame"
    assert isinstance(y, pd.Series), "y borde vara Series"
    assert isinstance(ids, pd.Series), "ids borde vara Series"

    assert "log_monetary_lifetime" in X.columns, "Feature saknas i X"
    assert y.isin([0, 1]).all(), "y borde vara binär"
    assert len(ids) == len(y), "ids och y borde ha samma längd"
    assert not X.isna().any().any(), "X borde inte innehålla NaN"
    assert not y.isna().any(), "y borde inte innehålla NaN"
    assert set(feature_names) == set(X.columns), "feature_names borde matcha X.columns"
    assert isinstance(reference_date, pd.Timestamp), "reference_date borde vara Timestamp"
