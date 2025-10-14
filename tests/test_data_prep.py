"""Enhetstester för data_prep.py
(datainläsning och feature engineering)
"""

import pytest
import pandas as pd
from unittest.mock import patch

from src.data_prep import (
    load_orders,
    clean_orders,
    prepare_features,
    Paths,
    APIConfig,
    ModelConfig,
)


@pytest.fixture(scope="module")
def paths():
    return Paths.create()


@pytest.fixture(scope="module")
def api_cfg():
    return APIConfig(base_url="", api_key=None)


def test_csv_load(paths, api_cfg):
    df, source = load_orders(paths, api_cfg)

    assert isinstance(df, pd.DataFrame), "Result must be a DataFrame"
    assert source == "csv", "CSV fallback must be used when no API is available"
    assert not df.empty, "CSV file must not be empty"

    cols = [c.lower().replace(" ", "").replace("_", "") for c in df.columns]
    assert "customerid" in cols, "CustomerID column missing in raw CSV"
    assert "invoicedate" in cols, "InvoiceDate column missing in raw CSV"


def test_clean_orders(paths, api_cfg):
    df, _ = load_orders(paths, api_cfg)
    cleaned = clean_orders(df)

    assert all(c in cleaned.columns for c in ["customer_id", "invoice_date", "sales_amount"]), \
        "clean_orders must produce standardised columns"
    assert cleaned["sales_amount"].ge(0).all(), "Negative sales_amount values are not allowed"
    assert cleaned["customer_id"].notna().all(), "customer_id must not contain NaN"
    assert pd.api.types.is_datetime64_any_dtype(cleaned["invoice_date"]), "invoice_date must be datetime"


def test_prepare_features(paths, api_cfg):
    df, _ = load_orders(paths, api_cfg)
    cleaned = clean_orders(df)

    cfg = ModelConfig()

    X, y, ids, _, ref_date, feat_names = prepare_features(cleaned, cfg)

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert isinstance(ids, pd.Series)
    assert isinstance(ref_date, pd.Timestamp)
    assert not X.isna().any().any(), "Feature matrix must not contain NaN values"
    assert y.isin([0, 1]).all(), "Target variable must be binary"
    assert set(feat_names) == set(X.columns), "Feature names do not match feature matrix columns"
    assert len(X) == len(y) == len(ids), "X, y and ids must have the same number of rows"


def test_load_orders_api_mock(monkeypatch, paths):
    from src.data_prep import _fetch_orders_page

    dummy_data = [{"customer.id": 1, "total": 100, "created_at": "2011-06-15T12:00:00Z"}]

    class MockResponse:
        def json(self):
            return {"data": dummy_data}
        def raise_for_status(self):
            pass

    with patch("src.data_prep.requests.get", return_value=MockResponse()):
        api = APIConfig(base_url="http://fake.api", api_key="abc123")
        rows = _fetch_orders_page(1, api)
        assert isinstance(rows, list), "API fetch must return a list"
        assert rows[0]["total"] == 100, "API response must contain order totals"
        assert "customer.id" in rows[0], "API response must contain customer.id field"


def test_feature_columns(paths, api_cfg):
    df, _ = load_orders(paths, api_cfg)
    cleaned = clean_orders(df)
    from src.data_prep import ModelConfig

    X, *_ = prepare_features(cleaned, ModelConfig())
    expected = {"recency", "frequency_lifetime", "monetary_recent", "share_Q1", "is_weekly_buyer"}

    assert expected.issubset(X.columns), \
        f"Missing important feature columns: {expected - set(X.columns)}"
    assert all(X[c].dtype != "O" for c in X.columns), "All features must be numeric"