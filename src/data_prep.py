"""
Databearbetning:

- inläsning av orderdata
- rengöring av rådata
- feature engineering (RFM)
- anti-leakage
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import requests


log = logging.getLogger(__name__)


# Beräknar säsongsbeteende per kund
SEASONS = {
    1: "Q1", 2: "Q1", 3: "Q1",
    4: "Q2", 5: "Q2", 6: "Q2",
    7: "Q3", 8: "Q3", 9: "Q3",
    10: "Q4", 11: "Q4", 12: "Q4"
}

# Kolumner som ska bort vid anti-leakage
LEAK = [
    "frequency_churn",
    "monetary_churn",
    "avg_order_value_churn",
    "target",
    "label",
]


# === Konfiguration ===
@dataclass(frozen=True)
class Paths:
    project_root: Path
    input_dir: Path
    output_dir: Path
    model_dir: Path
    images_dir: Path
    sqlite_path: Path

    @classmethod
    def create(cls) -> "Paths":
        root = Path(__file__).resolve().parent.parent
        dirs = {
            "input_dir": root / "data_input",
            "output_dir": root / "data_output",
            "model_dir": root / "models",
            "images_dir": root / "images",
        }
        for p in dirs.values():
            p.mkdir(parents=True, exist_ok=True)
        return cls(
            project_root=root,
            input_dir=dirs["input_dir"],
            output_dir=dirs["output_dir"],
            model_dir=dirs["model_dir"],
            images_dir=dirs["images_dir"],
            sqlite_path=root / "churn.db",
        )


@dataclass(frozen=True)
class ModelConfig:
    random_state: int = 42
    churn_days: int = 90
    frequency_threshold: int = 5
    perm_repeats: int = 5


@dataclass(frozen=True)
class APIConfig:
    base_url: str = os.getenv("ORDERS_API_BASE", "")
    api_key: str | None = os.getenv("ORDERS_API_KEY")
    api_path: str = "/v1/orders"
    page_size: int = 5000
    max_pages: int = 10_000
    date_field: str = "created_at"
    from_date: str | None = os.getenv("ORDERS_FROM_DATE")
    to_date: str | None = os.getenv("ORDERS_TO_DATE")

    @property
    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    @property
    def run_api(self) -> bool:
        return bool(self.base_url and self.api_key)


# === API-funktioner ===
def _fetch_orders_page(page: int, api: APIConfig) -> list[dict]:
    """Hämtar en sida med orderdata från API:et."""
    params = {"limit": api.page_size, "page": page}
    if api.from_date:
        params["from"] = api.from_date
    if api.to_date:
        params["to"] = api.to_date

    resp = requests.get(
            f"{api.base_url}{api.api_path}",
            headers=api.headers,
            params=params,
            timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    if isinstance(data, dict) and "data" in data:
        return data["data"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unexpected API response format: {type(data)}")


def _api_records_to_df(records: list[dict], api: APIConfig) -> pd.DataFrame:
    """Konverterar API-respons till DataFrame i standardiserat format."""
    if not records:
        return pd.DataFrame(columns=["customer_id", "invoice_date", "sales_amount"])

    raw = pd.json_normalize(records)
    mapping = {
        "customer.id": "customer_id",
        "total": "amount",
        api.date_field: "order_date",
    }
    for src, dst in mapping.items():
        if src in raw.columns:
            raw[dst] = raw[src]

    df = pd.DataFrame(
        {
            "customer_id": pd.to_numeric(raw.get("customer_id", pd.Series(dtype="float")), errors="coerce"),
            "order_date": pd.to_datetime(raw.get("order_date", pd.Series(dtype="object")), errors="coerce", utc=True)
            .dt.tz_localize(None),
            "amount": pd.to_numeric(raw.get("amount", pd.Series(dtype="float")), errors="coerce"),
        }
    ).dropna(subset=["customer_id", "order_date"])

    df["amount"] = df["amount"].fillna(0).clip(lower=0)
    df.rename(columns={"order_date": "invoice_date"}, inplace=True)
    df["sales_amount"] = df.pop("amount")
    df["customer_id"] = df["customer_id"].astype("Int64")
    return df.reset_index(drop=True)


# === Datainläsning ===
def load_orders(paths: Paths, api: APIConfig) -> tuple[pd.DataFrame, str]:
    """
    Läser in orderdata:
    1. API (om aktiverat)
    2. Annars CSV
    3. Annars XLSX
    """

    # Försök via API
    if api.run_api:
        all_rows = []
        for page in range(1, api.max_pages + 1):
            try:
                rows = _fetch_orders_page(page, api)
                if not rows:
                    break
                all_rows.extend(rows)
                if len(rows) < api.page_size:
                    break
            except Exception:
                break

        if all_rows:
            df_api = _api_records_to_df(all_rows, api)
            csv_path = paths.input_dir / "orders_api_export.csv"
            df_api.to_csv(csv_path, index=False)
            return df_api, "api"

    # CSV fallback
    csv_files = sorted(paths.input_dir.glob("*.csv"))
    if csv_files:
        df_csv = pd.read_csv(csv_files[0], encoding="ISO-8859-1")
        return df_csv, "csv"

    # XLSX fallback
    xlsx_files = sorted(paths.input_dir.glob("*.xlsx"))
    if xlsx_files:
        df_xlsx = pd.read_excel(xlsx_files[0])
        return df_xlsx, "xlsx"

    raise FileNotFoundError("No input data found (API, CSV, or XLSX).")


# === Städning ===
def clean_orders(df: pd.DataFrame) -> pd.DataFrame:
    """Rensar och standardiserar orderdata."""
    df = df.copy()
    rename_map = {
        "InvoiceNo": "invoice_no",
        "StockCode": "stock_code",
        "Description": "description",
        "Quantity": "quantity",
        "InvoiceDate": "invoice_date",
        "UnitPrice": "unit_price",
        "CustomerID": "customer_id",
        "Country": "country",
    }
    df.rename(columns=rename_map, inplace=True)
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")

    # Beräkna sales_amount om kolumnen saknas
    if "sales_amount" not in df.columns:
        df["sales_amount"] = df.get("unit_price", 0) * df.get("quantity", 0)

    df["sales_amount"] = pd.to_numeric(df["sales_amount"], errors="coerce").fillna(0)
    df = df[df["sales_amount"] >= 0]
    df["customer_id"] = pd.to_numeric(df["customer_id"], errors="coerce").astype("Int64")
    return df.dropna(subset=["customer_id", "invoice_date"])


# === Hjälpfunktioner för feature engineering ===
def _clean_features(X: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Tar bort NaN-värden från featurematrisen."""
    if int(X.isna().sum().sum()) > 0:
        X = X.fillna(0)
    assert not X.isna().any().any(), "NaN finns kvar i X efter rensning!"
    return X


def _rfm_scores_safe(series: pd.Series, labels: list[int]) -> pd.Series:
    """
    RFM-poängsättning med qcut.
    Faller tillbaka till medianklass vid få unika värden.
    """
    try:
        return pd.qcut(series, 5, labels=labels, duplicates="drop").astype(int)
    except Exception:
        return pd.Series(np.full(len(series), int(np.median(labels))), index=series.index)


def _build_future(df_future: pd.DataFrame) -> pd.DataFrame:
    """Beräknar framtida köpaktivitet per kund."""
    return (
        df_future.groupby("customer_id")
        .agg(frequency_future=("invoice_date", "count"),
             monetary_future=("sales_amount", "sum"))
        .reset_index()
    )


def _build_rfm_hist(df_hist: pd.DataFrame, cutoff_date: pd.Timestamp) -> pd.DataFrame:
    """Beräknar historisk RFM-score och segment per kund."""
    rfm = (
        df_hist.groupby("customer_id")
        .agg(last_purchase=("invoice_date", "max"),
             frequency=("invoice_date", "count"),
             monetary=("sales_amount", "sum"))
        .reset_index()
    )

    rfm["recency"] = (cutoff_date - rfm["last_purchase"]).dt.days
    rfm["R_score"] = _rfm_scores_safe(rfm["recency"], [5, 4, 3, 2, 1])
    rfm["F_score"] = _rfm_scores_safe(rfm["frequency"], [1, 2, 3, 4, 5])
    rfm["M_score"] = _rfm_scores_safe(rfm["monetary"], [1, 2, 3, 4, 5])
    rfm["RFM_sum"] = rfm[["R_score", "F_score", "M_score"]].sum(axis=1)

    def _segment_row(row) -> str:
        if row["RFM_sum"] >= 12:
            return "Loyal"
        if row["R_score"] >= 4 and row["F_score"] >= 3:
            return "Growth"
        if row["R_score"] <= 2 and row["F_score"] <= 2:
            return "Inactive"
        return "Standard"

    rfm["segment"] = rfm.apply(_segment_row, axis=1)
    return rfm


def _build_recent(df_hist: pd.DataFrame, lookback_date: pd.Timestamp) -> pd.DataFrame:
    """Beräknar köpaktivitet i närtid per kund."""
    recent = df_hist[df_hist["invoice_date"] >= lookback_date]
    return (
        recent.groupby("customer_id")
        .agg(frequency_recent=("invoice_date", "count"), monetary_recent=("sales_amount", "sum"))
        .reset_index()
    )


def _build_lifetime(df_hist: pd.DataFrame) -> pd.DataFrame:
    """Beräknar livstidsvärden per kund."""
    return (
        df_hist.groupby("customer_id")
        .agg(first_purchase=("invoice_date", "min"),
             last_purchase=("invoice_date", "max"),
             frequency_lifetime=("invoice_date", "count"),
             monetary_lifetime=("sales_amount", "sum"))
        .reset_index()
    )


def _build_season_share(df_hist: pd.DataFrame) -> pd.DataFrame:
    """Beräknar köpandel per säsong (kvartal) för varje kund."""
    df = df_hist.copy()
    df["season"] = df["invoice_date"].dt.month.map(SEASONS)
    season = (
        df.assign(count=1)
        .pivot_table(index="customer_id", columns="season", values="count", aggfunc="sum", fill_value=0)
        .pipe(lambda t: t.div(t.sum(axis=1).replace(0, np.nan), axis=0))
        .add_prefix("share_")
        .reset_index()
    )

    # Säkerställ att alla kvartal finns
    for c in ["share_Q1", "share_Q2", "share_Q3", "share_Q4"]:
        if c not in season.columns:
            season[c] = 0.0

    return season


# === Feature engineering + featurematris ===
def prepare_features(
    df: pd.DataFrame,
    cfg: ModelConfig,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Timestamp, List[str]]:
    """
    Bygger featurematris baserat på RFM-logik.
    Funktionen separerar historiska och framtida transaktioner, skapar features
    och hanterar anti-leakage.
    """
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty. Cannot build features.")

    # Säkerställ att datumkolumnen är korrekt typ
    if not np.issubdtype(df["invoice_date"].dtype, np.datetime64):
        raise ValueError("Column 'invoice_date' must be datetime. Run clean_orders(df) first.")

    # Skapa cutoff-datum (separerar historik från framtid)
    reference_date = df["invoice_date"].max() + pd.Timedelta(days=1)
    cutoff_date = reference_date - pd.Timedelta(days=cfg.churn_days)

    df_hist = df[df["invoice_date"] <= cutoff_date].copy()
    df_future = df[df["invoice_date"] > cutoff_date].copy()

    # Bygg deltabeller
    rfm_future = _build_future(df_future)
    rfm_hist = _build_rfm_hist(df_hist, cutoff_date)
    rfm_life = _build_lifetime(df_hist)
    rfm_recent = _build_recent(df_hist, cutoff_date - pd.Timedelta(days=cfg.churn_days))
    season_share = _build_season_share(df_hist)

    # Slå ihop featuretabell
    feats = (
        rfm_life
        .merge(rfm_recent, on="customer_id", how="left", validate="1:1")
        .merge(season_share, on="customer_id", how="left", validate="1:1")
        .merge(rfm_future, on="customer_id", how="left", validate="1:1")
    )

    # Härledda features
    feats["recency"] = (cutoff_date - feats["last_purchase"]).dt.days
    feats["days_since_first_purchase"] = (cutoff_date - feats["first_purchase"]).dt.days

    feats["avg_order_value_lifetime"] = (
        feats["monetary_lifetime"] / feats["frequency_lifetime"].replace(0, np.nan)
    ).fillna(0)

    feats["avg_order_value_recent"] = (
        feats["monetary_recent"] / feats["frequency_recent"].replace(0, np.nan)
    ).fillna(0)

    feats["is_weekly_buyer"] = (feats["frequency_recent"].fillna(0) >= (cfg.churn_days // 7)).astype(int)

    # Fyll saknade numeriska fält
    num_cols = feats.select_dtypes(include=[np.number]).columns
    feats[num_cols] = feats[num_cols].fillna(0)

    # Skapa churnindikatorer
    rfm_hist = rfm_hist.merge(
        rfm_future[["customer_id", "frequency_future"]],
        on="customer_id",
        how="left"
    ).fillna({"frequency_future": 0})

    rfm_hist["churned_future"] = (rfm_hist["frequency_future"] == 0).astype(int)
    rfm_hist["churned_past"] = np.where(
        (rfm_hist["recency"] > cfg.churn_days) &
        (rfm_hist["frequency"] < cfg.frequency_threshold),
        1,
        0
    )
    feats["churned"] = (feats["frequency_future"] == 0).astype(int)

    # Behåll relevanta RFM-fält
    feats_model = feats.merge(
        rfm_hist[
            [
                "customer_id",
                "recency",
                "frequency",
                "monetary",
                "R_score",
                "F_score",
                "M_score",
                "RFM_sum",
                "segment",
                "frequency_future",
                "churned_future",
                "churned_past",
            ]
        ],
        on="customer_id",
        how="left",
        suffixes=("", "_rfm"),
    )

    # Bygg featurematris
    base_cols = ["recency", "frequency_lifetime", "monetary_lifetime"]
    extra_cols = [
        "avg_order_value_lifetime",
        "avg_order_value_recent",
        "days_since_first_purchase",
        "frequency_recent",
        "monetary_recent",
        "is_weekly_buyer",
        "share_Q1",
        "share_Q2",
        "share_Q3",
        "share_Q4",
    ]
    features = feats_model[["customer_id"] + base_cols + extra_cols].copy()

    # Log-transformera monetärt värde (för att minska spridning)
    features["log_monetary_lifetime"] = np.log1p(
        features["monetary_lifetime"].clip(lower=0)
    )
    features.drop(columns=["monetary_lifetime"], inplace=True)

    X = features.drop(columns=["customer_id"]).astype(float)
    y = feats_model["churned"].astype(int)
    ids = features["customer_id"].astype(int)

    # Anti-leakage (ta bort kolumner som avslöjar framtidsinformation)
    suspect = [c for c in X.columns if any(term in c.lower() for term in ["future", "target", "label"])]
    to_drop = [c for c in set(LEAK + suspect) if c in X.columns]
    if to_drop:
        log.debug("Removing possible leakage columns: %s", to_drop)
        X.drop(columns=to_drop, inplace=True)

    # Rengör matris
    X = _clean_features(X, verbose=False)
    feature_names = X.columns.tolist()

    return X, y, ids, feats_model, reference_date, feature_names