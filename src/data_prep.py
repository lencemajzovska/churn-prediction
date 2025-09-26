from __future__ import annotations
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore", category=FutureWarning)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Paths:
    """
    Håller ordnade filvägar för projektet.
    """
    project_root: Path
    data_dir: Path
    images_dir: Path
    model_dir: Path
    sqlite_db: Path
    export_csv: Path
    dummy_file: Path


@dataclass(frozen=True)
class APIConfig:
    """
    Konfiguration för att hämta orderdata från ett API.
    """
    run_api: bool
    base_url: str
    path: str
    api_key: str | None
    page_size: int = 1000
    max_pages: int = 10_000
    date_field: str = "created_at"
    from_date: str | None = None
    to_date: str | None = None


@dataclass(frozen=True)
class PrepConfig:
    """
    Konfiguration för churn-modellen.
    """
    churn_days: int = 90
    random_state: int = 42


def make_paths(project_root: Path) -> Paths:
    """
    Skapar kataloger och returnerar ett Paths-objekt
    med alla centrala filvägar.
    """
    data_dir = project_root / "data"; data_dir.mkdir(exist_ok=True)
    images_dir = project_root / "images"; images_dir.mkdir(exist_ok=True)
    model_dir = project_root / "models"; model_dir.mkdir(exist_ok=True)
    log.info("Paths skapade under %s", project_root)
    return Paths(
        project_root=project_root,
        data_dir=data_dir,
        images_dir=images_dir,
        model_dir=model_dir,
        sqlite_db=project_root / "churn.db",
        export_csv=data_dir / "churn_predictions.csv",
        dummy_file=data_dir / "dummy_orders.csv",
    )


def make_api_config(date_field: str = "created_at") -> APIConfig:
    """
    Skapar en APIConfig baserad på miljövariabler.
    """
    api_cfg = APIConfig(
        run_api=os.getenv("RUN_API", "false").lower() == "true",
        base_url=os.getenv("ORDERS_API_BASE"),
        path="/v1/orders",
        api_key=os.getenv("ORDERS_API_KEY"),
        page_size=int(os.getenv("ORDERS_PAGE_SIZE", 1000)),
        max_pages=int(os.getenv("ORDERS_MAX_PAGES", 10_000)),
        date_field=date_field,
        from_date=os.getenv("ORDERS_FROM_DATE"),
        to_date=os.getenv("ORDERS_TO_DATE"),
    )
    log.info("APIConfig skapad (run_api=%s, base_url=%s)", api_cfg.run_api, api_cfg.base_url)
    return api_cfg


def _fetch_orders_page(api: APIConfig, page: int) -> list[dict]:
    """
    Hämtar en sida med orderdata från API:t.
    """
    headers = {"Authorization": f"Bearer {api.api_key}"} if api.api_key else {}
    params = {"limit": api.page_size, "page": page}
    if api.from_date: params["from"] = api.from_date
    if api.to_date:   params["to"] = api.to_date

    resp = requests.get(f"{api.base_url}{api.path}", headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data["data"] if isinstance(data, dict) and "data" in data else data


def to_df(records: list[dict], date_field: str) -> pd.DataFrame:
    """
    Normaliserar API-respons till DataFrame med standardkolumner.
    """
    if not records:
        return pd.DataFrame(columns=["customer_id", "order_date", "amount"])

    raw = pd.json_normalize(records)

    # Mappa API-fält till standardnamn
    mapping = {"customer.id": "customer_id", "total": "amount", date_field: "order_date"}
    for src, dst in mapping.items():
        if src in raw.columns:
            raw[dst] = raw[src]

    df = pd.DataFrame({
        "customer_id": pd.to_numeric(raw.get("customer_id", pd.Series(dtype="float")), errors="coerce"),
        "order_date":  pd.to_datetime(
            raw.get("order_date", pd.Series(dtype="object")), errors="coerce", utc=True
        ).dt.tz_localize(None),
        "amount": pd.to_numeric(raw.get("amount", pd.Series(dtype="float")), errors="coerce"),
    })

    df = df.dropna(subset=["customer_id", "order_date"])
    df["amount"] = df["amount"].fillna(0).clip(lower=0)
    return df


def load_orders(paths: Paths, api: APIConfig) -> tuple[pd.DataFrame, str]:
    """
    Laddar orderdata.
    Försöker i turordning:
    1) API
    2) Lokal CSV/XLSX
    3) Dummy-data
    """
    # API
    if api.run_api and api.base_url and api.api_key:
        log.info("Försöker hämta ordrar från API...")
        all_rows: list[dict] = []
        for page in range(1, api.max_pages + 1):
            tries = 0
            while True:
                try:
                    rows = _fetch_orders_page(api, page)
                    break
                except requests.HTTPError as e:
                    tries += 1
                    if tries > 3:
                        log.error("HTTP-fel på sida %s efter %s försök: %s", page, tries, e)
                        raise
                    wait = 2 ** tries
                    log.warning("HTTP-fel på sida %s: %s. Försöker igen om %ss", page, e, wait)
                    time.sleep(wait)
                except Exception as e:
                    log.exception("Avbryter API-hämtning: %s", e)
                    rows = []
                    break
            if not rows:
                break
            all_rows.extend(rows)
            if len(rows) < api.page_size:
                break

        if all_rows:
            df_api = to_df(all_rows, api.date_field)
            out_csv = paths.data_dir / "orders.csv"
            df_api.to_csv(out_csv, index=False)
            log.info("API-hämtning klar (%s rader). Sparat till %s", len(df_api), out_csv)
            return df_api, "api"

    # CSV/XLSX
    csv_path = paths.data_dir / "orders.csv"
    xlsx_path = paths.data_dir / "orders.xlsx"
    if csv_path.exists():
        log.info("Laddar ordrar från CSV (%s)", csv_path)
        return pd.read_csv(csv_path), "csv"
    if xlsx_path.exists():
        log.info("Laddar ordrar från Excel (%s)", xlsx_path)
        return pd.read_excel(xlsx_path), "xlsx"

    # Dummy
    log.warning("Ingen källa hittad, genererar dummy-data.")
    n_customers = 500
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    df = pd.DataFrame({
        "customer_id": np.random.randint(1, n_customers + 1, size=5000),
        "order_date": np.random.choice(dates, size=5000),
        "amount": np.random.randint(50, 1500, size=5000),
    })
    df["order_date"] = pd.to_datetime(df["order_date"]) + pd.to_timedelta(
        np.random.randint(-30, 30, size=len(df)), unit="D"
    )
    df["amount"] = (df["amount"] * np.random.normal(1.0, 0.2, size=len(df))).clip(lower=10)

    df.to_csv(paths.dummy_file, index=False)
    log.info("Dummy-data skapad (%s rader). Sparad till %s", len(df), paths.dummy_file)

    if api.run_api:
        raise RuntimeError("Dummy-data får inte användas i drift!")

    return df, "dummy"


def clean_orders(df: pd.DataFrame) -> pd.DataFrame:
    """
    Grundläggande datarensning.
    """
    df = df.copy()
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["customer_id", "order_date"])
    df["amount"] = df["amount"].fillna(0)
    log.info("Data rensad: %s rader kvar", len(df))
    return df


def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Beräknar RFM-segment (Recency, Frequency, Monetary).
    """
    reference_date = df["order_date"].max() + pd.Timedelta(days=1)
    rfm = (df.groupby("customer_id")
             .agg(last_purchase=("order_date", "max"),
                  frequency=("order_date", "count"),
                  monetary=("amount", "sum"))
             .reset_index())

    rfm["recency"] = (reference_date - rfm["last_purchase"]).dt.days
    rfm["R_score"] = pd.qcut(rfm["recency"], 5, labels=[5,4,3,2,1], duplicates="drop").astype(int)
    rfm["F_score"] = pd.qcut(rfm["frequency"], 5, labels=[1,2,3,4,5], duplicates="drop").astype(int)
    rfm["M_score"] = pd.qcut(rfm["monetary"], 5, labels=[1,2,3,4,5], duplicates="drop").astype(int)
    rfm["RFM_sum"] = rfm[["R_score","F_score","M_score"]].sum(axis=1)

    def segment_row(row):
        if row["RFM_sum"] >= 12: return "Loyal/VIP"
        if row["R_score"] >= 4 and row["F_score"] >= 3: return "Growth"
        if row["R_score"] <= 2 and row["F_score"] <= 2: return "At Risk/Inactive"
        return "Standard"

    rfm["segment"] = rfm.apply(segment_row, axis=1)
    log.info("RFM-analys klar för %s kunder", len(rfm))
    return rfm


def build_features(df: pd.DataFrame, churn_days: int) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Timestamp, list[str]]:
    """
    Bygger features och label för churn-modellen.
    Returnerar X, y, ids, feats_model, reference_date, feature_names.
    """
    reference_date = df["order_date"].max() + pd.Timedelta(days=1)

    rfm_base = (df.groupby("customer_id")
                  .agg(first_purchase=("order_date", "min"),
                       last_purchase=("order_date", "max"),
                       orders_lifetime=("order_date", "count"),
                       monetary_lifetime=("amount", "sum"))
                  .reset_index())

    look_90 = reference_date - pd.Timedelta(days=90)
    df_90 = df[df["order_date"] >= look_90]
    win_90 = (df_90.groupby("customer_id")
                 .agg(frequency_90d=("order_date", "count"),
                      monetary_90d=("amount", "sum"))
                 .reset_index())

    seasons = {1:"Q1",2:"Q1",3:"Q1",4:"Q2",5:"Q2",6:"Q2",7:"Q3",8:"Q3",9:"Q3",10:"Q4",11:"Q4",12:"Q4"}
    df = df.copy()
    df["season"] = df["order_date"].dt.month.map(seasons)
    season_share = (df.assign(cnt=1)
                      .pivot_table(index="customer_id", columns="season", values="cnt", aggfunc="sum", fill_value=0)
                      .pipe(lambda t: t.div(t.sum(1).replace(0, np.nan), axis=0))
                      .add_prefix("share_")
                      .reset_index())

    feats = (rfm_base
             .merge(win_90, on="customer_id", how="left")
             .merge(season_share, on="customer_id", how="left"))

    feats["recency"] = (reference_date - feats["last_purchase"]).dt.days
    feats["days_since_first_purchase"] = (reference_date - feats["first_purchase"]).dt.days
    feats["avg_order_value_lifetime"] = (feats["monetary_lifetime"] / feats["orders_lifetime"].replace(0, np.nan)).fillna(0)
    feats["avg_order_value_90d"] = (feats["monetary_90d"] / feats["frequency_90d"].replace(0, np.nan)).fillna(0)
    feats["is_weekly_buyer"] = (feats["frequency_90d"].fillna(0) >= 12).astype(int)

    for c in ["share_Q1", "share_Q2", "share_Q3", "share_Q4"]:
        if c not in feats.columns:
            feats[c] = 0.0

    num_cols = feats.select_dtypes(include=[np.number]).columns
    feats[num_cols] = feats[num_cols].fillna(0)

    feats["churned"] = (feats["recency"] > churn_days).astype(int)

    base_cols = ["recency", "orders_lifetime", "monetary_lifetime"]
    extra_cols = ["avg_order_value_lifetime","avg_order_value_90d",
                  "days_since_first_purchase","frequency_90d","monetary_90d",
                  "is_weekly_buyer","share_Q1","share_Q2","share_Q3","share_Q4"]
    features = feats[["customer_id"] + base_cols + extra_cols].copy()
    features["log_monetary_lifetime"] = np.log1p(features["monetary_lifetime"])
    features = features.drop(columns=["monetary_lifetime"])

    leak = ["recency","frequency_90d","monetary_90d","avg_order_value_90d","is_weekly_buyer"]
    X = features.drop(columns=["customer_id"] + [c for c in leak if c in features.columns]).astype(float)
    y = feats["churned"].astype(int)
    ids = features["customer_id"].astype(int)

    feature_names = X.columns.tolist()
    assert not X.isna().any().any(), "NaN i X"
    assert not y.isna().any(), "NaN i y"

    log.info("Features byggda: X=%s features, y=%s labels, kunder=%s", X.shape[1], len(y), len(ids))
    feats_model = feats.copy()
    return X, y, ids, feats_model, reference_date, feature_names
