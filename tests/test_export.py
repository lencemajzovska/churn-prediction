"""Enhetstester för export.py
(export av resultat och metadata)
"""

import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.data_prep import Paths
from src.export import export_results


@pytest.fixture
def tmp_paths(tmp_path: Path) -> Paths:
    root = tmp_path / "proj"
    input_dir = root / "data_input"
    output_dir = root / "data_output"
    model_dir = root / "models"
    images_dir = root / "images"

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    class _P(Paths):
        pass

    return _P(
        project_root=root,
        input_dir=input_dir,
        output_dir=output_dir,
        model_dir=model_dir,
        images_dir=images_dir,
        sqlite_path=root / "churn.db",
    )


@pytest.fixture
def tiny_data():
    X = pd.DataFrame({
        "recency": [5, 20, 40, 3, 15, 8],
        "frequency_lifetime": [2, 4, 1, 5, 3, 2],
    })
    y = pd.Series([0, 1, 0, 1, 0, 0], name="churned")
    feats_model = pd.DataFrame({
        "customer_id": [101, 102, 103, 104, 105, 106],
        "first_purchase": pd.to_datetime(["2010-12-15"]*6),
        "last_purchase": pd.to_datetime([
            "2011-06-10", "2011-07-05", "2011-05-28",
            "2011-08-12", "2011-09-03", "2011-10-22"
            ]),
        "frequency_lifetime": [2,4,1,5,3,2],
        "monetary_lifetime": [100, 200, 50, 350, 120, 90],
        "frequency_recent": [1,2,0,3,1,1],
        "monetary_recent": [50, 120, 0, 200, 60, 40],
        "share_Q1": [0.5]*6, "share_Q2": [0.2]*6, "share_Q3": [0.2]*6, "share_Q4": [0.1]*6,
        "recency": [5,20,40,3,15,8],
        "days_since_first_purchase": [400, 390, 410, 380, 395, 405],
        "avg_order_value_lifetime": [50, 50, 50, 70, 40, 45],
        "avg_order_value_recent": [50, 60, 0, 67, 60, 40],
        "is_weekly_buyer": [0,0,0,1,0,0],
        "churned": [0,1,0,1,0,0],
    })

    feature_names = ["recency", "frequency_lifetime"]
    return X, y, feats_model, feature_names


def test_export_runs_ok(tmp_paths: Paths, tiny_data):
    X, y, feats_model, feature_names = tiny_data
    best_model = LogisticRegression(max_iter=200, solver="liblinear")
    best_model.fit(X, y)

    export_results(
        best_model=best_model,
        X=X,
        y=y,
        feature_names=feature_names,
        feats_model=feats_model,
        importance_df=None,
        cv_df=None,
        eval_compare=None,
        riskband_summary=None,
        paths=tmp_paths,
        best_name="LogReg (baseline)",
        calib_used="none",
        reference_date=pd.Timestamp("2011-12-09"),
        use_calibrated_for_export=False,
    )

    scores = tmp_paths.output_dir / "churn_scores.csv"
    riskband = tmp_paths.output_dir / "riskband_summary.csv"
    assert scores.exists()
    assert riskband.exists()

    df_scores = pd.read_csv(scores)
    assert "RiskScore" in df_scores.columns or "risk_score" in df_scores.columns
    assert "RiskBand" in df_scores.columns or "risk_band" in df_scores.columns

    with sqlite3.connect(tmp_paths.sqlite_path) as con:
        cur = con.execute("SELECT name FROM sqlite_master WHERE type='table'")
        names = {r[0] for r in cur.fetchall()}
        assert "churn_scores" in names

    model_path = tmp_paths.model_dir / "final_model_LogReg (baseline).joblib"
    meta_path = tmp_paths.model_dir / "final_meta.json"

    assert model_path.exists()
    assert meta_path.exists()

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["model_name"].startswith("LogReg")


def test_supports_calibration(tmp_paths: Paths, tiny_data):
    X, y, feats_model, feature_names = tiny_data
    best_model = LogisticRegression(max_iter=200, solver="liblinear").fit(X, y)

    export_results(
        best_model=best_model,
        X=X,
        y=y,
        feature_names=feature_names,
        feats_model=feats_model,
        importance_df=pd.DataFrame({
            "feature": feature_names,
            "importance": [0.6, 0.4]}),

        cv_df=pd.DataFrame({
            "model": ["LogReg (baseline)"],
            "auc_mean": [0.7],
            "auc_std": [0.02]}),

        eval_compare=pd.DataFrame({
            "Model": ["A","B"],
            "AUC":[0.7,0.69],
            "F1@0.50":[0.5,0.48],
            "BestF1":[0.55,0.5],
            "BestThr":[0.42,0.44],
            "Precision@10":[0.6,0.59]}),

        paths=tmp_paths,
        best_name="LogReg (baseline)",
        calib_used="isotonic",
        reference_date=pd.Timestamp("2011-12-09"),
        use_calibrated_for_export=True,
    )

    assert (tmp_paths.output_dir / "cv_auc.csv").exists()
    assert (
        (tmp_paths.output_dir / "calibration.csv").exists()
        or (tmp_paths.output_dir / "calibration_comparison.csv").exists()
    )
    assert (tmp_paths.output_dir / "feature_importance.csv").exists()


def test_missing_customer_id_raises(tmp_paths: Paths, tiny_data):
    X, y, feats_model, feature_names = tiny_data
    feats_bad = feats_model.drop(columns=["customer_id"])
    best_model = LogisticRegression(max_iter=200, solver="liblinear").fit(X, y)

    with pytest.raises(ValueError, match="feats_model must contain 'customer_id'"):
        export_results(
            best_model=best_model,
            X=X,
            y=y,
            feature_names=feature_names,
            feats_model=feats_bad,
            importance_df=None,
            cv_df=None,
            eval_compare=None,
            riskband_summary=None,
            paths=tmp_paths,
            best_name="LogReg (baseline)",
            calib_used="none",
            reference_date=pd.Timestamp("2011-12-09"),
            use_calibrated_for_export=False,
        )