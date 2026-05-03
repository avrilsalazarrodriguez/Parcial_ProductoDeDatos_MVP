"""Repair/merge evaluation_curves_by_model for hybrid outputs.

This script does not train models. It only merges existing curves and, when possible,
rebuilds curves from evaluation_detail columns such as pred_specialist_recurrent,
pred_hgb_poisson, pred_naive_lag1, pred_rolling_mean_3.

Use it only if the app selector does not show some models even though they appear in
Model Registry.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PRED_COLUMN_TO_MODEL = {
    "prediction": ("hybrid_router_v1", "Hybrid Router: inactive + naive + specialist + HGB"),
    "pred_hybrid_router_v1": ("hybrid_router_v1", "Hybrid Router: inactive + naive + specialist + HGB"),
    "pred_champion": ("hurdle_hgb", "Hurdle HGB"),
    "pred_hurdle_hgb": ("hurdle_hgb", "Hurdle HGB"),
    "pred_naive_lag1": ("naive_lag1", "Naive último periodo"),
    "pred_rolling_mean_3": ("rolling_mean_3", "Rolling mean últimos 3 lags"),
    "pred_specialist_recurrent": ("specialist_recurrent", "Especialista demanda recurrente"),
    "pred_hgb_poisson": ("hgb_poisson", "HistGradientBoosting Poisson"),
    "pred_poisson_hgb": ("hgb_poisson", "HistGradientBoosting Poisson"),
    "pred_incumbent_two_stage": ("incumbent_two_stage", "LightGBM original dos etapas"),
    "pred_original_two_stage": ("incumbent_two_stage", "LightGBM original dos etapas"),
}


def read_parquet_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def normalize_existing_curves(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "demand_bucket" not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    if "model_id" not in out.columns:
        if "section" in out.columns:
            out["model_id"] = out["section"].astype(str)
        elif "series" in out.columns:
            out["model_id"] = out["series"].astype(str)
        else:
            out["model_id"] = "model"
    if "model_name" not in out.columns:
        out["model_name"] = out["model_id"].astype(str)
    if {"real_mean", "pred_mean"}.issubset(out.columns):
        keep = ["demand_bucket", "model_id", "model_name", "real_mean", "pred_mean"]
        return out[keep].copy()
    if {"series", "mean_value"}.issubset(out.columns):
        pivot = out.pivot_table(
            index=["demand_bucket", "model_id", "model_name"],
            columns="series",
            values="mean_value",
            aggfunc="mean",
        ).reset_index()
        real_col = "Real" if "Real" in pivot.columns else "real_mean" if "real_mean" in pivot.columns else None
        pred_candidates = [c for c in pivot.columns if c not in {"demand_bucket", "model_id", "model_name", real_col}]
        rows = []
        for _, row in pivot.iterrows():
            for pred_col in pred_candidates:
                rows.append(
                    {
                        "demand_bucket": row["demand_bucket"],
                        "model_id": row["model_id"],
                        "model_name": row["model_name"],
                        "real_mean": row[real_col] if real_col else pd.NA,
                        "pred_mean": row[pred_col],
                    }
                )
        return pd.DataFrame(rows)
    return pd.DataFrame()


def curves_from_detail(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    y_col = "y" if "y" in detail.columns else "y_true" if "y_true" in detail.columns else None
    if y_col is None:
        return pd.DataFrame()
    df = detail.copy()
    if "demand_bucket" not in df.columns:
        df["demand_bucket"] = pd.cut(
            pd.to_numeric(df[y_col], errors="coerce"),
            bins=[-0.1, 0, 1, 3, 7, 20],
            include_lowest=True,
        ).astype(str)
    rows = []
    for pred_col, (model_id, model_name) in PRED_COLUMN_TO_MODEL.items():
        if pred_col not in df.columns:
            continue
        tmp = df[["demand_bucket", y_col, pred_col]].copy()
        tmp[y_col] = pd.to_numeric(tmp[y_col], errors="coerce")
        tmp[pred_col] = pd.to_numeric(tmp[pred_col], errors="coerce")
        tmp = tmp.dropna(subset=[y_col, pred_col])
        if tmp.empty:
            continue
        agg = tmp.groupby("demand_bucket", as_index=False).agg(
            real_mean=(y_col, "mean"),
            pred_mean=(pred_col, "mean"),
        )
        agg["model_id"] = model_id
        agg["model_name"] = model_name
        rows.append(agg[["demand_bucket", "model_id", "model_name", "real_mean", "pred_mean"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hybrid-root", default="modelops_outputs_hybrid")
    parser.add_argument("--previous-root", default="modelops_outputs")
    args = parser.parse_args()

    roots = [Path(args.hybrid_root), Path(args.previous_root)]
    frames = []
    for root in roots:
        frames.append(normalize_existing_curves(read_parquet_if_exists(root / "latest/evaluation/evaluation_curves_by_model.parquet")))
        frames.append(curves_from_detail(read_parquet_if_exists(root / "latest/evaluation/evaluation_detail.parquet")))
        frames.append(curves_from_detail(read_parquet_if_exists(root / "evaluation/evaluation_detail.parquet")))

    merged = pd.concat([f for f in frames if f is not None and not f.empty], ignore_index=True) if any(not f.empty for f in frames) else pd.DataFrame()
    if merged.empty:
        raise SystemExit("No curves or evaluation_detail files found to merge/rebuild.")

    merged["model_id"] = merged["model_id"].astype(str)
    merged["model_name"] = merged["model_name"].astype(str)
    merged = (
        merged.dropna(subset=["demand_bucket", "model_id", "pred_mean"])
        .groupby(["demand_bucket", "model_id", "model_name"], as_index=False)
        .agg(real_mean=("real_mean", "mean"), pred_mean=("pred_mean", "mean"))
    )

    output = Path(args.hybrid_root) / "latest/evaluation/evaluation_curves_by_model.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output, index=False)
    print(f"wrote {output}")
    print("models:", sorted(merged["model_id"].unique().tolist()))


if __name__ == "__main__":
    main()
