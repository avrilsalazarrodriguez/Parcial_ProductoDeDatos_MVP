from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

INVALID = {"", "nan", "none", "null", "nat", "noce"}


def clean_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if text.lower() in INVALID:
        return None
    return text


def read_csv_optional(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def read_parquet_optional(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def read_json_optional(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def best_model_label(row: pd.Series, fallback: str = "Modelo ganador") -> str:
    for col in ["model_name", "model_id", "section", "model_scope", "series"]:
        if col in row.index:
            text = clean_text(row.get(col))
            if text:
                return text
    return fallback


def clean_curves(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()

    if "model_id" in out.columns and "model_name" in out.columns:
        out["model_name"] = out.apply(
            lambda row: clean_text(row.get("model_name")) or clean_text(row.get("model_id")) or "Modelo ganador",
            axis=1,
        )
    elif "model_id" in out.columns and "model_name" not in out.columns:
        out["model_name"] = out["model_id"].map(lambda x: clean_text(x) or "Modelo ganador")
    elif "model_name" in out.columns:
        out["model_name"] = out["model_name"].map(lambda x: clean_text(x) or "Modelo ganador")

    if "series" in out.columns:
        def fix_series(row: pd.Series) -> str:
            raw = clean_text(row.get("series"))
            if raw is None:
                return best_model_label(row)
            if raw.lower() in {"pred", "prediction", "pred_mean", "forecast", "mean_prediction"}:
                return best_model_label(row)
            if raw.lower() in {"real", "real_mean", "true", "true_mean", "y", "y_mean"}:
                return "Real"
            return raw
        out["series"] = out.apply(fix_series, axis=1)

    return out


def merge_model_runs(previous_root: Path, hybrid_root: Path) -> pd.DataFrame:
    prev = read_csv_optional(previous_root / "registry" / "model_runs.csv")
    curr = read_csv_optional(hybrid_root / "registry" / "model_runs.csv")
    frames = [df for df in [curr, prev] if not df.empty]
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True, sort=False)
    if "model_id" in merged.columns:
        merged = merged.drop_duplicates("model_id", keep="first")
    return merged


def merge_curves(previous_root: Path, hybrid_root: Path) -> pd.DataFrame:
    candidates = [
        hybrid_root / "latest" / "evaluation" / "evaluation_curves_by_model.parquet",
        previous_root / "latest" / "evaluation" / "evaluation_curves_by_model.parquet",
    ]
    frames = [clean_curves(read_parquet_optional(p)) for p in candidates if p.exists()]
    frames = [df for df in frames if not df.empty]
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True, sort=False)
    # Remove exact duplicates while preserving extra model curves.
    subset = [c for c in ["demand_bucket", "series", "model_id", "model_name", "mean_value", "real_mean", "pred_mean"] if c in merged.columns]
    if subset:
        merged = merged.drop_duplicates(subset=subset, keep="first")
    return merged


def merge_metrics(previous_root: Path, hybrid_root: Path, merged_runs: pd.DataFrame) -> dict[str, Any]:
    curr_path = hybrid_root / "latest" / "evaluation" / "model_metrics.json"
    prev_path = previous_root / "latest" / "evaluation" / "model_metrics.json"
    curr = read_json_optional(curr_path)
    prev = read_json_optional(prev_path)
    if not curr:
        curr = {}
    all_models = []
    for payload in [curr, prev]:
        models = payload.get("all_models") or payload.get("models") or []
        if isinstance(models, list):
            all_models.extend(models)
    if not merged_runs.empty:
        all_models.extend(merged_runs.to_dict(orient="records"))
    # unique by model_id
    uniq = {}
    for row in all_models:
        mid = clean_text(row.get("model_id")) or clean_text(row.get("model_name"))
        if mid and mid not in uniq:
            uniq[mid] = row
    curr["all_models"] = list(uniq.values())
    return curr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous-root", default="modelops_outputs")
    parser.add_argument("--hybrid-root", default="modelops_outputs_hybrid")
    args = parser.parse_args()

    previous_root = Path(args.previous_root)
    hybrid_root = Path(args.hybrid_root)

    if not hybrid_root.exists():
        raise FileNotFoundError(f"hybrid root not found: {hybrid_root}")

    merged_runs = merge_model_runs(previous_root, hybrid_root)
    if not merged_runs.empty:
        out = hybrid_root / "registry" / "model_runs.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        merged_runs.to_csv(out, index=False)
        print(f"wrote {out} rows={len(merged_runs)}")
        print(merged_runs[[c for c in ["model_id", "model_name", "rmse", "mae"] if c in merged_runs.columns]].to_string(index=False))

    merged_curves = merge_curves(previous_root, hybrid_root)
    if not merged_curves.empty:
        out = hybrid_root / "latest" / "evaluation" / "evaluation_curves_by_model.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        merged_curves.to_parquet(out, index=False)
        print(f"wrote {out} rows={len(merged_curves)}")
        if "series" in merged_curves.columns:
            print("curve series:", sorted(merged_curves["series"].dropna().astype(str).unique().tolist()))
        if "model_name" in merged_curves.columns:
            print("curve model_name:", sorted(merged_curves["model_name"].dropna().astype(str).unique().tolist()))

    metrics = merge_metrics(previous_root, hybrid_root, merged_runs)
    if metrics:
        out = hybrid_root / "latest" / "evaluation" / "model_metrics.json"
        write_json(out, metrics)
        print(f"wrote {out}")

    print("repair complete")


if __name__ == "__main__":
    main()
