#!/usr/bin/env python3
"""Merge previous model-zoo entries, such as incumbent_two_stage, into hybrid outputs.

Use this only if the previous ModelOps root still contains the original LightGBM
or incumbent model in registry/model_runs.csv or latest/evaluation/evaluation_curves_by_model.parquet.
It does not retrain models and does not change the champion unless it was already
marked in the hybrid output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--hybrid-root', default='modelops_outputs_hybrid')
    parser.add_argument('--previous-root', default='modelops_outputs')
    args = parser.parse_args()

    hybrid_root = Path(args.hybrid_root)
    previous_root = Path(args.previous_root)

    hybrid_runs_path = hybrid_root / 'registry/model_runs.csv'
    previous_runs_path = previous_root / 'registry/model_runs.csv'

    hybrid_runs = read_csv(hybrid_runs_path)
    previous_runs = read_csv(previous_runs_path)

    if not previous_runs.empty:
        wanted_mask = pd.Series(False, index=previous_runs.index)
        for col in ['model_id', 'model_name', 'model_family']:
            if col in previous_runs.columns:
                wanted_mask = wanted_mask | previous_runs[col].astype(str).str.contains(
                    'incumbent|lightgbm|original|two_stage', case=False, na=False
                )
        add_runs = previous_runs[wanted_mask].copy()
        if not add_runs.empty:
            if 'model_id' in hybrid_runs.columns and 'model_id' in add_runs.columns:
                add_runs = add_runs[~add_runs['model_id'].astype(str).isin(hybrid_runs['model_id'].astype(str))]
            merged_runs = pd.concat([hybrid_runs, add_runs], ignore_index=True)
            hybrid_runs_path.parent.mkdir(parents=True, exist_ok=True)
            merged_runs.to_csv(hybrid_runs_path, index=False)
            print(f'added model_runs rows: {len(add_runs)}')
        else:
            print('no incumbent/lightgbm rows found in previous model_runs.csv')
    else:
        print('previous model_runs.csv not found')

    hybrid_curves_path = hybrid_root / 'latest/evaluation/evaluation_curves_by_model.parquet'
    previous_curves_path = previous_root / 'latest/evaluation/evaluation_curves_by_model.parquet'
    hybrid_curves = read_parquet(hybrid_curves_path)
    previous_curves = read_parquet(previous_curves_path)

    if not previous_curves.empty:
        wanted_mask = pd.Series(False, index=previous_curves.index)
        for col in ['model_id', 'model_name', 'section', 'series']:
            if col in previous_curves.columns:
                wanted_mask = wanted_mask | previous_curves[col].astype(str).str.contains(
                    'incumbent|lightgbm|original|two_stage', case=False, na=False
                )
        add_curves = previous_curves[wanted_mask].copy()
        if not add_curves.empty:
            if not hybrid_curves.empty:
                key_cols = [c for c in ['model_id', 'model_name', 'section', 'series'] if c in hybrid_curves.columns and c in add_curves.columns]
                if key_cols:
                    existing_keys = set(map(tuple, hybrid_curves[key_cols].astype(str).to_numpy()))
                    add_curves = add_curves[~add_curves[key_cols].astype(str).apply(tuple, axis=1).isin(existing_keys)]
            merged_curves = pd.concat([hybrid_curves, add_curves], ignore_index=True)
            hybrid_curves_path.parent.mkdir(parents=True, exist_ok=True)
            merged_curves.to_parquet(hybrid_curves_path, index=False)
            print(f'added curve rows: {len(add_curves)}')
        else:
            print('no incumbent/lightgbm rows found in previous evaluation_curves_by_model.parquet')
    else:
        print('previous evaluation_curves_by_model.parquet not found')

    metrics_path = hybrid_root / 'latest/evaluation/model_metrics.json'
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
        runs_now = read_csv(hybrid_runs_path)
        if not runs_now.empty:
            metrics['all_models'] = runs_now.to_dict(orient='records')
            write_json(metrics_path, metrics)
            print('updated model_metrics.json all_models')


if __name__ == '__main__':
    main()
