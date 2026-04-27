"""
create_app_tables.py

Genera tablas pequeñas para la aplicación a partir de los artefactos existentes.

Entradas:
- data/prep/valid.parquet
- data/prep/test_pairs.parquet
- data/predictions/submission.csv
- artifacts/model.joblib

Salidas:
- data/app/forecast_results.csv
- data/app/problem_products.csv
- data/app/model_metadata.csv

Estas tablas son el puente entre el pipeline de ML existente y la aplicación.
No se rehace el modelo: solo se organizan resultados para que la app y RDS
los puedan consumir.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


MODEL_PATH = Path("artifacts/model.joblib")
VALID_PATH = Path("data/prep/valid.parquet")
TEST_PAIRS_PATH = Path("data/prep/test_pairs.parquet")
SUBMISSION_PATH = Path("data/predictions/submission.csv")
APP_DATA_DIR = Path("data/app")


def load_model() -> dict:
    """Cargamos el modelo existente entrenado en tareas previas."""
    return joblib.load(MODEL_PATH)


def predict_with_model(model_payload: dict, features: pd.DataFrame) -> np.ndarray:
    """
    Ejecuta inferencia con el modelo de dos etapas.

    El modelo combina:
    - un clasificador para estimar probabilidad de venta
    - un regresor para estimar unidades cuando hay venta
    """
    bundle = model_payload["bundle"]
    feature_cols = bundle["feature_cols"]

    x_data = features[feature_cols]
    prob = bundle["clf"].predict_proba(x_data)[:, 1].astype(np.float32)
    mu = bundle["reg"].predict(x_data).astype(np.float32)

    return np.clip(prob * mu, 0, 20)


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Calcula RMSE para comparar modelo contra baseline."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Calcula MAE para explicar el error promedio en unidades."""
    return float(np.mean(np.abs(y_true - y_pred)))


def build_forecast_results() -> pd.DataFrame:
    """
    Une los pares tienda-producto del test con el archivo de predicciones.

    Esta tabla representa el pronóstico batch que ve negocio.
    """
    test_pairs = pd.read_parquet(TEST_PAIRS_PATH)
    submission = pd.read_csv(SUBMISSION_PATH)

    forecast = test_pairs.copy()
    forecast["prediction"] = submission.iloc[:, -1].values
    forecast["prediction"] = forecast["prediction"].clip(0, 20)
    forecast["forecast_month"] = "next_month"

    return forecast[["shop_id", "item_id", "prediction", "forecast_month"]]


def build_evaluation_tables(model_payload: dict) -> tuple[pd.DataFrame, dict]:
    """
    Evalúa el modelo en validación y genera productos candidatos a revisión.

    Para que el script sea rápido, usamos una muestra reproducible.
    """
    valid_df = pd.read_parquet(VALID_PATH)
    sample = valid_df.sample(min(10000, len(valid_df)), random_state=42).copy()

    sample["prediction"] = predict_with_model(model_payload, sample)
    sample["naive_prediction"] = sample["cnt_lag_1"].clip(0, 20)
    sample["error"] = sample["prediction"] - sample["y"]
    sample["abs_error"] = sample["error"].abs()

    metrics = {
        "rmse_model": rmse(sample["y"], sample["prediction"]),
        "rmse_naive": rmse(sample["y"], sample["naive_prediction"]),
        "mae_model": mae(sample["y"], sample["prediction"]),
    }

    problem_products = (
        sample.sort_values("abs_error", ascending=False)
        [["shop_id", "item_id", "y", "prediction", "abs_error"]]
        .head(100)
        .rename(columns={"y": "y_true"})
        .copy()
    )
    problem_products["reason"] = "high_validation_error"

    return problem_products, metrics


def build_model_metadata(metrics: dict) -> pd.DataFrame:
    """Crea una fila de metadata para documentar el modelo activo."""
    return pd.DataFrame(
        [
            {
                "model_name": "predict_future_sales_lgbm_two_stage",
                "model_type": "LightGBM classifier + regressor",
                "artifact_path": "artifacts/model.joblib",
                "rmse_model": metrics["rmse_model"],
                "rmse_naive": metrics["rmse_naive"],
                "mae_model": metrics["mae_model"],
                "notes": (
                    "Modelo reutilizado de tareas previas. "
                    "La app usa inferencia in-process con @st.cache_resource "
                    "y batch precomputado para responder rápido."
                ),
            }
        ]
    )


def main() -> None:
    """Genera los CSVs operacionales que se cargarán en RDS."""
    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)

    model_payload = load_model()

    forecast_results = build_forecast_results()
    problem_products, metrics = build_evaluation_tables(model_payload)
    model_metadata = build_model_metadata(metrics)

    forecast_results.to_csv(APP_DATA_DIR / "forecast_results.csv", index=False)
    problem_products.to_csv(APP_DATA_DIR / "problem_products.csv", index=False)
    model_metadata.to_csv(APP_DATA_DIR / "model_metadata.csv", index=False)

    print("Tablas generadas en data/app/")
    print(f"forecast_results: {len(forecast_results):,} filas")
    print(f"problem_products: {len(problem_products):,} filas")
    print(f"model_metadata: {len(model_metadata):,} filas")
    print(metrics)


if __name__ == "__main__":
    main()
