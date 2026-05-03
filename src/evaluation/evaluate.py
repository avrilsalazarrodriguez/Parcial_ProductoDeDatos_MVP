import json
import os
import joblib
import numpy as np
import pandas as pd


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def main():
    base_dir = "/opt/ml/processing"

    
    # Cargamos modelos
    
    import tarfile
    
    model_tar_path = os.path.join(base_dir, "input", "model", "model.tar.gz")
    extract_dir = os.path.join(base_dir, "input", "model_extracted")
    
    os.makedirs(extract_dir, exist_ok=True)
    
    with tarfile.open(model_tar_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)
        
    model_path = os.path.join(extract_dir, "model.joblib")
    payload = joblib.load(model_path)
        
    bundle = payload["bundle"]
    feature_cols = bundle["feature_cols"]


    # Cargamos test data

    test_dir = os.path.join(base_dir, "input", "test")
    
    valid_path = os.path.join(test_dir, "valid.parquet")
    test_path = os.path.join(test_dir, "test.parquet")
    
    if os.path.exists(valid_path):
        df = pd.read_parquet(valid_path)
    else:
        df = pd.read_parquet(test_path)

    
    # Estructura real
    
    y_true = df["y"].values
    X = df[feature_cols]

    
    # Predicción
    prob = bundle["clf"].predict_proba(X)[:, 1].astype(np.float32)
    mu = bundle["reg"].predict(X).astype(np.float32)

    preds = np.clip(prob * mu, 0, 20)

    
    # RMSE

    score = rmse(y_true, preds)


    # Guardamos evaluation.json

    output_dir = os.path.join(base_dir, "output", "evaluation")
    os.makedirs(output_dir, exist_ok=True)

    report = {
        "regression_metrics": {
            "rmse": {
                "value": float(score)
            }
        }
    }

    with open(os.path.join(output_dir, "evaluation.json"), "w") as f:
        json.dump(report, f)


if __name__ == "__main__":
    main()