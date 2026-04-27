import os
import numpy as np
import pandas as pd
import joblib

RAW_DIR = "data/raw"
INFER_DIR = "data/inference"
PREP_DIR = "data/prep"
PRED_DIR = "data/predictions"
ARTIFACTS_DIR = "artifacts"


def main() -> None:
    os.makedirs(PRED_DIR, exist_ok=True)

    bundle = joblib.load(os.path.join(ARTIFACTS_DIR, "model.joblib"))
    clf = bundle["clf"]
    reg = bundle["reg"]
    feature_cols = bundle["feature_cols"]

    X_test = pd.read_parquet(os.path.join(INFER_DIR, "test_features.parquet"))
    X_test = X_test[feature_cols]

    p = clf.predict_proba(X_test)[:, 1].astype(np.float32)
    mu = reg.predict(X_test).astype(np.float32)
    pred = np.clip(p * mu, 0, 20)

    test_pairs = pd.read_parquet(os.path.join(PREP_DIR, "test_pairs.parquet"))
    pred_map = pd.DataFrame(
        {
            "shop_id": test_pairs["shop_id"].values,
            "item_id": test_pairs["item_id"].values,
            "item_cnt_month": pred,
        }
    )

    test = pd.read_csv(os.path.join(RAW_DIR, "test.csv"))
    test.columns = test.columns.str.strip()

    sub = test.merge(pred_map, on=["shop_id", "item_id"], how="left")
    sub["item_cnt_month"] = sub["item_cnt_month"].fillna(0).clip(0, 20)
    sub = sub[["ID", "item_cnt_month"]]

    sub.to_csv(os.path.join(PRED_DIR, "submission.csv"), index=False)


if __name__ == "__main__":
    main()
