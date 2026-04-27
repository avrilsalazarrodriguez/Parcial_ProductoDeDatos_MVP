import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib

# Cargamos la base intermedia para separar el modelado del feature engineering
OUT_DIR = "data/prep"

train_df = pd.read_parquet(f"{OUT_DIR}/train.parquet")
valid_df = pd.read_parquet(f"{OUT_DIR}/valid.parquet")
test_X   = pd.read_parquet(f"{OUT_DIR}/test_features.parquet")
test_pairs = pd.read_parquet(f"{OUT_DIR}/test_pairs.parquet")

with open(f"{OUT_DIR}/meta.json") as f:
    meta = json.load(f)

feature_cols = meta["feature_cols"]

X_train = train_df[feature_cols]
y_train = train_df["y"].astype("float32")

X_valid = valid_df[feature_cols]
y_valid = valid_df["y"].astype("float32")

X_test = test_X[feature_cols]

# Algunas columnas funcionan mejor como categóricas en modelos de árboles (IDs y calendario)
cat_features = [c for c in ["shop_id","item_id","month","year"] if c in feature_cols]

print("X_train:", X_train.shape, "X_valid:", X_valid.shape, "X_test:", X_test.shape)
print("cat_features:", cat_features)

# Etapa 1: Clasificación (venta vs no venta)
# La idea es separar el "evento de venta" de la magnitud, porque hay muchos ceros
y_train_bin = (y_train > 0).astype(int)
y_valid_bin = (y_valid > 0).astype(int)

clf = lgb.LGBMClassifier(
    n_estimators=6000,
    learning_rate=0.03,
    num_leaves=256,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
)

clf.fit(
    X_train, y_train_bin,
    eval_set=[(X_valid, y_valid_bin)],
    eval_metric="binary_logloss",
    categorical_feature=cat_features,
    callbacks=[lgb.early_stopping(200, verbose=True)]
)

p_valid = clf.predict_proba(X_valid)[:, 1].astype(np.float32)
p_test  = clf.predict_proba(X_test)[:, 1].astype(np.float32)

print("p_valid summary:", float(p_valid.min()), float(p_valid.mean()), float(p_valid.max()))

# Etapa 2: Regresión (cuántas unidades, condicionado a que haya venta)
# Entrenamos solo con observaciones donde y>0 para enfocarnos en la magnitud
mask_pos_tr = y_train > 0
mask_pos_va = y_valid > 0

reg = lgb.LGBMRegressor(
    n_estimators=8000,
    learning_rate=0.03,
    num_leaves=256,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    objective="regression"
)

reg.fit(
    X_train.loc[mask_pos_tr], y_train.loc[mask_pos_tr],
    eval_set=[(X_valid.loc[mask_pos_va], y_valid.loc[mask_pos_va])],
    eval_metric="rmse",
    categorical_feature=cat_features,
    callbacks=[lgb.early_stopping(200, verbose=True)]
)

mu_valid = reg.predict(X_valid).astype(np.float32)
mu_test  = reg.predict(X_test).astype(np.float32)

print("mu_valid summary:", float(mu_valid.min()), float(mu_valid.mean()), float(mu_valid.max()))

# Predicción final: combinamos probabilidad de venta y magnitud
# Se recorta a [0,20] para ser consistente con la evaluación del reto
pred_valid = np.clip(p_valid * mu_valid, 0, 20)
pred_test  = np.clip(p_test  * mu_test,  0, 20)

rmse = float(np.sqrt(np.mean((pred_valid - y_valid.values) ** 2)))
print("RMSE valid (hurdle):", rmse)

# Submission: usamos test_pairs para mapear shop_id/item_id al ID de Kaggle
pred_map = pd.DataFrame({
    "shop_id": test_pairs["shop_id"].values,
    "item_id": test_pairs["item_id"].values,
    "item_cnt_month": pred_test.astype(np.float32)
})

test= pd.read_csv("data/raw/test.csv")
test.columns = test.columns.str.strip()
test["ID"] = pd.to_numeric(test["ID"], errors="coerce").astype(np.int32)
test["shop_id"] = pd.to_numeric(test["shop_id"], errors="coerce").astype(np.int16)
test["item_id"] = pd.to_numeric(test["item_id"], errors="coerce").astype(np.int16)

submission = test.merge(pred_map, on=["shop_id","item_id"], how="left")
submission["item_cnt_month"] = submission["item_cnt_month"].fillna(0).clip(0, 20)

submission = submission[["ID","item_cnt_month"]]
submission.to_csv("data/predictions/submission.csv", index=False)

#print(submission.head())

# Guardamos el modelo como bundle para poder reproducir predicciones después
bundle = {
    "clf": clf,
    "reg": reg,
    "feature_cols": feature_cols,
    "cat_features": cat_features,
    "meta": meta
}

joblib.dump(bundle, "artifacts/model.joblib")
print("guardado: model.joblib")
joblib.dump({"model": baseline_lgbm, "basic_cols": basic_cols}, "artifacts/baseline.joblib")
print("guardado: baseline.joblib")
