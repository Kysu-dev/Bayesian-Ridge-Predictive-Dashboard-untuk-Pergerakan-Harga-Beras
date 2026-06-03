import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge
from analysis_utils import analyze_outliers, analyze_eda, build_scalers, capture_warnings, iqr_bounds, compute_metrics, save_prediction_plots, save_outlier_compare_plot, fit_diagnosis

DATASET_NAME = "Harga Beras Indonesia (PIHPS)"
TARGET_COL = "Price"
TARGET_LABEL = "Harga Rata-rata Nasional (Rp)"

# -----------------------------------------------------------------------------
# Feature engineering (best config: lags=3, use_rolling=False)
# -----------------------------------------------------------------------------

def create_lag_features(data, col, lags=7, use_rolling=False):
    df = data[[col]].copy()
    for i in range(1, lags + 1):
        df[f'lag_{i}'] = df[col].shift(i)
    if use_rolling:
        df['rolling_mean_7']  = df[col].shift(1).rolling(7).mean()
        df['rolling_mean_14'] = df[col].shift(1).rolling(14).mean()
        df['rolling_std_7']   = df[col].shift(1).rolling(7).std()
    return df.dropna()

def load_series():
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(base_dir, 'dataset', 'komoditas_beras_2022_2026.csv'))
    df['Date_Param'] = pd.to_datetime(df['Date_Param'])
    ts = df.groupby('Date_Param')[TARGET_COL].mean().reset_index()
    ts = ts.rename(columns={'Date_Param': 'Date', TARGET_COL: TARGET_COL})
    ts = ts.sort_values('Date').set_index('Date')
    ts = ts.resample('D').mean().ffill()
    return ts[[TARGET_COL]]

# -----------------------------------------------------------------------------
# Main run function
# -----------------------------------------------------------------------------

def run(run_ctx=None):
    ts = load_series()
    target = TARGET_COL
    outlier_info = analyze_outliers(ts[target], DATASET_NAME, target, run_ctx)
    eda_info = analyze_eda(ts[target], DATASET_NAME, target, run_ctx)

    # Best config
    LAGS = 7
    USE_ROLLING = False
    SCALER_TYPE = "standard"

    def train_eval(df_in):
        df_feat = create_lag_features(df_in, target, lags=LAGS, use_rolling=USE_ROLLING)
        X = df_feat.drop(target, axis=1)
        y = df_feat[target]

        split_ratio = 0.8
        if run_ctx and isinstance(run_ctx, dict):
            split_ratio = run_ctx.get("split", {}).get("ratio", split_ratio)
        split = int(len(X) * split_ratio)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        scaler_X, scaler_y = build_scalers(SCALER_TYPE)
        X_train_sc = scaler_X.fit_transform(X_train)
        X_test_sc  = scaler_X.transform(X_test)
        y_train_sc = scaler_y.fit_transform(y_train.values.reshape(-1,1)).ravel()

        model_params = {
            "max_iter": 300,
            "tol": 1e-3,
            "alpha_1": 1e-6,
            "alpha_2": 1e-6,
            "lambda_1": 1e-6,
            "lambda_2": 1e-6,
        }
        if run_ctx and isinstance(run_ctx, dict):
            model_params.update(run_ctx.get("model", {}) or {})
        model = BayesianRidge(**model_params)
        _, warn = capture_warnings(model.fit, X_train_sc, y_train_sc)

        y_train_pred_sc = model.predict(X_train_sc)
        y_train_pred = scaler_y.inverse_transform(y_train_pred_sc.reshape(-1, 1)).ravel()
        y_pred_sc, _ = model.predict(X_test_sc, return_std=True)
        y_pred = scaler_y.inverse_transform(y_pred_sc.reshape(-1,1)).ravel()

        metrics = compute_metrics(y_test, y_pred)
        diagnosis = fit_diagnosis(
            model,
            y_train,
            y_train_pred,
            y_test,
            y_pred,
            max_iter=model_params.get('max_iter', 200)
        )
        return metrics, model, scaler_X, scaler_y, y_test, y_pred, warn, diagnosis, y_train, X, df_feat

    # Base run (tanpa clipping, tapi best config outlier=none)
    df_base = ts[[target]].copy()
    metrics_base, _, _, _, _, _, warn_base, _, _, _, _ = train_eval(df_base)

    # Clipping tidak dilakukan karena best config outlier = none
    df_clip = ts[[target]].copy()
    handling = None
    iqr_k = 1.5
    split_ratio = 0.8
    if run_ctx and isinstance(run_ctx, dict):
        out_cfg = run_ctx.get("outlier", {})
        handling = str(out_cfg.get("handling") or "").lower()
        iqr_k = out_cfg.get("iqr_k", iqr_k)
        split_ratio = run_ctx.get("split", {}).get("ratio", split_ratio)
    if handling == "clip":
        split_idx = int(len(df_clip) * split_ratio)
        train_series = df_clip[target].iloc[:split_idx]
        lower, upper = iqr_bounds(train_series, iqr_k=iqr_k)
        if lower is not None and upper is not None:
            df_clip[target] = df_clip[target].clip(lower=lower, upper=upper)

    metrics_clip, model, scaler_X, scaler_y, y_test, y_pred, warn_clip, diagnosis, y_train, X, df_feat = train_eval(df_clip)
    ts_used = df_clip

    pred_output_dir = None
    pred_enable = True
    if run_ctx and isinstance(run_ctx, dict):
        pred_cfg = run_ctx.get("pred", {})
        pred_output_dir = pred_cfg.get("output_dir")
        pred_enable = pred_cfg.get("enable_plots", pred_enable)
    if pred_output_dir and pred_enable:
        save_prediction_plots(y_test, y_pred, pred_output_dir, DATASET_NAME, target)
        save_outlier_compare_plot(metrics_base, metrics_clip, pred_output_dir, DATASET_NAME, target)

    return {
        "Dataset": DATASET_NAME,
        "Target": TARGET_LABEL,
        "MAE": metrics_clip["MAE"],
        "RMSE": metrics_clip["RMSE"],
        "MAPE(%)": metrics_clip["MAPE(%)"],
        "R2": metrics_clip["R2"],
        "_diagnosis": diagnosis,
        "_model": model,
        "_scaler_X": scaler_X,
        "_scaler_y": scaler_y,
        "_ts": ts_used,
        "_series": ts_used,
        "_y_test": y_test,
        "_y_pred": y_pred,
        "_y_std": None,
        "_y_train": y_train,
        "_X": X,
        "_df_feat": df_feat,
        "_lags": LAGS,
        "_use_rolling": USE_ROLLING,
        "_outlier": outlier_info,
        "_eda": eda_info,
        "_warnings": list(dict.fromkeys(warn_base + warn_clip)),
    }

if __name__ == "__main__":
    result = run()
    print(f"\n=== {result['Dataset']} ===")
    for k, v in result.items():
        print(f"  {k}: {v}")