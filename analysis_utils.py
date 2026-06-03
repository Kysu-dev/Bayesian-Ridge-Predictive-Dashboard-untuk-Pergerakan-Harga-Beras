import os
import re
import pandas as pd


def slugify(text):
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "item"


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def dataset_output_dir(base_dir, dataset_name):
    if not base_dir:
        return None
    out_dir = os.path.join(base_dir, slugify(dataset_name))
    ensure_dir(out_dir)
    return out_dir


def capture_warnings(callable_obj, *args, **kwargs):
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = callable_obj(*args, **kwargs)

    messages = []
    for item in caught:
        category = getattr(item, "category", None)
        name = category.__name__ if category else "Warning"
        msg = str(item.message)
        messages.append(f"{name}: {msg}" if msg else name)

    return result, messages


def iqr_outlier_summary(series, iqr_k=1.5):
    s = pd.Series(series).dropna()
    total = int(s.shape[0])
    if total == 0:
        return {
            "count": 0,
            "total": 0,
            "pct": 0.0,
            "q1": None,
            "q3": None,
            "iqr": None,
            "lower": None,
            "upper": None,
        }

    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - iqr_k * iqr
    upper = q3 + iqr_k * iqr
    mask = (s < lower) | (s > upper)

    count = int(mask.sum())
    pct = (count / total) * 100

    return {
        "count": count,
        "total": total,
        "pct": float(pct),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "lower": float(lower),
        "upper": float(upper),
    }


def iqr_bounds(series, iqr_k=1.5):
    summary = iqr_outlier_summary(series, iqr_k=iqr_k)
    return summary.get("lower"), summary.get("upper")


def _scale_std(scaler_y, std_sc):
    if std_sc is None:
        return None
    if hasattr(scaler_y, "data_max_") and hasattr(scaler_y, "data_min_"):
        scale_range = float(scaler_y.data_max_[0] - scaler_y.data_min_[0])
    elif hasattr(scaler_y, "scale_"):
        scale_range = float(scaler_y.scale_[0])
    else:
        scale_range = 1.0
    return float(std_sc) * scale_range


def forecast_future(
    model,
    scaler_X,
    scaler_y,
    series_df,
    target_col,
    create_features,
    n_future=30,
    lags=30,
    use_rolling=True,
    inverse_transform_fn=None,
):
    """
    Melakukan forecasting multi-step recursive.
    - inverse_transform_fn: jika tidak None, akan diterapkan pada pred_raw, lower, upper
    """
    history = series_df[[target_col]].copy()
    if history.empty:
        raise ValueError("series_df is empty")

    preds = []
    stds = []
    future_index = []

    for _ in range(n_future):
        df_feat = create_features(history, target_col, lags=lags, use_rolling=use_rolling)
        if df_feat.empty:
            raise ValueError("No rows after feature engineering")

        X_last = df_feat.drop(target_col, axis=1).iloc[[-1]]
        X_last_sc = scaler_X.transform(X_last)

        std_sc = None
        try:
            pred_sc, std_sc_arr = model.predict(X_last_sc, return_std=True)
            pred_sc = float(pred_sc[0])
            std_sc = float(std_sc_arr[0])
        except TypeError:
            pred_sc = float(model.predict(X_last_sc)[0])

        pred_raw = float(scaler_y.inverse_transform([[pred_sc]])[0][0])
        std_actual = _scale_std(scaler_y, std_sc)

        pred_out = pred_raw
        lower = pred_raw - 2 * std_actual if std_actual is not None else None
        upper = pred_raw + 2 * std_actual if std_actual is not None else None
        if inverse_transform_fn:
            pred_out = float(inverse_transform_fn(pred_raw))
            if lower is not None:
                lower = float(inverse_transform_fn(lower))
            if upper is not None:
                upper = float(inverse_transform_fn(upper))

        last_idx = history.index[-1]
        if isinstance(history.index, pd.DatetimeIndex):
            next_idx = last_idx + pd.Timedelta(days=1)
        else:
            next_idx = last_idx + 1

        history.loc[next_idx] = pred_raw
        future_index.append(next_idx)
        preds.append(pred_out)
        stds.append(std_actual)

    # --- PERBAIKAN: interval keyakinan untuk inverse_transform_fn ---
    if inverse_transform_fn:
        lower = [inverse_transform_fn(p - 2 * s) if s is not None else None for p, s in zip(preds, stds)]
        upper = [inverse_transform_fn(p + 2 * s) if s is not None else None for p, s in zip(preds, stds)]
    else:
        lower = [p - 2 * s if s is not None else None for p, s in zip(preds, stds)]
        upper = [p + 2 * s if s is not None else None for p, s in zip(preds, stds)]

    return pd.DataFrame({
        "Prediksi": preds,
        "Lower": lower,
        "Upper": upper,
        "Uncertainty": stds,
    }, index=future_index)


def save_forecast_plot(series_df, forecast_df, output_dir, dataset_name, target_col, horizon):
    import matplotlib.pyplot as plt

    output_dir = dataset_output_dir(output_dir, dataset_name)
    if not output_dir:
        return None

    history = series_df[target_col].iloc[-90:]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(history.index, history.values, label="Historis (90 hari)", color="#2F4B7C", linewidth=1.3)
    ax.plot(forecast_df.index, forecast_df["Prediksi"], label=f"Prediksi {horizon} hari", color="#F58518", linestyle="--", marker="o", markersize=3)

    if forecast_df["Lower"].notna().any():
        ax.fill_between(
            forecast_df.index,
            forecast_df["Lower"].astype(float),
            forecast_df["Upper"].astype(float),
            color="#F58518",
            alpha=0.2,
            label="Uncertainty (±2σ)",
        )

    ax.set_title(f"Forecast {horizon} Hari - {dataset_name}")
    ax.set_ylabel(target_col)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()

    filename = f"{slugify(dataset_name)}__forecast_{horizon}d.png"
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def apply_outlier_clipping(series, run_ctx=None):
    handling = None
    iqr_k = 1.5

    if run_ctx and isinstance(run_ctx, dict):
        cfg = run_ctx.get("outlier", {})
        handling = str(cfg.get("handling") or "").lower()
        iqr_k = cfg.get("iqr_k", iqr_k)

    if handling != "clip":
        return pd.Series(series), None

    summary = iqr_outlier_summary(series, iqr_k=iqr_k)
    lower = summary.get("lower")
    upper = summary.get("upper")
    if lower is None or upper is None:
        return pd.Series(series), None

    clipped = pd.Series(series).clip(lower=lower, upper=upper)
    meta = {
        "method": "IQR",
        "iqr_k": iqr_k,
        "lower": lower,
        "upper": upper,
    }
    return clipped, meta


def outlier_note(pct, warn_pct=5.0, high_pct=10.0):
    if pct >= high_pct:
        return "TERLALU BANYAK OUTLIER"
    if pct >= warn_pct:
        return "Cukup banyak outlier"
    return "Wajar"


def save_boxplot(series, output_path, title, ylabel):
    import matplotlib.pyplot as plt

    data = pd.Series(series).dropna()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(
        data,
        vert=True,
        patch_artist=True,
        boxprops=dict(facecolor="#4C78A8", color="#2F4B7C"),
        medianprops=dict(color="#F58518", linewidth=2),
        whiskerprops=dict(color="#2F4B7C"),
        capprops=dict(color="#2F4B7C"),
        flierprops=dict(marker="o", markersize=3, markerfacecolor="#E45756", markeredgecolor="none", alpha=0.6),
    )
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def analyze_outliers(series, dataset_name, target_name, run_ctx=None):
    method = "IQR"
    iqr_k = 1.5
    warn_pct = 5.0
    high_pct = 10.0
    output_dir = None

    if run_ctx and isinstance(run_ctx, dict):
        cfg = run_ctx.get("outlier", {})
        method = cfg.get("method", method)
        iqr_k = cfg.get("iqr_k", iqr_k)
        warn_pct = cfg.get("warn_pct", warn_pct)
        high_pct = cfg.get("high_pct", high_pct)
        output_dir = cfg.get("output_dir", output_dir)

    output_dir = dataset_output_dir(output_dir, dataset_name)

    if method.upper() != "IQR":
        method = "IQR"

    summary = iqr_outlier_summary(series, iqr_k=iqr_k)
    note = outlier_note(summary["pct"], warn_pct=warn_pct, high_pct=high_pct)

    boxplot_path = None
    if output_dir:
        ensure_dir(output_dir)
        filename = f"{slugify(dataset_name)}__{slugify(target_name)}__boxplot.png"
        boxplot_path = os.path.join(output_dir, filename)
        save_boxplot(series, boxplot_path, f"Boxplot {dataset_name} - {target_name}", target_name)

    return {
        "dataset": dataset_name,
        "target": target_name,
        "method": method,
        "iqr_k": iqr_k,
        "count": summary["count"],
        "total": summary["total"],
        "pct": summary["pct"],
        "lower": summary["lower"],
        "upper": summary["upper"],
        "note": note,
        "boxplot_path": boxplot_path,
    }


def build_scalers(scaler_type):
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    scaler = str(scaler_type or "minmax").lower()
    if scaler == "standard":
        return StandardScaler(), StandardScaler()
    return MinMaxScaler(), MinMaxScaler()


def eda_summary(series):
    s = pd.Series(series)
    total = int(s.shape[0])
    missing = int(s.isna().sum())
    missing_pct = (missing / total) * 100 if total else 0.0
    desc = s.dropna().describe(percentiles=[0.25, 0.5, 0.75])

    def pick(key):
        return float(desc.get(key)) if key in desc else None

    return {
        "count": int(desc.get("count", 0)),
        "missing": missing,
        "missing_pct": float(missing_pct),
        "mean": pick("mean"),
        "std": pick("std"),
        "min": pick("min"),
        "q1": pick("25%"),
        "median": pick("50%"),
        "q3": pick("75%"),
        "max": pick("max"),
    }


def save_histogram(series, output_path, title, xlabel):
    import matplotlib.pyplot as plt

    data = pd.Series(series).dropna()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(data, bins=40, color="#4C78A8", edgecolor="white", alpha=0.85)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_timeseries(series, output_path, title, ylabel):
    import matplotlib.pyplot as plt

    data = pd.Series(series).dropna()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(data.index, data.values, color="#2F4B7C", linewidth=1.2)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def analyze_eda(series, dataset_name, target_name, run_ctx=None):
    output_dir = None
    enable_plots = True

    if run_ctx and isinstance(run_ctx, dict):
        cfg = run_ctx.get("eda", {})
        output_dir = cfg.get("output_dir", output_dir)
        enable_plots = cfg.get("enable_plots", enable_plots)

    output_dir = dataset_output_dir(output_dir, dataset_name)

    summary = eda_summary(series)
    hist_path = None
    ts_path = None

    if output_dir and enable_plots:
        ensure_dir(output_dir)
        hist_name = f"{slugify(dataset_name)}__{slugify(target_name)}__hist.png"
        ts_name = f"{slugify(dataset_name)}__{slugify(target_name)}__series.png"
        hist_path = os.path.join(output_dir, hist_name)
        ts_path = os.path.join(output_dir, ts_name)
        save_histogram(series, hist_path, f"Histogram {dataset_name} - {target_name}", target_name)
        save_timeseries(series, ts_path, f"Time Series {dataset_name} - {target_name}", target_name)

    return {
        "dataset": dataset_name,
        "target": target_name,
        "count": summary["count"],
        "missing": summary["missing"],
        "missing_pct": summary["missing_pct"],
        "mean": summary["mean"],
        "std": summary["std"],
        "min": summary["min"],
        "q1": summary["q1"],
        "median": summary["median"],
        "q3": summary["q3"],
        "max": summary["max"],
        "hist_path": hist_path,
        "series_path": ts_path,
    }


def compute_metrics(y_true, y_pred):
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred, index=y_true.index)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    denom = y_true.replace(0, np.nan)
    mape_series = (y_true - y_pred).abs() / denom
    mape = float(mape_series.mean(skipna=True) * 100)

    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        r2 = 0.0
    else:
        r2 = 1 - (np.sum((y_true - y_pred) ** 2) / ss_tot)

    return {
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "MAPE(%)": round(float(mape), 4),
        "R2": round(float(r2), 4),
    }


def fit_diagnosis(model, y_train_true, y_train_pred, y_test_true, y_test_pred, max_iter=200):

    train_metrics = compute_metrics(y_train_true, y_train_pred)
    test_metrics  = compute_metrics(y_test_true, y_test_pred)

    train_mape = train_metrics["MAPE(%)"]
    test_mape  = test_metrics["MAPE(%)"]

    gap = abs(test_mape - train_mape)

    if gap < 5:
        status = "[OK] FIT (selisih < 5%)"
    elif gap < 10:
        status = "[WARN] Sedikit Overfitting"
    else:
        status = "[ERR] OVERFITTING"

    n_iter = getattr(model, "n_iter_", None)

    if n_iter is not None:
        if n_iter >= max_iter:
            conv_status = "[WARN] belum konvergen"
        else:
            conv_status = "konvergen [OK]"
    else:
        conv_status = "-"

    return {
        "train_mape": round(train_mape, 2),
        "test_mape": round(test_mape, 2),
        "gap": round(gap, 2),
        "status": status,
        "n_iter": n_iter,
        "conv_status": conv_status,
        "alpha": getattr(model, "alpha_", None),
        "lambda": getattr(model, "lambda_", None),
    }


def save_prediction_plots(y_test, y_pred, output_dir, dataset_name, target_name):
    import matplotlib.pyplot as plt

    output_dir = dataset_output_dir(output_dir, dataset_name)
    if not output_dir:
        return None

    y_test = pd.Series(y_test)
    y_pred = pd.Series(y_pred, index=y_test.index)

    ts_name = f"{slugify(dataset_name)}__{slugify(target_name)}__pred_timeseries.png"
    sc_name = f"{slugify(dataset_name)}__{slugify(target_name)}__pred_scatter.png"

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(y_test.index, y_test.values, label="Aktual", color="#2F4B7C", linewidth=1.4)
    ax.plot(y_pred.index, y_pred.values, label="Prediksi", color="#F58518", linewidth=1.2, linestyle="--")
    ax.set_title(f"Aktual vs Prediksi - {dataset_name}", fontsize=11)
    ax.set_ylabel(target_name)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, ts_name), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(y_test.values, y_pred.values, alpha=0.5, s=12, color="#4C78A8")
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="#E45756", linewidth=1.5)
    ax.set_title("Scatter Aktual vs Prediksi", fontsize=11)
    ax.set_xlabel("Aktual")
    ax.set_ylabel("Prediksi")
    ax.grid(axis="both", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, sc_name), dpi=150)
    plt.close(fig)

    return {
        "timeseries_path": os.path.join(output_dir, ts_name),
        "scatter_path": os.path.join(output_dir, sc_name),
    }


def save_outlier_compare_plot(metrics_base, metrics_outlier, output_dir, dataset_name, target_name):
    import numpy as np
    import matplotlib.pyplot as plt

    output_dir = dataset_output_dir(output_dir, dataset_name)
    if not output_dir:
        return None

    def to_num(val):
        if isinstance(val, (int, float, np.floating)):
            return float(val)
        return np.nan

    def pick(metrics, key):
        if isinstance(metrics, dict):
            return to_num(metrics.get(key))
        return np.nan

    metrics_cfg = [
        ("MAE", "MAE"),
        ("RMSE", "RMSE"),
        ("MAPE", "MAPE(%)"),
        ("R2", "R2"),
    ]

    labels = ["Tanpa Penanganan Outlier", "Dengan Penanganan Outlier"]
    colors = ["#4C78A8", "#F58518"]

    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
    axes = axes.ravel()

    for ax, (title, key) in zip(axes, metrics_cfg):
        vals = [pick(metrics_base, key), pick(metrics_outlier, key)]
        ax.bar(labels, vals, color=colors)
        ax.set_title(f"{title} - {dataset_name}", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    filename = f"{slugify(dataset_name)}__{slugify(target_name)}__outlier_compare.png"
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def save_metrics_compare_plot(compare_rows, output_dir, filename="metrics_compare.png"):
    import numpy as np
    import matplotlib.pyplot as plt

    if not compare_rows:
        return None

    ensure_dir(output_dir)
    datasets = [r.get("Dataset", "-") for r in compare_rows]

    def to_num(v):
        if isinstance(v, (int, float, np.floating)):
            return float(v)
        return np.nan

    metrics = [
        ("MAE", "MAE Raw", "MAE Scaled"),
        ("RMSE", "RMSE Raw", "RMSE Scaled"),
        ("MAPE", "MAPE Raw", "MAPE Scaled"),
        ("R2", "R2 Raw", "R2 Scaled"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes = axes.ravel()

    x = np.arange(len(datasets))
    width = 0.36

    for ax, (label, raw_key, scaled_key) in zip(axes, metrics):
        raw_vals = [to_num(r.get(raw_key)) for r in compare_rows]
        scaled_vals = [to_num(r.get(scaled_key)) for r in compare_rows]

        ax.bar(x - width / 2, raw_vals, width, label="Raw", color="#4C78A8")
        ax.bar(x + width / 2, scaled_vals, width, label="Scaled", color="#F58518")
        ax.set_title(f"{label} (Raw vs Scaled)", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=25, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.35)

    axes[0].legend(loc="upper right")
    fig.tight_layout()

    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path