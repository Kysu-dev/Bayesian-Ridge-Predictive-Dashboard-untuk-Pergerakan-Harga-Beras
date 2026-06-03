import sys
import os
import joblib
import pandas as pd

# Pastikan script bisa mengimpor modul dari folder utama
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import daily_climate

def main():
    print("================================================================")
    print(" MEMULAI PELATIHAN MODEL BERAS (PIHPS) DENGAN CONFIG TERBAIK")
    print("================================================================")
    print("Konfigurasi yang digunakan:")
    print(" - Lags      : 7")
    print(" - Rolling   : False")
    print(" - Model_tol : 0.0001")
    print(" - Outlier   : none")
    print("----------------------------------------------------------------\n")

    # Buat direktori 'model' dan 'grafik' di dalam folder project
    model_dir = os.path.join(os.path.dirname(__file__), 'model')
    grafik_dir = os.path.join(model_dir, 'grafik_training')
    os.makedirs(grafik_dir, exist_ok=True)

    run_ctx = {
        "model": {"tol": 0.0001},
        "outlier": {"handling": "none"},
        "split": {"ratio": 0.8},
        "pred": {
            "enable_plots": True, 
            "output_dir": grafik_dir
        }
    }

    # Eksekusi fungsi run dari daily_climate.py (dataset beras)
    result = daily_climate.run(run_ctx)

    print("--- HASIL EVALUASI MODEL ---")
    print(f"MAPE(%) : {result['MAPE(%)']:.4f}%")
    print(f"R2 Score: {result['R2']:.4f}")
    print(f"MAE     : {result['MAE']:.4f}")
    print(f"RMSE    : {result['RMSE']:.4f}")

    # Buat direktori 'model' di dalam folder project
    model_dir = os.path.join(os.path.dirname(__file__), 'model')
    os.makedirs(model_dir, exist_ok=True)
    
    # Simpan model dan scaler menggunakan joblib
    joblib.dump(result['_model'], os.path.join(model_dir, 'bayesian_ridge_model.pkl'))
    joblib.dump(result['_scaler_X'], os.path.join(model_dir, 'scaler_X.pkl'))
    joblib.dump(result['_scaler_y'], os.path.join(model_dir, 'scaler_y.pkl'))
    
    # Simpan urutan fitur (penting untuk saat inferensi / prediksi masa depan)
    X_cols = result['_X'].columns.tolist()
    joblib.dump(X_cols, os.path.join(model_dir, 'feature_columns.pkl'))

    print(f"\n[SUKSES] Model dan Scaler berhasil disimpan di direktori: {model_dir}")
    print("Sekarang Anda bisa mengintegrasikannya langsung ke dalam app.py!")

if __name__ == "__main__":
    main()
