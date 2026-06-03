# 🌾 RicePrice Forecaster: Bayesian Ridge Predictive Dashboard

![Banner](https://img.shields.io/badge/Machine%20Learning-Bayesian%20Ridge-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Web%20App-Flask-green?style=for-the-badge&logo=flask)
![Bootstrap](https://img.shields.io/badge/UI-Modern-purple?style=for-the-badge)

Sistem Prediksi Harga Komoditas Beras berbasis Web interaktif menggunakan algoritma **Bayesian Ridge Regression**. Aplikasi ini dibangun untuk memprediksi tren harga beras di masa depan dan menganalisis metrik performa secara mendalam. Dilengkapi dengan fitur **Online Learning / Dynamic Ground-Truth Injection**, memungkinkan model untuk dilatih ulang (*retrain*) secara dinamis seiring masuknya data aktual terbaru.

---

## 👥 Tim Pengembang (Kelompok D8)

Proyek ini dikembangkan untuk memenuhi Tugas Besar Praktikum IFB-310 Machine Learning oleh **Kelompok D8**:

| NIM | Nama |
| :--- | :--- |
| **15-2023-147** | Rizki Hidayatulloh |
| **15-2023-167** | Raelqiansyah P.d |
| **15-2023-180** | Ai Resti Saniah |
| **15-2023-183** | Rafina Az Zahra |
| **15-2023-186** | Difie Anggely |

---

## ✨ Fitur Utama

- **Dashboard Interaktif**: Menampilkan perbandingan harga aktual vs prediksi terbaru dalam *card* metrik yang elegan.
- **Prediksi Akurat & Fleksibel**: Menggunakan *Autoregressive Forecasting* dengan lags=7 dan metode *Bayesian Ridge* yang tahan terhadap masalah multikolinearitas.
- **Ground-Truth Data Injection**: Fitur bagi administrator/pengguna untuk menyuntikkan (inject) data harga aktual bulan terbaru. Model secara otomatis melakukan *retraining* dan memperbarui prediksinya.
- **Visualisasi Historis & Metrik**: Chart visual (*Line charts*, *Scatter Plots*) untuk menganalisis pergerakan tren masa lalu dan sisa (*residual*) dari algoritma Machine Learning.
- **Model Info & Evaluasi**: Halaman khusus untuk memantau nilai MAPE, $R^2$, MAE, dan status model untuk menjaga kredibilitas hasil.

---

## 📂 Struktur Proyek

```text
├── dataset/
│   ├── komoditas_beras_2022_2026.csv   # Dataset awal PIHPS
│   └── komoditas_beras_tambahan.csv    # Dataset hasil injeksi pengguna (Online Learning)
├── project/
│   ├── app.py                          # Main Flask App
│   ├── train_beras.py                  # Script untuk melatih ulang (retrain) model beras
│   ├── requirements.txt                # Dependensi Python
│   ├── model/                          # Tempat penyimpanan file .pkl dari Bayesian Ridge
│   ├── static/                         # Assets CSS, JS, dan images
│   └── templates/                      # File HTML (Jinja2)
├── daily_climate.py                    # Script pemrosesan dataset dan pemodelan dasar
├── analysis_utils.py                   # Fungsi helper untuk kalkulasi evaluasi (MAPE, R2, dll)
└── README.md
```

---

## 🚀 Panduan Instalasi dan Penggunaan

### 1. Prasyarat (*Prerequisites*)
Pastikan Python 3.8+ sudah terinstal di perangkat Anda. Disarankan untuk menggunakan Virtual Environment.

### 2. Instalasi Dependensi
Buka terminal/CMD, arahkan ke folder utama proyek, lalu ketikkan perintah berikut:
```bash
# Pindah ke direktori project web
cd project

# Install semua library yang dibutuhkan
pip install -r requirements.txt
```

### 3. Cara Menjalankan Web App
Setelah semua dependensi terinstal, jalankan server Flask dengan perintah:
```bash
python app.py
```
Aplikasi akan berjalan di `http://127.0.0.1:5000/`. Buka alamat tersebut di *browser* Anda.

### 4. Cara Melatih Ulang Model Secara Manual (Opsional)
Meskipun aplikasi sudah dilengkapi dengan fitur injeksi data dari *frontend*, Anda juga bisa melakukan inisialisasi pelatihan model murni (dari awal) dengan cara menjalankan script ini dari *root* folder:
```bash
python project/train_beras.py
```
Ini akan mengevaluasi dataset secara komprehensif, mengembalikan metrik evaluasinya, lalu menyimpan objek *Model* (`.pkl`) kembali ke folder `project/model/`.

---

## 📊 Teknologi yang Digunakan
- **Backend/ML**: Python, Flask, Scikit-Learn, Pandas, Numpy, Joblib.
- **Frontend**: HTML5, CSS3, JavaScript, Jinja2, Chart.js (Visualisasi interaktif).

> Dibuat dengan 💻 dan ☕ oleh **Kelompok D8** - 2026
