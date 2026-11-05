# Prediksi Biaya Medis Menggunakan Deep Neural Network dengan Perbandingan Model Baseline

Proyek UTS Deep Learning (Kelompok 6): memprediksi biaya medis asuransi (`expenses`) menggunakan **Deep Neural Network (DNN) regresi** dan membandingkannya dengan model baseline (Linear Regression dan Random Forest).

## Anggota Kelompok
| Nama | NIM | Peran |
|---|---|---|
| Delarosa Butar Butar | 231111254 | Penulisan laporan (dataset dan preprocessing), verifikasi EDA dan grafik |
| Ryan Peace Simbolon | 231111190 | Pengembangan model DNN dan baseline, aplikasi demo, dokumentasi kode |
| Grace Debora Sitompul | 231111836 | Penulisan laporan (arsitektur dan eksperimen DNN), analisis tabel eksperimen |
| Saskia Anggi Pratiwi | 231110909 | Pengujian repository dari awal, pembuatan video demo, pengecekan link |
| Boris Sanjaya Siregar | 231111783 | Penyusunan dan penyatuan laporan teknis, deployment aplikasi ke Streamlit |

## Dataset
- File: `data/medical_insurance.csv` (1.338 baris, 8 kolom)
- Fitur: `age`, `gender`, `bmi`, `children`, `discount_eligibility`, `region`
- Target: `expenses` (biaya medis)
- Kolom `premium` **tidak dipakai** karena data leakage: nilainya selalu 1%, 2%, atau 4% dari `expenses`.

## Metode
1. **Preprocessing:** hapus data duplikat, one-hot encoding fitur kategorikal, transformasi `log1p` pada target (distribusi miring ke kanan), split 70% train / 15% validation / 15% test, dan `StandardScaler` yang hanya di-fit pada data train.
2. **Baseline:** Linear Regression dan Random Forest.
3. **DNN (Keras):** hidden layer ReLU, output linear, optimizer Adam, loss MSE, EarlyStopping, dan Dropout.
4. **Eksperimen arsitektur:** 8 konfigurasi (jumlah layer, dropout, optimizer, fungsi aktivasi) dibandingkan pada validation set.
5. **Evaluasi akhir** pada test set dengan MAE, RMSE, dan R².

## Hasil (test set)
| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 3570.21 | 7230.61 | 0.694 |
| Random Forest | 2082.46 | 4583.14 | 0.877 |
| DNN (terbaik) | 3086.64 | 5938.26 | 0.793 |

DNN mengungguli Linear Regression, tetapi Random Forest memberi error terendah. Pada data tabular berukuran kecil (sekitar 1,3 ribu baris) dengan interaksi antar fitur yang kuat, model berbasis pohon umumnya lebih sesuai daripada DNN. Konfigurasi DNN terbaik pada validation adalah arsitektur sederhana, yang menunjukkan bahwa model yang lebih dalam cenderung overfitting pada dataset sekecil ini.

## Struktur Repository
```
├── app/app.py                      # aplikasi demo Streamlit
├── data/medical_insurance.csv      # dataset
├── models/                         # model DNN dan preprocessor tersimpan
├── notebooks/medical_cost_dnn.ipynb
├── requirements.txt
└── README.md
```

## Cara Menjalankan
Python yang digunakan: **3.11**.

```bash
git clone https://github.com/boris-sanjaya/Medical_Cost_Prediction_DNN.git
cd Medical_Cost_Prediction_DNN
pip install -r requirements.txt
streamlit run app/app.py
```
Aplikasi terbuka di `http://localhost:8501`. Model sudah tersimpan di folder `models/`, sehingga tidak perlu melatih ulang. Untuk melatih ulang, jalankan `notebooks/medical_cost_dnn.ipynb` dengan **Run All** (hasil dapat berbeda sedikit antar komputer).

## Demo
- Aplikasi (sudah di-deploy): https://medicalcostpredictiondnn-jnbqvke2tgpebqlryh8xgp.streamlit.app/
- Video demo dan tahapan deployment: <isi link video>

## Lisensi
MIT, lihat berkas `LICENSE`.