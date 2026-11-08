# Prediksi Biaya Medis Asuransi Menggunakan Deep Neural Network

**UTS Pembelajaran Mendalam — Kelompok 6**  
Semester Ganjil, Tahun Akademik 2026/2027

Proyek ini menggunakan *Deep Neural Network* (DNN) untuk memperkirakan biaya medis asuransi berdasarkan karakteristik individu. Model yang dikembangkan dibandingkan dengan Linear Regression dan Random Forest agar hasil DNN dapat dievaluasi terhadap metode regresi lain. Proses pengolahan data, pelatihan, dan evaluasi terdokumentasi di notebook, sedangkan aplikasi Streamlit digunakan untuk mencoba prediksi secara langsung.

**Tautan proyek:** [Aplikasi Streamlit](https://medicalcostpredictiondnn-jnbqvke2tgpebqlryh8xgp.streamlit.app/) | [Video demonstrasi](https://drive.google.com/file/d/17TalNHyt0PP1aYiI1nqoZFQkRoaSFuSp/view?usp=sharing) | [Laporan UTS (PDF)](docs/Laporan_UTS_Pembelajaran_Mendalam_Kelompok_6.pdf) | [Notebook eksperimen](notebooks/medical_cost_dnn.ipynb)

## Anggota kelompok

| Nama | NIM | Pembagian tugas |
|---|---|---|
| Delarosa Butar Butar | 231111254 | Persiapan dataset, EDA, dan visualisasi |
| Ryan Peace Simbolon | 231111190 | Pengembangan kode pemodelan dan aplikasi bersama Boris |
| Grace Debora Sitompul | 231111836 | Eksperimen arsitektur DNN, analisis hasil, dan pembahasan laporan |
| Saskia Anggi Pratiwi | 231110909 | Pengujian aplikasi, dokumentasi, dan video demo |
| Boris Sanjaya Siregar | 231111783 | Pengembangan kode bersama Ryan, integrasi aplikasi, dan deployment |

## Dataset

Data proyek terdapat pada [`data/medical_insurance.csv`](data/medical_insurance.csv). Berkas berisi 1.338 observasi dan delapan kolom sebelum pembersihan. Setelah satu baris duplikat dihapus, terdapat 1.337 observasi yang digunakan untuk eksperimen. Tidak ditemukan nilai kosong pada pemeriksaan data.

| Variabel | Keterangan |
|---|---|
| `age` | Usia |
| `gender` | Jenis kelamin |
| `bmi` | Indeks massa tubuh |
| `children` | Jumlah anak |
| `discount_eligibility` | Status merokok (`yes` = merokok, `no` = tidak merokok) |
| `region` | Wilayah tempat tinggal |
| `expenses` | Biaya medis, sebagai target regresi |
| `premium` | Nilai yang diturunkan dari `expenses`; tidak dipakai sebagai prediktor |

Nama `discount_eligibility` dipertahankan pada data dan fitur internal agar sesuai dengan artefak model tersimpan. Pada dataset publik yang menjadi rujukan, variabel tersebut bernama `smoker`; jadi, variabel ini **bukan** kelayakan diskon. Nama `expenses` berpadanan dengan `charges` pada data sumber.

Data dasar proyek cocok dengan [Medical Cost Personal Datasets di Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance), setelah penyesuaian nama variabel dan pembulatan angka. Tautan Kaggle digunakan sebagai rujukan dataset publik, bukan sebagai pernyataan mengenai tautan yang pertama kali dipakai untuk mengunduh berkas proyek.

## Pengolahan data

Tahapan pengolahan pada [`notebooks/medical_cost_dnn.ipynb`](notebooks/medical_cost_dnn.ipynb) meliputi:

1. Menghapus satu observasi duplikat dan memisahkan `expenses` sebagai target.
2. Mengeluarkan `premium` dari masukan karena nilainya merupakan turunan target.
3. Melakukan *one-hot encoding* pada `gender`, `discount_eligibility`, dan `region`, dengan `drop_first=True`.
4. Membagi data menjadi 935 latih, 201 validasi, dan 201 uji menggunakan `random_state=42`.
5. Menggunakan `StandardScaler` pada fitur `age`, `bmi`, dan `children`. Objek scaler hanya di-*fit* pada data latih.
6. Melatih model dengan target `log1p(expenses)` dan mengembalikan hasil prediksi ke skala biaya menggunakan `expm1`.

Enam atribut awal menghasilkan delapan fitur numerik berikut:

```text
age, bmi, children, gender_male, discount_eligibility_yes,
region_northwest, region_southeast, region_southwest
```

Daftar dan urutan fitur disimpan dalam [`models/preprocessor.joblib`](models/preprocessor.joblib) bersama objek scaler. Aplikasi menggunakan informasi tersebut untuk menyiapkan data baru dengan cara yang sama seperti saat pelatihan.

## Pemodelan DNN

Model utama menggunakan jaringan *Multilayer Perceptron* berbasis Keras dengan keluaran regresi. Notebook menguji delapan susunan dan pengaturan jaringan, termasuk perubahan jumlah neuron, jumlah lapisan, *Dropout*, optimizer, dan fungsi aktivasi.

Konfigurasi terpilih berdasarkan RMSE validasi adalah **Shallow (16)**, dengan delapan input, satu lapisan `Dense(16)` beraktivasi ReLU, dan satu neuron keluaran linear. Konfigurasi ini mempunyai 161 parameter. Pelatihan menggunakan Adam (`learning_rate=0.001`), MSE pada target log, `batch_size=32`, serta `EarlyStopping` dengan `patience=25` dan pemulihan bobot terbaik. Batas maksimum pelatihan adalah 800 epoch.

Hasil delapan konfigurasi yang tersimpan pada notebook:

| Konfigurasi | MAE validasi | RMSE validasi | R² validasi |
|---|---:|---:|---:|
| Shallow (16) | 2.989,057 | **5.183,080** | 0,843 |
| 3 layer + Dropout, SGD | 2.800,050 | 5.244,905 | 0,839 |
| 3 layer + Dropout 0,1 | 3.003,861 | 5.519,690 | 0,822 |
| 2 layer (32–16) | 3.233,663 | 5.595,109 | 0,817 |
| Deep (128–64–32–16) + D0,2 | 3.035,360 | 5.804,232 | 0,803 |
| 3 layer + Dropout, RMSprop | 3.355,071 | 5.853,420 | 0,800 |
| 3 layer (64–32–16) | 3.579,636 | 7.405,933 | 0,679 |
| 3 layer + Dropout, tanh | 4.499,881 | 7.454,930 | 0,675 |

Pemilihan dilakukan pada data validasi, bukan data uji. Model DNN awal dalam notebook memiliki susunan berbeda dari model Shallow (16) yang dipilih untuk aplikasi. Penjelasan arsitektur dan hasil pelatihan dapat dilihat lebih lanjut pada [laporan UTS](docs/Laporan_UTS_Pembelajaran_Mendalam_Kelompok_6.pdf).

## Hasil pengujian

Evaluasi akhir menggunakan 201 observasi data uji. MAE dan RMSE dihitung setelah prediksi dikembalikan ke skala `expenses`.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 3.570,21 | 7.230,61 | 0,694 |
| Random Forest | **2.082,56** | **4.583,21** | **0,877** |
| DNN Shallow (16) | 3.086,64 | 5.938,26 | 0,793 |

Random Forest memberikan hasil terbaik pada pengujian ini. DNN menghasilkan galat lebih kecil daripada Linear Regression, tetapi belum lebih baik daripada Random Forest. Hasil ini menunjukkan pentingnya membandingkan DNN dengan baseline, terutama pada data tabular yang jumlah observasinya relatif terbatas.

### Pemeriksaan dua kasus prediksi

Dua contoh dari data uji digunakan untuk memperlihatkan perbedaan hasil model pada kasus dengan galat kecil dan galat besar.

| Keterangan | Kasus galat kecil | Kasus galat besar |
|---|---:|---:|
| Usia | 47 | 45 |
| BMI | 26,60 | 30,40 |
| Jumlah anak | 2 | 0 |
| Gender | female | male |
| Status merokok | no | yes |
| Region | northeast | southeast |
| Nilai aktual | 9.715,84 | 62.592,87 |
| Hasil pada aplikasi | 9.721,62 | 30.043,66 |
| Galat relatif | sekitar 0,06% | sekitar 52,00% |

Grafik perbandingan dan hasil pemeriksaan data uji dapat dilihat pada [gambar dua kasus](results/Bukti_Dua_Kasus_Prediksi_DNN_Kelompok_6.png) dan [CSV 201 prediksi](results/Hasil_201_Prediksi_Data_Uji_Kelompok_6.csv). Screenshot pengujian tersedia pada [kasus galat kecil](results/streamlit_kasus_galat_kecil.png) dan [kasus galat besar](results/streamlit_kasus_galat_besar.png).

Prediksi kasus galat besar dalam CSV tercatat sekitar 30.043,63, sedangkan screenshot aplikasi menunjukkan 30.043,66. Nilai tersebut dipertahankan sesuai sumber masing-masing; selisih pembulatan dan komputasinya tidak digunakan untuk mengubah hasil eksperimen utama.

## Pemeriksaan kebocoran target

Kolom `premium` memiliki hubungan langsung dengan `expenses`: pada seluruh 1.338 baris awal, perbandingan `premium/expenses` bernilai 1%, 2%, atau 4%. Oleh karena itu, `premium` tidak digunakan sebagai masukan DNN.

Pengaruh penggunaan fitur tersebut diperiksa lebih lanjut melalui percobaan terpisah dengan Linear Regression dan Random Forest, menggunakan pembagian data yang sama. Hasilnya tersimpan di [`results/Hasil_Ablasi_Kelompok_6.csv`](results/Hasil_Ablasi_Kelompok_6.csv).

| Model | Kondisi | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| Linear Regression | Tanpa `premium` | 3.570,21 | 7.230,61 | 0,694 |
| Linear Regression | Dengan `premium` | 4.239,16 | 11.865,72 | 0,175 |
| Random Forest | Tanpa `premium` | 2.082,56 | 4.583,21 | 0,877 |
| Random Forest | Dengan `premium` | 359,00 | 1.523,66 | 0,986 |

Performa Random Forest yang tampak lebih baik ketika `premium` disertakan tidak dapat dianggap sebagai hasil prediksi yang valid, karena model memperoleh informasi yang berasal dari target. Eksperimen ablasi ini hanya melibatkan dua baseline dan tidak mengubah bobot atau hasil DNN utama. Skripnya tersedia pada [`scripts/Analisis_Ablasi_Data_Leakage_Kelompok_6.py`](scripts/Analisis_Ablasi_Data_Leakage_Kelompok_6.py).

## Aplikasi Streamlit

Aplikasi berada pada [`app/app.py`](app/app.py) dan dapat digunakan melalui [Streamlit Community Cloud](https://medicalcostpredictiondnn-jnbqvke2tgpebqlryh8xgp.streamlit.app/). Formulir menerima usia, BMI, jumlah anak, jenis kelamin, status merokok, dan region. Setelah tombol **Prediksi** ditekan, aplikasi menyiapkan delapan fitur sesuai preprocessor, memanggil model DNN tersimpan, lalu menampilkan hasil dalam skala `expenses`.

Alur aplikasi merupakan *inferensi*, sehingga model tidak dilatih ulang setiap kali pengguna menekan tombol. Nilai yang ditampilkan adalah estimasi berdasarkan dataset eksperimen, bukan nilai tagihan medis yang pasti.

## Struktur berkas

```text
Medical_Cost_Prediction_DNN/
├── app/
│   └── app.py
├── data/
│   └── medical_insurance.csv
├── docs/
│   └── Laporan_UTS_Pembelajaran_Mendalam_Kelompok_6.pdf
├── models/
│   ├── dnn_medical_cost.keras
│   └── preprocessor.joblib
├── notebooks/
│   └── medical_cost_dnn.ipynb
├── results/
│   ├── Bukti_Dua_Kasus_Prediksi_DNN_Kelompok_6.png
│   ├── Hasil_201_Prediksi_Data_Uji_Kelompok_6.csv
│   ├── Hasil_Ablasi_Kelompok_6.csv
│   ├── streamlit_kasus_galat_besar.png
│   └── streamlit_kasus_galat_kecil.png
├── scripts/
│   ├── Analisis_Ablasi_Data_Leakage_Kelompok_6.py
│   └── Verifikasi_Dua_Kasus_Prediksi_Kelompok_6.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Menjalankan proyek

Gunakan Python 3.11. Jalankan perintah berikut dari terminal:

```bash
git clone https://github.com/boris-sanjaya/Medical_Cost_Prediction_DNN.git
cd Medical_Cost_Prediction_DNN
```

Buat lingkungan virtual dengan Python 3.11:

- Windows PowerShell: `py -3.11 -m venv .venv`, kemudian `.\.venv\Scripts\Activate.ps1`
- Linux/macOS: `python3.11 -m venv .venv`, kemudian `source .venv/bin/activate`

Kemudian pasang dependensi dan jalankan aplikasi:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app/app.py
```

Setelah server berjalan, buka `http://localhost:8501` di browser. Model dan preprocessor sudah tersedia di folder `models/`, sehingga pelatihan ulang tidak diperlukan untuk menggunakan aplikasi.

Untuk memeriksa kembali hasil prediksi pada data uji, jalankan:

```bash
python scripts/Verifikasi_Dua_Kasus_Prediksi_Kelompok_6.py
```

Untuk menjalankan percobaan ablasi dua model baseline:

```bash
python scripts/Analisis_Ablasi_Data_Leakage_Kelompok_6.py
```

Kedua skrip dapat menghasilkan ulang berkas di folder `results/`. Sebaiknya gunakan salinan proyek apabila ingin mempertahankan file hasil yang telah disimpan. Notebook utama berada di [`notebooks/medical_cost_dnn.ipynb`](notebooks/medical_cost_dnn.ipynb) dan memuat kode beserta output eksperimen.

## Deployment dan dokumentasi

Aplikasi dipublikasikan melalui Streamlit Community Cloud dengan repository GitHub ini sebagai sumber, branch `main`, dan file utama `app/app.py`. Dependensi dibaca dari `requirements.txt`, sedangkan model DNN dan preprocessor diambil dari folder `models/`.

Dokumentasi proyek tersedia melalui tautan berikut:

- [Aplikasi prediksi biaya medis](https://medicalcostpredictiondnn-jnbqvke2tgpebqlryh8xgp.streamlit.app/)
- [Video demonstrasi aplikasi](https://drive.google.com/file/d/17TalNHyt0PP1aYiI1nqoZFQkRoaSFuSp/view?usp=sharing)
- [Laporan UTS Kelompok 6 (PDF)](docs/Laporan_UTS_Pembelajaran_Mendalam_Kelompok_6.pdf)
- [Notebook eksperimen DNN](notebooks/medical_cost_dnn.ipynb)
- [Hasil pengujian dan visualisasi](results/)

Video menunjukkan penggunaan antarmuka dan contoh keluaran aplikasi. Penjelasan tentang pengolahan dataset, delapan eksperimen DNN, kontrol kebocoran target, serta konfigurasi deployment dicantumkan dalam notebook, kode, dan laporan.

## Referensi

- [Medical Cost Personal Datasets — Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- [Dokumentasi Keras](https://keras.io/)
- [Dokumentasi evaluasi model scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Dokumentasi Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud)

Kode proyek menggunakan [lisensi MIT](LICENSE). Ketentuan penggunaan dataset mengikuti sumber data yang bersangkutan.
