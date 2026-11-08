"""Verifikasi dua kasus prediksi regresi dari model DNN Kelompok 6.

Tidak melakukan fit, compile, atau pelatihan ulang model. Hanya memuat
model/preprocessor tersimpan dan melakukan inferensi pada data uji yang sama.

Penggunaan (dari root repository):
    python scripts/Bukti_Prediksi_Benar_Salah_Kelompok_6.py

Opsional: berikan direktori proyek sebagai argumen pertama.

Keluaran: PNG visualisasi dan CSV 201 prediksi uji; ringkasan dicetak di terminal.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

# Tentukan backend Keras sebelum import keras.
if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "tensorflow" if importlib.util.find_spec("tensorflow") else "torch"

import joblib
import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

BASE = Path(sys.argv[1]).expanduser().resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
if not (BASE / "models/dnn_medical_cost.keras").exists():
    raise FileNotFoundError(f"Folder proyek tidak sesuai: {BASE}")
OUT = BASE / "results"
OUT.mkdir(parents=True, exist_ok=True)

# Samakan urutan operasi dan indeks dengan notebook utama.
data = pd.read_csv(BASE / "data/medical_insurance.csv")
data = data.drop_duplicates().reset_index(drop=True)
encoded = pd.get_dummies(
    data.drop(columns=["premium"]),
    columns=["gender", "discount_eligibility", "region"],
    drop_first=True,
    dtype=int,
)
X = encoded.drop(columns=["expenses"])
y = encoded["expenses"]
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, random_state=42)
X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=42)

# WAJIB memakai preprocessor tersimpan, bukan memasang scaler baru.
prep = joblib.load(BASE / "models/preprocessor.joblib")
X_te_s = X_te.loc[:, prep["columns"]].astype(float).copy()
X_te_s.loc[:, prep["num_cols"]] = prep["scaler"].transform(X_te_s[prep["num_cols"]])
model = keras.models.load_model(BASE / "models/dnn_medical_cost.keras", compile=False)
y_pred = np.expm1(model.predict(X_te_s.astype(float), verbose=0).ravel())

# Simpan semua hasil uji (201 baris) untuk audit bila diperlukan.
results = data.loc[X_te.index, ["age", "gender", "bmi", "children", "discount_eligibility", "region", "expenses"]].copy()
results.insert(0, "indeks_setelah_pembersihan", results.index)
results["prediksi"] = y_pred
results["galat_absolut"] = (results["expenses"] - results["prediksi"]).abs()
results["galat_persen"] = 100 * results["galat_absolut"] / results["expenses"]
results.sort_values("indeks_setelah_pembersihan").to_csv(OUT / "Hasil_201_Prediksi_Data_Uji_Kelompok_6.csv", index=False)

# Dua baris yang benar-benar berasal dari bagian data uji (index 275 dan 1299).
selected = results.loc[[275, 1299]]
assert all(i in X_te.index for i in (275, 1299)), "Indeks data uji berubah; periksa split"
mae = mean_absolute_error(y_te, y_pred)
rmse = np.sqrt(mean_squared_error(y_te, y_pred))
r2 = r2_score(y_te, y_pred)

report = [
    "BUKTI PREDIKSI DATA UJI - DNN KELOMPOK 6",
    "Evaluasi regresi: dua contoh dengan galat kecil dan galat besar.",
    f"Jumlah data uji: {len(X_te)}; MAE: {mae:,.3f}; RMSE: {rmse:,.3f}; R2: {r2:.6f}",
    "Model: models/dnn_medical_cost.keras (tersimpan, tidak dilatih ulang).",
    "",
]
for key, idx in (("CONTOH A - GALAT KECIL", 275), ("CONTOH B - GALAT BESAR", 1299)):
    r = selected.loc[idx]
    report += [
        key,
        f"Indeks setelah pembersihan: {idx}",
        f"Usia {int(r['age'])}; BMI {r['bmi']}; anak {int(r['children'])}; gender {r['gender']}; status merokok {r['discount_eligibility']}; region {r['region']}",
        f"Aktual: {r['expenses']:,.2f} | Prediksi: {r['prediksi']:,.2f}",
        f"Galat absolut: {r['galat_absolut']:,.2f} | Galat relatif: {r['galat_persen']:.2f}%",
        "",
    ]

# Diagram yang berasal dari hasil hitung nyata, bukan tangkapan layar aplikasi.
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=160)
for ax, (label, idx) in zip(axes, (("A. Galat kecil", 275), ("B. Galat besar", 1299))):
    r = selected.loc[idx]
    vals = [float(r["expenses"]), float(r["prediksi"])]
    bars = ax.bar(["Aktual", "Prediksi DNN"], vals, width=0.55)
    ax.set_title(label, fontsize=13, fontweight="bold")
    ax.set_ylabel("Biaya medis (expenses)")
    ymax = max(vals) * 1.23
    ax.set_ylim(0, ymax)
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:,.2f}", (b.get_x() + b.get_width()/2, b.get_height()),
                    textcoords="offset points", xytext=(0, 6), ha="center", fontsize=10)
    ax.text(0.5, 0.96, f"Galat: {r['galat_absolut']:,.2f} ({r['galat_persen']:.2f}%)",
            transform=ax.transAxes, ha="center", va="top", fontsize=10,
            bbox={"facecolor":"white", "edgecolor":"none", "alpha":0.85})
fig.suptitle("Perbandingan Prediksi DNN dan Nilai Aktual pada Dua Kasus Data Uji", fontsize=15, weight="bold")
fig.text(0.5, 0.02, "Sumber: model DNN dan preprocessor tersimpan; split data uji random_state=42. Tidak ada pelatihan ulang.",
         ha="center", fontsize=9)
fig.tight_layout(rect=[0, 0.055, 1, 0.93], w_pad=3)
fig.savefig(OUT / "Bukti_Dua_Kasus_Prediksi_DNN_Kelompok_6.png", bbox_inches="tight", dpi=220)
plt.close(fig)
print("\n".join(report))
print(f"Berkas dibuat di: {OUT}")
