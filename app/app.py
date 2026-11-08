import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow import keras

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "..", "models", "dnn_medical_cost.keras")
PREP_PATH = os.path.join(BASE, "..", "models", "preprocessor.joblib")

st.set_page_config(page_title="Prediksi Biaya Medis", page_icon="🏥", layout="centered")


@st.cache_resource
def load_artifacts():
    model = keras.models.load_model(MODEL_PATH)
    prep = joblib.load(PREP_PATH)
    return model, prep


def predict(model, prep, age, gender, bmi, children, smoking_status, region):
    """Inferensi dengan nama fitur internal lama demi kompatibilitas model.

    discount_eligibility_yes dalam model = smoker=yes pada dataset sumber.
    """
    row = {c: 0 for c in prep["columns"]}
    row.update({"age": age, "bmi": bmi, "children": children})
    if gender == "male" and "gender_male" in row:
        row["gender_male"] = 1
    # Fitur ini berasal dari kolom smoker, bukan kelayakan diskon.
    if smoking_status == "yes" and "discount_eligibility_yes" in row:
        row["discount_eligibility_yes"] = 1
    key = f"region_{region}"
    if key in row:  # region 'northeast' adalah kategori acuan (semua 0)
        row[key] = 1
    X = pd.DataFrame([row])[prep["columns"]].astype(float)
    X[prep["num_cols"]] = prep["scaler"].transform(X[prep["num_cols"]])
    log_pred = model.predict(X, verbose=0).ravel()[0]
    return float(np.expm1(log_pred))


st.title("🏥 Prediksi Biaya Medis")
st.caption("Deep Neural Network (regresi) — Proyek UTS Deep Learning")

try:
    model, prep = load_artifacts()
except Exception as e:
    st.error("Artefak model tidak dapat dimuat. Periksa folder `models/` dan versi dependensi.")
    st.exception(e)
    st.stop()

with st.form("input_form"):
    c1, c2 = st.columns(2)
    age = c1.number_input("Usia", min_value=18, max_value=100, value=30, step=1)
    bmi = c2.number_input("BMI", min_value=10.0, max_value=60.0, value=27.0, step=0.1)
    children = c1.number_input("Jumlah anak", min_value=0, max_value=10, value=0, step=1)
    gender = c2.selectbox("Gender", ["female", "male"])
    smoking_status = c1.selectbox(
        "Status merokok",
        ["no", "yes"],
        format_func=lambda value: "Tidak merokok" if value == "no" else "Merokok",
        help=("Status merokok pada data sumber. Nama teknis fitur model "
              "tetap discount_eligibility_yes untuk kompatibilitas.")
    )
    region = c2.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])
    submitted = st.form_submit_button("Prediksi", use_container_width=True)

if submitted:
    result = predict(model, prep, age, gender, bmi, children, smoking_status, region)
    st.metric("Perkiraan biaya medis (expenses)", f"{result:,.2f}")
    st.caption("Hasil merupakan estimasi model, bukan nilai pasti.")

with st.expander("Tentang model"):
    st.markdown(
        """
- **Model:** Deep Neural Network (MLP), aktivasi ReLU, optimizer Adam, loss MSE pada target `log1p(expenses)`.
- **Data:** medical_insurance.csv (kolom `premium` tidak dipakai karena data leakage).
- **Status merokok:** nilai `yes` = merokok dan `no` = tidak merokok. CSV proyek menamainya `discount_eligibility`, yang berasal dari `smoker` pada sumber publik, sehingga **bukan** variabel kelayakan diskon. Nama internal `discount_eligibility_yes` dipertahankan agar model tersimpan tetap dapat digunakan.
- **Pembanding:** Linear Regression dan Random Forest (lihat notebook).
"""
    )
