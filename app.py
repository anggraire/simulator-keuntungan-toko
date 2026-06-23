
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import streamlit as st

st.markdown("""
<style>

.stApp {
    background-color: #fff5fa;
}

h1 {
    color: #d63384;
    text-align: center;
}

h2, h3 {
    color: #c2185b;
}

[data-testid="stSidebar"] {
    background-color: #ffe6f2;
}

div[data-testid="stMetric"] {
    background-color: white;
    border: 2px solid #ffb6d9;
    padding: 15px;
    border-radius: 20px;
}

</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():

    X_train = np.array([
        [5,10],
        [10,20],
        [15,5],
        [20,25],
        [25,15]
    ])


    y_train = np.array([
        50,
        80,
        110,
        90,
        150
    ])


    model = LinearRegression().fit(
        X_train,
        y_train
    )

    return model


# =========================
# BASELINE
# =========================


def get_baseline(model):

    baseline_input = np.array([[10,10]])

    baseline_pred = model.predict(
        baseline_input
    )[0]

    return baseline_pred



# =========================
# SIMULATION
# =========================


def run_simulation(
    model,
    new_iklan,
    new_diskon,
    baseline_pred
):

    input_user = np.array([
        [new_iklan,new_diskon]
    ])


    prediction = model.predict(
        input_user
    )[0]


    delta = prediction - baseline_pred


    return prediction, delta


# =========================
# LOAD MODEL
# =========================


model = load_model()

baseline_pred = get_baseline(model)



# =========================
# HEADER
# =========================


st.title(
    "🌸 Smart Store Profit Simulator AI 🌸"
)


st.markdown(
"""
<div style="
background-color:#ffd6e8;
padding:20px;
border-radius:20px;
text-align:center;
">

✨ Simulator Prediksi Keuntungan Toko ✨

Uji strategi iklan dan diskon menggunakan AI

</div>
""",
unsafe_allow_html=True
)



# =========================
# SIDEBAR
# =========================


st.sidebar.markdown(
"""
## 🌷 Pengaturan Strategi

Atur variabel bisnis:
"""
)


iklan_slider = st.sidebar.slider(
    "💰 Anggaran Iklan (Juta)",
    0,
    50,
    10
)


diskon_slider = st.sidebar.slider(
    "🏷️ Besaran Diskon (%)",
    0,
    50,
    10
)


# =========================
# OUTPUT
# =========================


hasil_pred, delta = run_simulation(
    model,
    iklan_slider,
    diskon_slider,
    baseline_pred
)



col1, col2 = st.columns(2)



with col1:

    st.metric(
        "💰 Prediksi Keuntungan",
        f"Rp {hasil_pred:.2f} Jt",
        f"{delta:.2f} Jt"
    )



with col2:

    st.info(
        f"""
📊 Perubahan dari baseline:

{delta:.2f} Juta
"""
    )




st.success(
f"""
🌸 Kondisi Awal

Iklan: 10 Juta

Diskon: 10%

Prediksi:
Rp {baseline_pred:.2f} Juta
"""
)



data_plot = pd.DataFrame({

    "Skenario":[
        "Baseline",
        "Intervensi"
    ],

    "Keuntungan":[
        baseline_pred,
        hasil_pred
    ]

})


st.subheader(
    "📈 Perbandingan Keuntungan"
)


st.bar_chart(
    data_plot,
    x="Skenario",
    y="Keuntungan"
)



if delta > 0:

    st.success(
        "✨ Strategi meningkatkan keuntungan"
    )

elif delta < 0:

    st.warning(
        "⚠️ Strategi menurunkan keuntungan"
    )

else:

    st.info(
        "Strategi sama dengan baseline"
    )
