
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import streamlit as st

# =========================
# PINK THEME UI 🌸
# =========================

st.markdown("""
<style>

.stApp {
    background-color: #fff7fb;
}


h1 {
    color: #d85a8a;
}


[data-testid="stSidebar"] {
    background-color: #fdebf2;
}


[data-testid="stSidebar"] * {
    color: #6b3b4d;
}


div[data-testid="stMetric"] {

    background-color: #fff0f5;

    border: 1px solid #f3b6cc;

    border-radius: 15px;

    padding: 15px;

}


div[data-testid="stMetricLabel"] {
    color: #b04a73;
}


div[data-testid="stMetricValue"] {
    color: #7a3550;
}


.stAlert {
    border-radius: 15px;
}


</style>
""", unsafe_allow_html=True)

# Gunakan Caching
@st.cache_resource
def load_model():
    # 1. Menyiapkan data historis sederhana
    # Fitur: [Iklan (Juta), Diskon (%)]
    X_train = np.array([[5, 10], [10, 20], [15, 5], [20, 25], [25, 15]])

    # Target: Keuntungan (Juta)
    y_train = np.array([50, 80, 110, 90, 150])

    # 2. Melatih model (Mesin Replika)
    model = LinearRegression().fit(X_train, y_train)

    return model


# Organisasi Kode
def get_baseline(model):
    # 3. Menetapkan Skenario Dasar (Baseline)
    # Kondisi saat ini: Iklan 10 Juta, Diskon 10%
    baseline_input = np.array([[10, 10]])
    baseline_pred = model.predict(baseline_input)[0]

    return baseline_pred


def run_simulation(new_iklan, new_diskon):
    # Input baru dari user (Intervensi)
    intervention_input = np.array([[new_iklan, new_diskon]])

    # Prediksi hasil intervensi
    prediction = model.predict(intervention_input)[0]

    # Menghitung Delta (Selisih)
    delta_y = prediction - baseline_pred

    return prediction, delta_y


# Load model dan baseline
model = load_model()
baseline_pred = get_baseline(model)

print(f"Prediksi Keuntungan Baseline: Rp {baseline_pred:.2f} Juta")


st.title("🌸 Simulator Kebijakan Keuntungan Toko")
st.write("Gunakan slider untuk menguji skenario 'What-If'.")

# --- SIDEBAR: Variabel Kontrol ---
st.sidebar.header("Tuas Kebijakan (Intervensi)")
iklan_slider = st.sidebar.slider("Anggaran Iklan (Juta)", 0, 50, 10)
diskon_slider = st.sidebar.slider("Besaran Diskon (%)", 0, 50, 10)

# --- ENGINE: Jalankan Simulasi ---
hasil_pred, delta = run_simulation(iklan_slider, diskon_slider)

# --- UI: Tampilkan Hasil ---
col1, col2 = st.columns(2)
col1.metric("Prediksi Keuntungan", f"Rp {hasil_pred:.2f} Jt", f"{delta:.2f} Jt")
col2.write(f"Skenario ini menghasilkan perubahan sebesar {delta:.2f} Juta dibandingkan kondisi baseline.")

# Validasi Baseline
st.info(
    f"Baseline saat ini adalah Iklan 10 Juta dan Diskon 10%, "
    f"dengan prediksi keuntungan Rp {baseline_pred:.2f} Juta."
)

# Visualisasi Perbandingan
data_plot = pd.DataFrame({
    'Skenario': ['Baseline', 'Intervensi'],
    'Keuntungan': [baseline_pred, hasil_pred]
})

st.bar_chart(data=data_plot, x='Skenario', y='Keuntungan')

# Storytelling
if delta > 0:
    st.success("Skenario ini menunjukkan peningkatan keuntungan dibandingkan kondisi baseline.")
elif delta < 0:
    st.warning("Skenario ini menunjukkan penurunan keuntungan dibandingkan kondisi baseline.")
else:
    st.info("Skenario ini menghasilkan keuntungan yang sama dengan kondisi baseline.")
