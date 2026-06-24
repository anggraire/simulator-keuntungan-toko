
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import streamlit as st

# Set Konfigurasi Halaman agar lebih rapi
st.set_page_config(
    page_title="Smart Store Profit Simulator AI",
    page_icon="💖",
    layout="centered"
)

# =========================================================
# MODERN PINK & LIGHT INTERACTIVE THEME 🌸
# =========================================================
st.markdown("""
<style>
/* Background Utama & Font */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background-color: #FFF5F8;
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Judul Utama */
h1 {
    color: #D6336C;
    text-align: center;
    font-weight: 700;
    padding-bottom: 20px;
}

/* Tulisan Default */
p, span, label {
    color: #4A4A4A;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 2px solid #FFF5F8;
    box-shadow: 2px 0px 10px rgba(214, 51, 108, 0.05);
}

[data-testid="stSidebar"] h2 {
    color: #D6336C !important;
}

/* Custom Card untuk Metric */
div[data-testid="stMetric"] {
    background-color: #ffffff;
    border: none;
    border-radius: 16px;
    padding: 20px 25px;
    box-shadow: 0 8px 24px rgba(214, 51, 108, 0.08);
    transition: transform 0.3s ease;
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-4px);
}

/* Text Metric */
div[data-testid="stMetricLabel"] {
    color: #4A4A4A;
    font-weight: 600;
    font-size: 0.95rem;
}

div[data-testid="stMetricValue"] {
    color: #D6336C;
    font-weight: 700;
}

/* Delta Metric Customization */
div[data-testid="stMetricDelta"] svg {
    fill: #FF4081;
}

/* Info & Status Box Customization */
.stAlert {
    border-radius: 12px;
    border: none;
    box-shadow: 0 4px 12px rgba(0,0,0,0.02);
}

/* Customize Sliders */
.stSlider div[data-testid="stThumbValue"] {
    background-color: #FF4081;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# BACKEND: MODEL & SIMULATION
# =========================================================

# Gunakan Caching
@st.cache_resource
def load_model():
    # 1. Menyiapkan data historis sederhana
    # Fitur: [Iklan (Juta), Diskon (%)]
    X_train = np.array([[5, 10], [10, 20], [15, 5], [20, 25], [25, 15]])
    
    # Target: Keuntungan (Juta)
    y_train = np.array([50, 80, 110, 90, 150])

    # 2. Melatih model
    model = LinearRegression().fit(X_train, y_train)
    return model

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

# =========================================================
# FRONTEND: UI / UX INTERFACE
# =========================================================

st.title("💖 Smart Store Profit Simulator AI")
st.markdown("<p style='text-align: center; color: #666; margin-top: -15px;'>Gunakan tuas di panel samping untuk mensimulasikan skenario 'What-If' secara real-time.</p>", unsafe_allow_html=True)
st.markdown("---")

# --- SIDEBAR: Variabel Kontrol ---
st.sidebar.markdown("## ⚙️ Kebijakan Intervensi")
st.sidebar.write("Sesuaikan parameter di bawah ini:")

iklan_slider = st.sidebar.slider("Anggaran Iklan (Juta)", 0, 50, 10)
diskon_slider = st.sidebar.slider("Besaran Diskon (%)", 0, 50, 10)

# --- ENGINE: Jalankan Simulasi ---
hasil_pred, delta = run_simulation(iklan_slider, diskon_slider)

# --- UI: Tampilkan Hasil ---
col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    st.metric(
        label="Prediksi Keuntungan Aktual", 
        value=f"Rp {hasil_pred:.2f} Jt", 
        delta=f"{delta:+.2f} Jt sejak Baseline"
    )

with col2:
    st.markdown("### 📊 Ringkasan Skenario")
    st.write(f"Modifikasi pada iklan dan diskon menghasilkan pergeseran profit sebesar **Rp {delta:.2f} Juta** dari kondisi normal bisnis Anda.")

st.markdown(" ")

# Validasi Baseline
st.info(
    f"💡 **Informasi Baseline:** Parameter acuan saat ini berada pada **Iklan Rp 10 Juta** & **Diskon 10%** "
    f"dengan estimasi profit default **Rp {baseline_pred:.2f} Juta**."
)

st.markdown("### Perbandingan Grafik")

# Visualisasi Perbandingan menggunakan St.bar_chart dengan warna tema kustom
data_plot = pd.DataFrame({
    'Skenario': ['Baseline', 'Intervensi'],
    'Keuntungan': [baseline_pred, hasil_pred]
})

# Menggunakan parameter color bawaan streamlit agar pas dengan tema pink/magenta
st.bar_chart(data=data_plot, x='Skenario', y='Keuntungan', color='#FF4081', use_container_width=True)

# Storytelling / Dynamic Feedback
st.markdown("### 🤖 Rekomendasi AI")
if delta > 0:
    st.success(f"📈 **Keputusan Bagus!** Skenario ini diproyeksikan mampu mendongkrak keuntungan sebesar **{delta:.2f} Juta** di atas rata-rata.")
elif delta < 0:
    st.warning(f"📉 **Perlu Dievaluasi:** Strategi ini berisiko menurunkan profit sebesar **{abs(delta):.2f} Juta**. Coba kurangi diskon atau optimalkan iklan.")
else:
    st.info("⚖️ **Stabil:** Skenario ini menghasilkan performa finansial yang identik dengan kondisi baseline saat ini.")
