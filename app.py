import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Prediksi Harga Tiket Pesawat",
    page_icon="✈️",
    layout="wide"
)

# --- Fungsi untuk Melatih Model ---
@st.cache_data
def train_model():
    """
    Fungsi untuk memuat data, memproses, dan melatih model regresi linier.
    Menggunakan caching untuk performa yang lebih baik.
    """
    # 1. Pemuatan Data
    try:
        df = pd.read_csv('Clean_Dataset.csv')
    except FileNotFoundError:
        st.error("File 'Clean_Dataset.csv' tidak ditemukan. Pastikan file berada di direktori yang sama dengan skrip Anda.")
        return None, None, None, None

    # Hapus kolom yang tidak digunakan
    df = df.drop(columns=['Unnamed: 0', 'flight'])

    # 2. Pra-pemrosesan Data
    X = df.drop('price', axis=1)
    y = df['price']

    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=np.number).columns

    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder.fit(X[categorical_features])

    X_encoded_cat = pd.DataFrame(
        encoder.transform(X[categorical_features]),
        columns=encoder.get_feature_names_out(categorical_features)
    )

    X_processed = pd.concat([X[numerical_features].reset_index(drop=True), X_encoded_cat], axis=1)

    # 3. Pelatihan Model
    model = LinearRegression()
    model.fit(X_processed, y)

    return model, encoder, X_processed.columns, df

# --- Pemuatan dan Pelatihan Model ---
model, encoder, processed_columns, df = train_model()

# --- Antarmuka Pengguna Streamlit ---
if model is not None:
    st.title("✈️ Prediksi Harga Tiket Pesawat")
    st.markdown("Gunakan aplikasi ini untuk memprediksi harga tiket pesawat berdasarkan detail penerbangan yang Anda masukkan.")
    st.markdown("---")

    # --- Sidebar untuk Input Pengguna ---
    st.sidebar.header("Masukkan Detail Penerbangan")

    airline = st.sidebar.selectbox("Maskapai Penerbangan", sorted(df['airline'].unique()))
    source_city = st.sidebar.selectbox("Kota Keberangkatan", sorted(df['source_city'].unique()))
    departure_time = st.sidebar.selectbox("Waktu Keberangkatan", sorted(df['departure_time'].unique()))
    stops = st.sidebar.selectbox("Jumlah Transit (Stops)", sorted(df['stops'].unique()))
    arrival_time = st.sidebar.selectbox("Waktu Kedatangan", sorted(df['arrival_time'].unique()))
    destination_city = st.sidebar.selectbox("Kota Tujuan", sorted(df['destination_city'].unique()))
    flight_class = st.sidebar.selectbox("Kelas Penerbangan", sorted(df['class'].unique()))
    days_left = st.sidebar.slider("Hari Sebelum Keberangkatan", 1, 50, 15)

    if st.sidebar.button("Prediksi Harga"):
        # --- Proses Input Pengguna dan Lakukan Prediksi ---

        ### PERUBAHAN 1: HITUNG 'duration' BERDASARKAN INPUT WAKTU ###
        time_mapping = {
            'Early_Morning': 5, 'Morning': 8, 'Afternoon': 15,
            'Evening': 19, 'Night': 22, 'Late_Night': 2
        }
        dep_time_num = time_mapping[departure_time]
        arr_time_num = time_mapping[arrival_time]

        if arr_time_num <= dep_time_num:
            duration = (arr_time_num + 24) - dep_time_num
        else:
            duration = arr_time_num - dep_time_num

        # Buat DataFrame dari input kategorikal pengguna
        user_input_categorical = pd.DataFrame({
            'airline': [airline], 'source_city': [source_city],
            'departure_time': [departure_time], 'stops': [stops],
            'arrival_time': [arrival_time], 'destination_city': [destination_city],
            'class': [flight_class]
        })

        # One-Hot Encode input pengguna
        user_input_encoded = pd.DataFrame(
            encoder.transform(user_input_categorical),
            columns=encoder.get_feature_names_out(user_input_categorical.columns)
        )

        # Buat DataFrame dari input numerik pengguna (termasuk durasi yang sudah dihitung)
        user_input_numerical = pd.DataFrame({
            'duration': [duration],
            'days_left': [days_left]
        })

        # Gabungkan fitur numerik dan kategorikal
        user_input_combined = pd.concat([user_input_numerical, user_input_encoded], axis=1)

        ### PERUBAHAN 2: GUNAKAN .reindex() UNTUK MENCOCOKKAN KOLOM SECARA OTOMATIS ###
        # Ini cara yang lebih andal untuk memastikan semua kolom sama persis dengan saat training
        final_input = user_input_combined.reindex(columns=processed_columns, fill_value=0)

        # Lakukan Prediksi
        try:
            prediction = model.predict(final_input)
            predicted_price = prediction[0]

            st.subheader("Hasil Prediksi Harga")
            st.success(f"**Estimasi Harga Tiket: ₹ {predicted_price:,.2f}**")
            st.info("Catatan: Prediksi ini didasarkan pada model Regresi Linier dan data historis. Harga sebenarnya dapat bervariasi.")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")

else:
    st.warning("Model tidak dapat dilatih. Silakan periksa pesan error di atas.")