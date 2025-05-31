import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np # Buat bikin data contoh dan prediksi contoh

# --- (Opsional) Konfigurasi Halaman Streamlit ---
# Ini bagus buat ngasih judul di tab browser dan ikonnya
st.set_page_config(
    page_title="Dashboard Data Mining Keren Kamu",
    page_icon="✨",
    layout="wide" # Biar layoutnya lebih lebar
)

# --- (Contoh) Fungsi-Fungsi Bantuan ---
# Di aplikasi beneran, fungsi ini bakal ngambil data asli atau model asli kamu

@st.cache_data # Biar data cuma di-load sekali dan app jadi cepet
def load_example_data():
    # Kita bikin data contoh (dummy data) aja ya buat ilustrasi
    data = pd.DataFrame({
        'ID_Pelanggan': range(1, 101),
        'Lama_Berlangganan_Bulan': np.random.randint(1, 72, 100),
        'Tagihan_Bulanan_USD': np.random.uniform(20, 200, 100).round(2),
        'Total_Tagihan_USD': np.random.uniform(20, 5000, 100).round(2),
        'Jenis_Kelamin': np.random.choice(['Pria', 'Wanita'], 100),
        'Kontrak': np.random.choice(['Bulanan', 'Satu Tahun', 'Dua Tahun'], 100),
        'Churn': np.random.choice(['Ya', 'Tidak'], 100, p=[0.26, 0.74]) # Kira-kira 26% pelanggan churn
    })
    return data

def get_mock_model_results():
    # Ini contoh hasil metrik model (dummy)
    metrics = {
        "Nama Model": "Random Forest Classifier (Contoh)",
        "Akurasi": 0.85,
        "Presisi (untuk Churn 'Ya')": 0.70,
        "Recall (untuk Churn 'Ya')": 0.65,
        "F1-Score (untuk Churn 'Ya')": 0.67
    }
    # Ini contoh fitur penting (dummy)
    feature_importance = pd.DataFrame({
        'Fitur': ['Kontrak_Bulanan', 'Lama_Berlangganan_Bulan', 'Total_Tagihan_USD', 'Tagihan_Bulanan_USD'],
        'Skor Pentingnya': [0.55, 0.25, 0.15, 0.05]
    }).sort_values(by='Skor Pentingnya', ascending=False)
    return metrics, feature_importance

def make_mock_prediction(input_features):
    # Ini contoh logika prediksi (dummy)
    # Di aplikasi beneran, kamu bakal pake model.predict(data_input_yang_sudah_diproses)
    
    # Logika sederhana buat contoh:
    if input_features['Lama_Berlangganan_Bulan'] < 6 and input_features['Kontrak'] == 'Bulanan':
        prediksi_churn = "Ya (Berpotensi Besar Churn)"
        probabilitas = np.random.uniform(0.75, 0.98)
    elif input_features['Tagihan_Bulanan_USD'] > 150:
        prediksi_churn = "Ya (Berpotensi Churn)"
        probabilitas = np.random.uniform(0.6, 0.74)
    else:
        prediksi_churn = "Tidak (Kemungkinan Kecil Churn)"
        probabilitas = np.random.uniform(0.05, 0.4)
    
    return prediksi_churn, f"{probabilitas*100:.2f}%"

# --- Sidebar untuk Navigasi Halaman ---
st.sidebar.title("Menu Navigasi 🧭")
pilihan_halaman = st.sidebar.radio(
    "Mau lihat halaman apa, bang?",
    ["🏠 Halaman Awal (EDA)", "🧠 Hasil Model", "🔮 Formulir Prediksi"]
)
st.sidebar.markdown("---") # Garis pemisah
st.sidebar.info("Dashboard ini dibuat dengan penuh cinta oleh Kelompok 11! ❤️")


# --- Konten Halaman Utama (berdasarkan pilihan di sidebar) ---

if pilihan_halaman == "🏠 Halaman Awal (EDA)":
    st.title("📊 Exploratory Data Analysis (EDA) - Kenalan Sama Datamu!")
    st.markdown("Di halaman ini, kita bakal 'kepoin' dataset pelanggan kita buat cari tau insight-insight menarik. Yuk mulai!")

    # Load data contoh
    df_pelanggan = load_example_data()

    st.subheader("Cuplikan Dataset Pelanggan (5 Baris Pertama):")
    st.dataframe(df_pelanggan.head())

    st.subheader("Statistik Deskriptif:")
    st.write(df_pelanggan.describe())

    st.subheader("Distribusi Churn Pelanggan:")
    churn_counts = df_pelanggan['Churn'].value_counts()
    st.bar_chart(churn_counts)
    st.caption(f"Dari data contoh, ada {churn_counts.get('Ya', 0)} pelanggan yang churn dan {churn_counts.get('Tidak', 0)} yang tidak.")

    st.subheader("Distribusi Lama Berlangganan (dalam Bulan):")
    # Kamu bisa pake st.bar_chart atau bikin visualisasi pake matplotlib/seaborn terus tampilin pake st.pyplot()
    st.bar_chart(df_pelanggan['Lama_Berlangganan_Bulan'].value_counts().sort_index())
    
    st.subheader("Sebaran Tagihan Bulanan:")
    # Histogram sederhana
    fig, ax = plt.subplots() # Perlu import matplotlib.pyplot as plt di awal file
    ax.hist(df_pelanggan['Tagihan_Bulanan_USD'], bins=20, color='skyblue', edgecolor='black')
    ax.set_title('Histogram Tagihan Bulanan')
    ax.set_xlabel('Tagihan Bulanan (USD)')
    ax.set_ylabel('Frekuensi')
    st.pyplot(fig) # Tampilkan plot matplotlib

    st.markdown("---")
    st.success("EDA selesai! Di aplikasi beneran, kamu bisa tambahin lebih banyak visualisasi dan analisis ya!")


elif pilihan_halaman == "🧠 Hasil Model":
    st.title("📈 Hasil Pelatihan Model Prediksi Churn")
    st.markdown("Ini dia ringkasan performa model machine learning kita yang udah dilatih buat prediksi churn. Keren kan!")

    # Di aplikasi beneran, kamu bakal load modelmu (misal dari file .pkl) dan evaluasi di sini
    # Contoh: model = joblib.load('model_churn_kerenku.pkl')
    # Lalu hitung akurasi, presisi, recall, dll. dari data test.
    
    st.warning("Disclaimer: Metrik dan fitur penting di bawah ini adalah CONTOH dan bukan dari model yang sesungguhnya ya, bang!😉")

    metrik_model, fitur_penting = get_mock_model_results()

    st.subheader("Metrik Evaluasi Model:")
    # Tampilkan metrik dalam beberapa kolom biar rapi
    col1, col2 = st.columns(2)
    col1.metric("Nama Model (Contoh)", metrik_model["Nama Model"])
    col1.metric("Akurasi", f"{metrik_model['Akurasi']:.2f}") # Format jadi 2 angka di belakang koma
    col2.metric("Presisi (untuk Churn 'Ya')", f"{metrik_model['Presisi (untuk Churn \'Ya\')']:.2f}")
    col2.metric("Recall (untuk Churn 'Ya')", f"{metrik_model['Recall (untuk Churn \'Ya\')']:.2f}")
    st.metric("F1-Score (untuk Churn 'Ya')", f"{metrik_model['F1-Score (untuk Churn \'Ya\')']:.2f}", delta="Contoh Delta")


    st.subheader("Fitur Paling Penting Menurut Model (Contoh):")
    st.dataframe(fitur_penting)
    # Visualisasi fitur penting pake bar chart
    st.bar_chart(fitur_penting.set_index('Fitur'))

    st.markdown("---")
    st.info("Di aplikasi nyata, kamu bisa nambahin visualisasi Confusion Matrix, ROC Curve, atau tabel perbandingan beberapa model di sini!")


elif pilihan_halaman == "🔮 Formulir Prediksi":
    st.title("📝 Formulir Prediksi Churn Pelanggan Secara Interaktif")
    st.markdown("Yuk, coba masukkin data pelanggan baru di bawah ini buat dapetin prediksinya dari model kita!")

    # Bikin form biar inputnya dikumpulin dulu sebelum diproses
    with st.form(key="form_prediksi_churn"):
        st.subheader("Masukkan Detail Pelanggan:")

        # Bikin input field sesuai fitur yang dipake modelmu
        # Ini cuma contoh ya, bang!
        input_lama_berlangganan = st.slider(
            label="Lama Berlangganan (Bulan):", 
            min_value=0, max_value=100, value=12, step=1,
            help="Berapa bulan pelanggan ini sudah berlangganan?"
        )
        input_kontrak = st.selectbox(
            label="Jenis Kontrak:",
            options=['Bulanan', 'Satu Tahun', 'Dua Tahun'],
            index=0, # Pilihan default
            help="Apa jenis kontrak yang dimiliki pelanggan?"
        )
        input_tagihan_bulanan = st.number_input(
            label="Tagihan Bulanan (USD):",
            min_value=0.0, max_value=500.0, value=70.0, step=0.1,
            help="Berapa tagihan bulanan pelanggan ini?"
        )
        # Tambahin input field lain sesuai kebutuhan modelmu di sini...
        # Misalnya:
        # input_layanan_internet = st.selectbox("Layanan Internet:", ['DSL', 'Fiber Optic', 'Tidak Ada'])
        # input_dukungan_teknis = st.radio("Dukungan Teknis:", ['Ya', 'Tidak', 'Tidak ada layanan internet'])

        # Tombol buat submit form
        tombol_submit_prediksi = st.form_submit_button(label="Prediksi Sekarang! 🚀")

    # Setelah tombol submit di form ditekan
    if tombol_submit_prediksi:
        # Kumpulin semua input jadi satu dictionary atau DataFrame
        data_input_pengguna = {
            'Lama_Berlangganan_Bulan': input_lama_berlangganan,
            'Kontrak': input_kontrak,
            'Tagihan_Bulanan_USD': input_tagihan_bulanan
            # ... masukin variabel input lain di sini
        }
        
        # Di aplikasi beneran, kamu perlu lakuin pre-processing ke data_input_pengguna
        # biar formatnya sama kayak data yang dipake buat training model (misal: one-hot encoding)
        
        # Panggil fungsi prediksi (ini pake fungsi contoh kita)
        hasil_prediksi, probabilitas_prediksi = make_mock_prediction(data_input_pengguna)

        st.subheader("🎉 Hasil Prediksi Churn:")
        if "Berpotensi" in hasil_prediksi: # Cek kata kunci di hasil prediksi contoh
            st.error(f"Prediksi Model: Pelanggan ini **{hasil_prediksi}**")
            st.warning(f"Dengan estimasi probabilitas churn: **{probabilitas_prediksi}**")
            st.markdown("😱 Wah, gawat! Sebaiknya segera hubungi pelanggan ini dan tawarkan sesuatu yang menarik biar nggak jadi churn!")
        else:
            st.success(f"Prediksi Model: Pelanggan ini **{hasil_prediksi}**")
            st.info(f"Dengan estimasi probabilitas tidak churn: **{probabilitas_prediksi}**")
            st.balloons() # Efek balon biar seneng! 🎈
            st.markdown("🥳 Aman! Pelanggan ini kayaknya masih setia. Pertahankan terus ya!")
        
        st.markdown("---")
        st.write("Data yang Kamu Masukkan:")
        st.json(data_input_pengguna) # Tampilkan data input dalam format JSON biar rapi
        st.caption("Ingat ya, ini cuma prediksi contoh. Akurasi prediksi di aplikasi nyata tergantung kualitas model dan data yang kamu punya!")

# --- (Opsional) Footer biar makin kece ---
st.markdown("---")
st.markdown("© 2025 Dibuat dengan Penuh Semangat oleh Kelompok 11! | Happy Coding! 🔥💻")

# Jangan lupa import library yang dibutuhkan di bagian paling atas file ya!
# Misalnya: import matplotlib.pyplot as plt (kalau pake matplotlib)
#           import joblib (kalau mau load model dari file .pkl)

