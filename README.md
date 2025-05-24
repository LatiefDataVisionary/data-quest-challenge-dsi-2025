# data-quest-challenge-dsi-2025
Repository ini digunakan untuk menyimpan dan mengeola file proyek lomba Data Quest Challenge by Data Science Indonesia (DSI) 2025.


<p align="center">
  <!-- Jika ada logo tim atau banner lomba, bisa ditaruh di sini -->
  <!-- <img src="URL_LOGO_ATAU_BANNER" alt="Team Cubit Aku Dong / DSI Data Quest 2025" width="300"/> -->
</p>

# 🚀 DSI Data Quest 2025: Prediksi Nasabah Deposito Potensial 🚀
## Tim: Cubit Aku Dong 🤏✨

---

Selamat datang di *repository* resmi tim **Cubit Aku Dong** untuk **Data Quest Challenge DSI MeetUp 2025**! Kami adalah tim yang terdiri dari dua individu antusias, berkolaborasi untuk memecahkan tantangan prediksi di industri perbankan. Misi kami adalah membangun model *machine learning* yang handal untuk mengidentifikasi nasabah yang berpotensi berlangganan produk deposito berjangka.

---

### 🎯 **Objektif Utama & Tantangan**

> "Memahami nasabah adalah fondasi dari strategi pemasaran yang efektif."

Proyek ini berfokus pada:
1.  **Identifikasi Prospek**: Menggali data historis untuk menemukan sinyal-sinyal yang mengindikasikan ketertarikan nasabah terhadap deposito berjangka.
2.  **Pengembangan Model Klasifikasi**: Merancang, melatih, dan mengevaluasi model yang mampu membedakan antara:
    *   ✅ **Nasabah Akan Berlangganan (Target = 1)**
    *   ❌ **Nasabah Tidak Akan Berlangganan (Target = 0)**
3.  **Ukuran Keberhasilan**: Performa model akan dinilai berdasarkan **Area Under the ROC Curve (AUC)**, sebuah metrik standar untuk mengevaluasi kualitas model klasifikasi.

---

### 🧑‍💻 **Tim "Cubit Aku Dong"**

| Nama Anggota                | Fokus & Kontribusi Utama                         | Tautan Profesional (Opsional) |
| :-------------------------- | :----------------------------------------------- | :---------------------------- |
| **[Nama Lengkap Anggota 1]**  | Analisis Data Eksploratif & _Feature Engineering_ | [LinkedIn]() / [GitHub]()     |
| **[Nama Lengkap Anggota 2]**  | Pengembangan Model & Evaluasi Performa           | [LinkedIn]() / [GitHub]()     |

---

### 📁 **Struktur Proyek & Navigasi**

Untuk memudahkan eksplorasi, berikut adalah arsitektur *repository* kami:
DSI_DataQuest_2025_CubitAkuDong/
├── 📜 .gitignore # Mengabaikan file yang tidak relevan
├── 📖 README.md # Panduan komprehensif ini
├── 📦 requirements.txt # Daftar dependensi Python proyek
│
├── 🗃️ data/
│ ├── raw/ # Dataset asli (training & validation)
│ └── processed/ # Dataset yang telah melalui tahap pembersihan
│
├── 📈 notebooks/
│ ├── 01_Exploratory_Data_Analysis.ipynb # Penjelajahan & visualisasi data awal
│ ├── 02_Data_Preprocessing.ipynb # Pembersihan data & rekayasa fitur
│ ├── 03_Model_Development.ipynb # Eksperimen & pelatihan model
│ └── 🏆 DCM_DMU_2025_Notebook_CubitAkuDong.ipynb # NOTEBOOK FINAL UNTUK SUBMISSION
│
├── 🛠️ src/ # Kumpulan script Python modular & reusable
│
├── 🧠 models/ # Penyimpanan model-model yang telah dilatih
│
├── 📤 submissions/ # Arsip file prediksi (.csv) yang disubmit
│
└── 📝 reports/ # (Opsional) Laporan, insight, & materi presentasi
└── figures/ # Visualisasi kunci (plot, grafik)
---

---
### 🚀 **Langkah Persiapan & Instalasi**

Mari siapkan lingkungan kerja Anda:

1.  **Clone Repositori:**
    ```bash
    git clone https://github.com/[UsernameGitHubAnda]/DSI_DataQuest_2025_CubitAkuDong.git
    cd DSI_DataQuest_2025_CubitAkuDong
    ```

2.  **Buat & Aktifkan Lingkungan Virtual** (Sangat Direkomendasikan):
    *   Menggunakan `venv`:
        ```bash
        python -m venv .venv-cubit
        # Windows
        .venv-cubit\Scripts\activate
        # macOS/Linux
        source .venv-cubit/bin/activate
        ```
    *   Atau `conda`:
        ```bash
        conda create -n dsi_cubit_env python=3.9  # Sesuaikan versi Python
        conda activate dsi_cubit_env
        ```

3.  **Instal Semua Kebutuhan:**
    ```bash
    pip install -r requirements.txt
    ```

---

### ▶️ **Cara Menjalankan Analisis & Prediksi**

1.  **Persiapkan Data**: Pastikan dataset dari panitia (`training_data.csv`, `validation_data.csv`) telah ditempatkan di dalam direktori `data/raw/`.
2.  **Notebook Utama**: Buka dan jalankan *notebook* `notebooks/🏆 DCM_DMU_2025_Notebook_CubitAkuDong.ipynb`. *Notebook* ini merangkum keseluruhan proses dari pemuatan data hingga generasi prediksi.
3.  **Hasil Prediksi**: File *submission* (`DCM_DMU_2025_Model_CubitAkuDong.csv`) akan otomatis tersimpan di folder `submissions/`.

---

### 💡 **Pendekatan Teknis & Teknologi**

*   **Bahasa & Ekosistem**: Python 3.x
*   **Pustaka Kunci**:
    *   Data Manipulation: `Pandas`, `NumPy`
    *   Visualisasi: `Matplotlib`, `Seaborn`
    *   Machine Learning: `Scikit-learn`, (misal: `XGBoost`, `LightGBM`, atau `CatBoost`)
*   **Metodologi**:
    1.  **Eksplorasi & Pemahaman Data (EDA)**: Investigasi mendalam terhadap karakteristik data.
    2.  **Pra-pemrosesan & Rekayasa Fitur**: Transformasi data mentah menjadi format yang optimal untuk model.
    3.  **Seleksi & Pelatihan Model**: Eksperimen dengan berbagai algoritma klasifikasi.
    4.  **Penyetelan Hiperparameter**: Optimasi parameter model untuk performa maksimal.
    5.  **Validasi Silang**: Memastikan robustisitas dan kemampuan generalisasi model.

---

Terima kasih telah mengunjungi *repository* tim **Cubit Aku Dong**! Kami sangat antusias untuk berpartisipasi dan belajar dalam kompetisi ini.

Semoga sukses untuk semua tim peserta! 💪
