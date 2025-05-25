# 🏆 DSI Data Quest 2025: Prediksi Berlangganan Deposito Bank
## Tim: Cubit Aku Dong 🤏✨
### Notebook Utama: DCM_DMU_2025_Notebook_Cubit_Aku_Dong.ipynb

---

## 📝 Daftar Isi

1.  **[Pendahuluan](#1-pendahuluan)**
    *   1.1. Latar Belakang Masalah
    *   1.2. Tujuan Proyek & Metrik Evaluasi (AUC)
    *   1.3. Deskripsi Dataset (Ringkasan dari Guideline)
    *   1.4. Alur Kerja Proyek

2.  **[Persiapan Lingkungan & Import Library](#2-persiapan-lingkungan--import-library)**
    *   2.1. Pengaturan Konfigurasi (Seed, Opsi Tampilan Pandas, dll.)
    *   2.2. Import Library yang Dibutuhkan (NumPy, Pandas, Scikit-learn, Visualisasi, dll.)
    *   2.3. (Opsional) Fungsi Bantuan (Helper Functions)

3.  **[Pemuatan Data (Data Loading)](#3-pemuatan-data-data-loading)**
    *   3.1. Mendefinisikan Path ke Dataset (Training & Validation)
    *   3.2. Membaca Dataset Training (`df_train`)
    *   3.3. Membaca Dataset Validasi (`df_val`)
    *   3.4. Tinjauan Awal Data (Head, Tail, Shape, Info) untuk Kedua Dataset

4.  **[Eksplorasi Data Awal (Initial EDA) & Pemahaman Data](#4-eksplorasi-data-awal-initial-eda--pemahaman-data)**
    *   4.1. Ringkasan Statistik Deskriptif (`.describe()`) untuk Fitur Numerik
    *   4.2. Identifikasi Tipe Data Kolom & Kardinalitas Fitur Kategorikal
    *   4.3. **Analisis Variabel Target (`berlangganan_deposito`)**
        *   4.3.1. Distribusi Kelas Target (Imbalance Check)
        *   4.3.2. Visualisasi Distribusi Kelas Target
    *   4.4. **Analisis Fitur Numerik**
        *   4.4.1. Distribusi Masing-masing Fitur Numerik (Histogram, KDE Plot)
        *   4.4.2. Deteksi Outlier (Box Plot)
        *   4.4.3. Korelasi antar Fitur Numerik (Heatmap Korelasi)
    *   4.5. **Analisis Fitur Kategorikal**
        *   4.5.1. Distribusi Masing-masing Fitur Kategorikal (Bar Plot Frekuensi)
        *   4.5.2. Hubungan Fitur Kategorikal dengan Variabel Target (Stacked Bar Plot, Grouped Bar Plot)
    *   4.6. **Analisis Missing Values**
        *   4.6.1. Identifikasi Jumlah & Persentase Missing Values per Kolom
        *   4.6.2. Visualisasi Missing Values (Heatmap, Bar Plot)
    *   4.7. **Analisis Duplikasi Data**
        *   4.7.1. Cek Data Duplikat
    *   4.8. Ringkasan Temuan EDA & Hipotesis Awal

5.  **[Pra-pemrosesan Data (Data Preprocessing)](#5-pra-pemrosesan-data-data-preprocessing)**
    *   5.1. **Strategi Pra-pemrosesan Gabungan (Training & Validation Set)**
        *   *Catatan: Penting untuk menerapkan transformasi yang sama pada kedua set data untuk menghindari data leakage dan memastikan konsistensi.*
    *   5.2. Penanganan Data Duplikat (jika ada)
    *   5.3. **Penanganan Missing Values**
        *   5.3.1. Strategi Imputasi untuk Fitur Numerik (Mean, Median, Modus, atau Model-based)
        *   5.3.2. Strategi Imputasi untuk Fitur Kategorikal (Modus, Kategori Baru 'Unknown', atau Model-based)
    *   5.4. **Penanganan Outlier** (jika diperlukan dan berdasarkan analisis EDA)
        *   5.4.1. Metode (Capping, Trimming, Transformasi)
    *   5.5. **Encoding Fitur Kategorikal**
        *   5.5.1. Identifikasi Fitur Kategorikal Nominal vs. Ordinal
        *   5.5.2. One-Hot Encoding (untuk fitur nominal dengan kardinalitas rendah)
        *   5.5.3. Label Encoding / Ordinal Encoding (untuk fitur ordinal atau nominal dengan kardinalitas tinggi jika model mendukung)
        *   5.5.4. (Opsional) Target Encoding atau Frekuensi Encoding (dengan hati-hati untuk menghindari overfitting)
    *   5.6. **Transformasi Fitur Numerik**
        *   5.6.1. Scaling/Normalisasi (StandardScaler, MinMaxScaler) - *Penting untuk model yang sensitif terhadap skala fitur seperti SVM, KNN, Neural Networks.*
        *   5.6.2. (Opsional) Transformasi Logaritmik atau Box-Cox (untuk fitur skewed)
    *   5.7. Verifikasi Hasil Pra-pemrosesan (Cek Shape, Info, Missing Values lagi)

6.  **[Rekayasa Fitur (Feature Engineering)](#6-rekayasa-fitur-feature-engineering)**
    *   *Bagian ini sangat bergantung pada kreativitas dan domain knowledge. Berikut beberapa ide umum:*
    *   6.1. Pembuatan Fitur Baru dari Fitur yang Ada
        *   6.1.1. Interaksi antar Fitur (misal, rasio, perkalian)
        *   6.1.2. Ekstraksi Informasi dari Fitur Tanggal/Waktu (jika ada dan relevan, misal: `bulan_kontak_terakhir`, `hari_kontak_terakhir` bisa diubah jadi fitur siklikal atau interaksi)
        *   6.1.3. Agregasi Fitur (jika ada data transaksional atau sekuensial yang lebih detail)
        *   6.1.4. Binning Fitur Numerik menjadi Kategorikal (misal, kelompok usia)
    *   6.2. (Opsional) Pengurangan Dimensi (PCA, t-SNE) - *Gunakan dengan hati-hati karena bisa mengurangi interpretasi.*
    *   6.3. Seleksi Fitur (Feature Selection)
        *   6.3.1. Metode Filter (Korelasi, Chi-squared, ANOVA F-value)
        *   6.3.2. Metode Wrapper (Recursive Feature Elimination - RFE)
        *   6.3.3. Metode Embedded (Lasso Regularization, Feature Importance dari Tree-based Models)
    *   6.4. Verifikasi Dataset Setelah Feature Engineering & Selection

7.  **[Pemodelan (Modeling)](#7-pemodelan-modeling)**
    *   7.1. Pemisahan Fitur (X) dan Target (y) untuk Dataset Training
    *   7.2. (Opsional) Penanganan Ketidakseimbangan Kelas (Imbalanced Class Handling) pada Data Training
        *   7.2.1. Oversampling (SMOTE, ADASYN)
        *   7.2.2. Undersampling (Random Undersampling, Tomek Links, NearMiss)
        *   7.2.3. Kombinasi Over- & Undersampling
        *   7.2.4. Penggunaan Bobot Kelas (Class Weighting) dalam Model
    *   7.3. **Pemilihan Model Baseline & Eksperimen Awal**
        *   7.3.1. Logistic Regression
        *   7.3.2. K-Nearest Neighbors (KNN)
        *   7.3.3. Support Vector Machine (SVM)
        *   7.3.4. Decision Tree Classifier
        *   7.3.5. Random Forest Classifier
        *   7.3.6. Gradient Boosting Machines (XGBoost, LightGBM, CatBoost)
        *   7.3.7. (Opsional) Naive Bayes
        *   *Catatan: Lakukan evaluasi awal dengan cross-validation pada data training untuk setiap model dengan parameter default atau parameter sederhana.*
    *   7.4. **Evaluasi Model Awal Menggunakan Cross-Validation (AUC)**
        *   7.4.1. Fungsi untuk Cross-Validation & Pelaporan Skor AUC
        *   7.4.2. Perbandingan Performa Model Baseline
    *   7.5. **Pemilihan Model Kandidat Terbaik** (berdasarkan hasil evaluasi awal)

8.  **[Penyetelan Hiperparameter (Hyperparameter Tuning)](#8-penyetelan-hiperparameter-hyperparameter-tuning)**
    *   *Fokus pada model kandidat terbaik dari tahap sebelumnya.*
    *   8.1. Strategi Tuning (GridSearchCV, RandomizedSearchCV, Bayesian Optimization)
    *   8.2. Penyetelan Hiperparameter untuk Model 1 (misal, XGBoost)
    *   8.3. Penyetelan Hiperparameter untuk Model 2 (misal, LightGBM)
    *   8.4. (Jika ada model lain yang menjanjikan)
    *   8.5. Penyimpanan Parameter Terbaik untuk Setiap Model

9.  **[Evaluasi Model Final & Pemilihan Model Terbaik](#9-evaluasi-model-final--pemilihan-model-terbaik)**
    *   9.1. Melatih Model Kandidat dengan Parameter Terbaik pada Seluruh Data Training (atau subset jika menggunakan validasi terpisah)
    *   9.2. Evaluasi pada Data Training (untuk cek overfitting) - Metrik: AUC, Akurasi, Presisi, Recall, F1-score, Confusion Matrix, ROC Curve.
    *   9.3. **Prediksi pada Data Validasi (`df_val_processed`)**
    *   9.4. **Evaluasi Performa pada Data Validasi (jika memiliki label lokal untuk validasi, jika tidak, ini akan jadi prediksi akhir)**
        *   *Jika panitia tidak memberikan label untuk validation set, bagian ini tidak bisa dilakukan secara langsung. Skor AUC akan didapat dari Leaderboard setelah submission.*
    *   9.5. Analisis Error (Error Analysis) - (Opsional, jika memungkinkan)
        *   Melihat kasus-kasus di mana model salah prediksi.
    *   9.6. Interpretasi Model (Model Interpretability)
        *   9.6.1. Feature Importance (untuk model tree-based)
        *   9.6.2. (Opsional) SHAP Values atau LIME
    *   9.7. Pemilihan Model Final untuk Submission berdasarkan Performa dan Stabilitas.

10. **[Pembuatan File Submission](#10-pembuatan-file-submission)**
    *   10.1. Memastikan Model Final Dilatih dengan Seluruh Data Training yang Tersedia (jika belum)
    *   10.2. Melakukan Prediksi Probabilitas pada Dataset Validasi (`df_val_processed`) menggunakan Model Final
    *   10.3. Membuat DataFrame Submission Sesuai Format yang Diminta
        *   Kolom: `customer_number`, `berlangganan_deposito` (berisi probabilitas kelas 1)
    *   10.4. Menyimpan File Submission ke Format `.csv` dengan Nama Sesuai Guideline (`DCM_DMU_2025_Model_CubitAkuDong.csv`)
    *   10.5. Verifikasi Cepat File Submission (Jumlah Baris, Nama Kolom, Rentang Nilai Probabilitas)

11. **[Kesimpulan & Potensi Pengembangan Lanjutan](#11-kesimpulan--potensi-pengembangan-lanjutan)**
    *   11.1. Ringkasan Hasil Terbaik (Model, Skor AUC jika diketahui dari leaderboard sementara)
    *   11.2. Pembelajaran Utama dari Proyek
    *   11.3. Tantangan yang Dihadapi & Bagaimana Mengatasinya
    *   11.4. Ide untuk Pengembangan atau Peningkatan Model di Masa Depan (jika ada waktu lebih)

---
