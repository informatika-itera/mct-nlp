# Spesifikasi Workshop NLP - Sesi 1: Traditional Machine Learning & AutoML

**Konteks:**
Workshop *hands-on* berdurasi 2 jam untuk mahasiswa Sains Data. Fokus utama adalah pemahaman konsep *pipeline* NLP, ekstraksi fitur (TF-IDF/Bag of Words), dan evaluasi model *traditional ML*, tanpa menghabiskan waktu pada *boilerplate code*.

## 1. Stack Teknologi
* **Pendekatan:** Traditional Machine Learning via AutoML
* **Library Utama:** `pycaret` (untuk *training* & *evaluation*), `pandas` (manipulasi data), `re` / `nltk` / `Sastrawi` (untuk *custom preprocessing* bahasa Indonesia).

## 2. Detail Dataset
* **Sumber:** Kaggle - Indonesian Chat Dataset (Roblox & Minecraft)
* **URL:** `https://www.kaggle.com/datasets/jprestiliano/indonesian-chat-dataset`
* **Karakteristik:** Teks *chat gamer* yang sangat kotor. Penuh dengan *leetspeak* (angka sebagai huruf, misal: "4nj1n9"), singkatan ekstrem, dan *slang* lokal.
* **Tugas:** *Multiclass Classification* (4 Label: *Neutral, Violence, Racist, Harassment*).

## 3. Aturan Arsitektur Kode (CRITICAL INSTRUCTION)
Saya TIDAK ingin eksekusi kode menumpuk di dalam satu file Jupyter Notebook. Arsitektur proyek harus **berbasis skrip modular (`.py`)**. 

* **File `.py` (Core Logic):** Seluruh logika ekstraksi, *preprocessing*, inisialisasi PyCaret (`setup`), eksekusi `compare_models`, *tuning*, dan ekspor model harus dibuat di dalam *script* Python yang rapi dan terpisah (misal: `config.py`, `preprocess.py`, `train.py`).
* **File `.ipynb` (Runner & Viewer):** Jupyter Notebook HANYA difungsikan sebagai *runner* presentasi. Notebook ini hanya boleh mengimpor fungsi/kelas dari file `.py`, menjalankan fungsinya (satu baris pemanggilan), dan menampilkan *output* visualnya (seperti tabel komparasi algoritma PyCaret atau grafik evaluasi).

## 4. Alur Pipeline yang Diminta
Buatkan kerangka kodenya dengan alur berikut:
1.  **Custom Text Cleaning:** Fungsi pembersihan teks khusus untuk menangani *leetspeak* angka dan *slang* *gamer* Indonesia.
2.  **AutoML Setup:** Konfigurasi `pycaret.classification.setup` dengan parameter teks untuk melakukan TF-IDF secara otomatis.
3.  **Model Arena:** Menjalankan komparasi semua model (termasuk LightGBM/CatBoost/SVM) untuk mencari akurasi dan F1-Score tertinggi.
4.  **Evaluation:** Memanggil visualisasi seperti *Confusion Matrix* dan *Feature Importance* (kata apa yang memicu label *toxic*).
5.  **Finalize:** Ekspor *pipeline* utuh menjadi `.pkl`.