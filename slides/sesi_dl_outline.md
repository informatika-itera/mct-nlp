# Outline Slide — Sesi Deep Learning
**Judul**: *Deep Learning for NLP — From RNN to Transformers*

---

## Slide Pendahuluan
- **Capaian Pembelajaran** — apa yang peserta bisa lakukan setelah sesi ini
- **Mengapa Deep Learning?** — keterbatasan Traditional ML (TF-IDF + SVM tidak menangkap urutan/konteks kata), motivasi beralih ke DL
- **Disclaimer dataset** — konten mental health (depresi, suicidal) untuk edukasi

---

## Section 1: Rekap & Jembatan dari Sesi Sebelumnya
- Traditional ML pipeline vs. Deep Learning pipeline (perbandingan visual)
- Dari fitur manual (TF-IDF) ke representasi terpelajar (*learned representations*)
- Dataset shift: dari Indonesian Chat (sesi 1) ke *Mental Health Sentiment* (7 kelas)

---

## Section 2: Dasar-Dasar Deep Learning
- **Neuron buatan**: input, weight, bias, fungsi aktivasi → output; analogi neuron biologis
- **Fungsi Aktivasi**: ReLU, Sigmoid, Tanh, Softmax — peran dan kapan digunakan
- **Layer**: Input layer, Hidden layer(s), Output layer — konsep *depth* pada deep learning
- **Weight & Bias**: parameter yang dipelajari model; inisialisasi (Xavier, He)
- **Loss Function**: mengukur seberapa salah prediksi — Cross-Entropy untuk klasifikasi
- **Feedforward Network (MLP)**: aliran data dari kiri ke kanan tanpa loop
- **Overfitting & Regularisasi**: Dropout, L2 regularization, Early Stopping

---

## Section 3: Dari MLP ke Model Sekuensial (RNN, LSTM, BiLSTM)
- Keterbatasan MLP untuk teks: tidak mempertimbangkan urutan kata
- RNN: konsep hidden state sebagai "memori" — gambaran sederhana, bukan mendalam
- Masalah vanishing gradient pada RNN → solusi: LSTM dengan gating mechanism
- BiLSTM: baca urutan dari dua arah — ringkasan arsitektur yang dipakai di workshop
- Word Embeddings: dari one-hot → trainable embedding layer

---

## Section 4: Attention Mechanism
- Motivasi: BiLSTM hanya bergantung pada *last hidden state* — informasi hilang untuk teks panjang
- Intuisi Attention: model *memilih* bagian teks yang paling relevan
- Formula Bahdanau Attention: $\alpha_t = \text{softmax}(\tanh(W \cdot h_t))$, context vector
- **Explainability**: visualisasi heatmap kata mana yang di-*attend* model
- Arsitektur BiLSTM + Attention vs. BiLSTM murni

---

## Section 5: Transfer Learning & Transformers
- Paradigma transfer learning: pre-train → fine-tune
- Transformers: Self-Attention, positional encoding, [CLS] token
- DistilBERT: versi distilasi BERT (40% lebih kecil, 97% akurasi BERT, 60% lebih cepat)
- Fine-tuning DistilBERT untuk 7 kelas: freeze backbone → tambah head FC
- Perbandingan parameter: BiLSTM (~7M) vs DistilBERT (~67M)

---

## Section 6: CNN untuk Teks vs Attention / Transformer
- **TextCNN**: konvolusi 1D di atas embedding — menangkap n-gram lokal (bigram, trigram) secara paralel
- Arsitektur: Embedding → Conv1D (multi-filter size) → MaxPooling → FC
- Kekuatan CNN: cepat, hemat memori, efektif untuk fitur lokal
- **Kelemahan CNN**: tidak menangkap dependensi jarak jauh (*long-range dependencies*)
- Attention/Transformer: mengatasi limitasi CNN dan RNN — setiap token bisa melihat semua token lain secara langsung
- Tabel perbandingan: CNN vs RNN/LSTM vs Attention vs Transformer (kecepatan, konteks, memori, interpretabilitas)

---

## Section 7: Data Pipeline & Preprocessing
- Pipeline pembersihan teks: lowercase → hapus URL → HTML → non-alpha → whitespace
- Vocabulary building: `<PAD>`, `<UNK>`, text → integer indices
- Stratified train/val/test split (80/10/10) — kenapa stratified penting untuk kelas imbalanced
- PyTorch `Dataset` & `DataLoader` — batching dan padding

---

## Section 8: Training Fundamentals
- **Forward pass**: aliran data dari input → hidden layers → output → loss
- **Backward pass & Backpropagation**: chain rule, komputasi gradien otomatis (`loss.backward()`)
- **Gradient Descent**: update parameter $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$; perbedaan Batch, Stochastic, Mini-batch GD
- **Optimizer**: SGD vs Adam vs AdamW — intuisi momentum dan adaptive learning rate
- **Scheduler**: StepLR, ReduceLROnPlateau, WarmupScheduler — mengapa learning rate perlu diturunkan seiring training
- Early stopping: monitor val loss, patience=2–3

---

## Section 9: Training & Evaluasi
- Training loop end-to-end: forward → loss → backward → optimizer step → scheduler step
- Metrik evaluasi: Accuracy, F1-Macro, F1-Weighted, Confusion Matrix
- Weighted CrossEntropyLoss untuk class imbalance — bobot kelas berbanding terbalik frekuensi
- Perbandingan ketiga model: tabel + bar chart visual

---

## Section 10: Deployment — Dari Model ke Aplikasi
- Model saving: `.pt` file + `vocab.json` + `label_encoder.json`
- Attention heatmap di UI (hanya BiLSTM+Attention)

---

## Section 11: Arsitektur Kode Modular
- Penjelasan pemisahan `config.py`, `dataset.py`, `models.py`, `train.py`, `train_run.py`
- Notebook sebagai *runner & viewer*, bukan tempat logika utama

---

## Penutup
- Perbandingan final tiga pendekatan: Traditional ML vs BiLSTM vs BERT
- Kapan pilih mana? (resource, data size, interpretability)
- Referensi: paper *Attention is All You Need*, DistilBERT, HuggingFace

---

> **Estimasi total**: ~30–40 frame
