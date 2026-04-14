---
title: Mental Health Sentiment Analyzer
emoji: 🧠
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: mit
---

# 🧠 Mental Health Sentiment Analyzer

Klasifikasi status kesehatan mental dari teks menggunakan tiga model deep learning.

## Deskripsi

Aplikasi ini menganalisis teks dan memprediksi salah satu dari **7 status kesehatan mental**:

- **Normal** — kondisi mental sehat
- **Depression** — depresi
- **Suicidal** — pikiran bunuh diri
- **Anxiety** — kecemasan
- **Stress** — stres
- **Bipolar** — gangguan bipolar
- **Personality Disorder** — gangguan kepribadian

## Model yang Tersedia

| Model | Deskripsi |
|-------|-----------|
| **BiLSTM** | Bidirectional LSTM, embedding train from scratch |
| **BiLSTM+Attention** | BiLSTM + mekanisme attention dengan visualisasi heatmap |
| **DistilBERT** | Fine-tuned DistilBERT (akurasi tertinggi) |

## Cara Menggunakan

1. Pilih model dari dropdown
2. Masukkan teks (dalam bahasa Inggris)
3. Klik **Analisis**
4. Lihat prediksi kelas dan confidence tiap kelas
5. Untuk BiLSTM+Attention, lihat attention heatmap untuk interpretasi

## Dataset

[Sentiment Analysis for Mental Health](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health) — 7 kelas status kesehatan mental dari berbagai platform media sosial.

## Disclaimer

> ⚠️ Ini adalah alat demonstrasi akademis untuk Workshop Pemrosesan Bahasa Alami (PBA) di Institut Teknologi Sumatera (ITERA). **Bukan alat diagnosis medis.** Jika Anda atau orang di sekitar Anda membutuhkan bantuan kesehatan mental, hubungi profesional kesehatan mental atau layanan krisis kesehatan mental di negara Anda.

---
_Workshop PBA — Institut Teknologi Sumatera (ITERA), 2026_
