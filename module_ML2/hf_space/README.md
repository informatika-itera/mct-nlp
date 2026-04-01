---
title: Analisis Sentimen Review Film IMDB
emoji: 🎬
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 5.20.1
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
---

# 🎬 Analisis Sentimen Review Film IMDB

Model NLP untuk mendeteksi apakah sebuah review film bersifat **positive** atau **negative**.

## Fitur
- Custom preprocessing untuk HTML tags & kontraksi bahasa Inggris
- TF-IDF vectorization via PyCaret
- Model klasifikasi otomatis (AutoML via PyCaret)

## Dataset
- **IMDB Dataset of 50K Movie Reviews** — 25K positive, 25K negative
- Binary sentiment classification

## Cara Penggunaan
Ketik review film di text box, lalu klik **Submit** untuk melihat hasil prediksi sentimen.
