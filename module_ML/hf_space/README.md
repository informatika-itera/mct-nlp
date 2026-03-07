---
title: Deteksi Toksisitas Chat Gamer Indonesia
emoji: 🎮
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 5.20.1
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
---

# 🎮 Deteksi Toksisitas Chat Gamer Indonesia

Model NLP untuk mendeteksi apakah chat dalam game online Indonesia bersifat **toxic** atau **non-toxic**.

## Fitur
- Custom preprocessing untuk slang & leetspeak gamer Indonesia
- TF-IDF vectorization via PyCaret
- Model klasifikasi otomatis (AutoML via PyCaret)

## Cara Penggunaan
Ketik chat game di text box, lalu klik **Submit** untuk melihat hasil prediksi.
