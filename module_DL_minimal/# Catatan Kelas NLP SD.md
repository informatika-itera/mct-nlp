# Catatan Kelas NLP SD

Yang mempengaruhi training performance (metric performance):
- Learning Rate: lompatan pembelajaran (penalti yang dikenakan berdasarkan loss)
- Checkpointing: menyimpan model terbaik berdasarkan val_loss
- Early stopping: menghentikan training jika val_loss tidak membaik selama beberapa epoch
- Scheduler: mengubah learning rate selama training (misal: menurunkan learning rate jika val_loss tidak membaik)
- Optimizer: algoritma untuk memperbarui bobot model (misal: AdamW, Adam, SGD, Muon)
- Weight Decay: regularisasi untuk mencegah overfitting (penalti pada bobot besar)
- Dropout: teknik regularisasi untuk mencegah overfitting dengan mengabaikan beberapa neuron selama training