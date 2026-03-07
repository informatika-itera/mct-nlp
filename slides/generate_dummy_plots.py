import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set visual style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("muted")

# 1. Generate Dummy Confusion Matrix
plt.figure(figsize=(7, 6))
# Create fake data that looks realistic for a decent model
cm_data = np.array([
    [752, 12, 15, 21],  # neutral
    [10, 680, 25, 45],  # violence 
    [15, 30, 710, 35],  # racist
    [20, 25, 35, 720]   # harassment
])
labels = ['neutral', 'violence', 'racist', 'harassment']

# Create heatmap
ax = sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', 
                 xticklabels=labels, yticklabels=labels,
                 annot_kws={'size': 14, 'weight': 'bold'})

plt.title('Confusion Matrix (LightGBM)', fontsize=16, pad=15)
plt.ylabel('True Class', fontsize=14, weight='bold')
plt.xlabel('Predicted Class', fontsize=14, weight='bold')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('logos/dummy_cm.png', dpi=300, bbox_inches='tight')
print("Generated dummy_cm.png")

# 2. Generate Dummy Feature Importance
plt.figure(figsize=(9, 6))

features = ['cina', 'anjing', 'babi', 'miskin', 'yatim', 'tolol', 'bocah', 'jawa', 'goblok', 'mati']
importance = [0.18, 0.15, 0.12, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03]

# Create horizontal bar chart
sns.barplot(x=importance, y=features, color='#3498db')

plt.title('Top 10 Feature Importance (Toxic Indicators)', fontsize=16, pad=15)
plt.xlabel('Importance Score (Information Gain)', fontsize=14)
plt.ylabel('TF-IDF Token', fontsize=14)

# Add values on bars
for i, v in enumerate(importance):
    plt.text(v + 0.005, i + 0.1, str(v), color='black', fontsize=10)

plt.tight_layout()
plt.savefig('logos/dummy_fi.png', dpi=300, bbox_inches='tight')
print("Generated dummy_fi.png")
