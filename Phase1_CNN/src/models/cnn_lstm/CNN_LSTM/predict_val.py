"""
Validation prediction script for CNN-LSTM model
"""

import os
import keras
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score, confusion_matrix
import math
import random
import numpy as np
import scipy.io as spio
from keras.utils import to_categorical
from keras.models import load_model
from collections import Counter

import sys
sys.path.append("..")
from loadData import *
from utils import *

# Configuration
batch_size = 32
fs = 200
w1 = 200
step = 50
w_len = (20 * 50 + 200)
data_dim = w1
n_cl = 2
skip = 10

# Paths
data_dir = './../../data/files/'
f_set = './../../data/file_sets.mat'
model_path = './model.h5'

# Load file splits
mat = spio.loadmat(f_set)

files_val = []
tmp = mat['files_val']
for i in range(len(tmp)):
    file = [str(''.join(l)) for la in tmp[i] for l in la]
    files_val.extend(file)

print("=====================")
print("Loading model...")
model = load_model(model_path)

print("=====================")
print("Reading validation dataset:")
(data_val, targets_val, N_samples_val) = load_data(data_dir, files_val, w_len)


def my_generator(data_val, targets_val, sample_list, shuffle=False, batch_size=32):
    """Generator for validation data"""
    if shuffle:
        random.shuffle(sample_list)
    sample_list_skipped = sample_list[::skip]
    
    while True:
        for batch in batch_generator(sample_list_skipped, batch_size):
            batch_data = []
            batch_targets = []
            for sample in batch:
                [f, b, e] = sample
                # Get both EOG channels (E1 and E2)
                sample_xx1 = data_val[f][0][b:e]  # E1
                sample_xx2 = data_val[f][1][b:e]  # E2
                sample_xx = np.concatenate((sample_xx1, sample_xx2), axis=2)
                sample_yy = targets_val[f][0][b:e]
                
                sample_x = []
                sample_y = []
                z = 0
                while z < len(sample_xx) - w1:
                    sample_x.append(sample_xx[z:z + w1])
                    tmp_lbl = sample_yy[z + w1 // 2 - 20:z + w1 // 2 + 20]
                    a = Counter(tmp_lbl)
                    r = a.most_common(1)[0][0]
                    sample_y.append(r)
                    z += step
                
                batch_data.append(sample_x)
                batch_targets.append(sample_y)
            
            batch_data = np.array(batch_data)
            batch_targets = np.array(batch_targets)
            batch_targets = np.expand_dims(batch_targets, axis=-1)
            batch_targets = to_categorical(batch_targets, n_cl)
            batch_data = batch_data / 100
            batch_data = np.clip(batch_data, -1, 1)
            yield batch_data, batch_targets


# Create sample list for validation
sample_list_val = []
for i in range(len(targets_val)):
    sample_list_val.append([])
    for j in range(len(targets_val[i][0]) - w_len):
        sample_list_val[i].append([i, j, j + w_len])

# Predict on validation set
print("=====================")
print("Predicting on validation set...")
val_y_ = []
val_y = []

for j in range(len(data_val)):
    f = files_val[j]
    print(f"Processing {f}...")
    generator_val = my_generator(data_val, targets_val, sample_list_val[j], shuffle=False, batch_size=batch_size)
    n_steps = int(math.ceil((len(sample_list_val[j]) + 0.0) / batch_size / skip))
    y_pred = model.predict(generator_val, steps=n_steps)
    
    # Flatten predictions
    y_pred_flat = []
    for seq in y_pred:
        for pred in seq:
            y_pred_flat.append(np.argmax(pred))
    
    val_y_.extend(y_pred_flat[:len(targets_val[j][0])])
    val_y.extend(targets_val[j][0][:len(y_pred_flat)])

# Calculate metrics
kappa_overall = cohen_kappa_score(val_y, val_y_)
accuracy = accuracy_score(val_y, val_y_)
f1 = f1_score(val_y, val_y_, average='binary')
conf_matrix = confusion_matrix(val_y, val_y_)
kappa_per_class = kappa_metric(val_y, val_y_, n_cl)

print("=====================")
print("Validation Results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Cohen's Kappa (overall): {kappa_overall:.4f}")
print(f"Cohen's Kappa per class: {kappa_per_class}")
print("Confusion Matrix:")
print(conf_matrix)

# Save results
spio.savemat('./validation_results.mat',
             mdict={'val_y': val_y, 'val_y_': val_y_,
                    'accuracy': accuracy, 'f1_score': f1, 'kappa_overall': kappa_overall,
                    'kappa_per_class': kappa_per_class, 'confusion_matrix': conf_matrix,
                    'files_val': files_val})

print("=====================")
print("Results saved to validation_results.mat")

