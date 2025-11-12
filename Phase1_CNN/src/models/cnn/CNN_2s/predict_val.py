"""
Validation prediction script for CNN 2s window model
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

import sys
sys.path.append("../..")
from loadData import *
from utils import *

# Configuration
batch_size = 200
fs = 200
w_len = 1 * fs  # half window size
data_dim = w_len * 2
prec = 1
n_cl = 2  # binary classification

# Paths
data_dir = './../../../data/files/'
f_set = './../../../data/file_sets.mat'
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


def my_generator(data_val, targets_val, sample_list, shuffle=False):
    """Generator for validation data"""
    if shuffle:
        random.shuffle(sample_list)
    while True:
        for batch in batch_generator(sample_list, batch_size):
            batch_data1 = []
            batch_targets = []
            for sample in batch:
                [f, s, b, e] = sample
                sample_label = targets_val[f][0][s]
                # Get both EOG channels (E1 and E2)
                sample_x1 = data_val[f][0][b:e+1]  # E1
                sample_x2 = data_val[f][1][b:e+1]  # E2
                sample_x = np.concatenate((sample_x1, sample_x2), axis=2)
                batch_data1.append(sample_x)
                batch_targets.append(sample_label)
            batch_data1 = np.stack(batch_data1, axis=0)
            batch_targets = np.array(batch_targets)
            batch_targets = to_categorical(batch_targets, n_cl)
            # Normalization
            batch_data1 = (batch_data1) / 100
            batch_data1 = np.clip(batch_data1, -1, 1)
            yield batch_data1, batch_targets


# Create sample list for validation
sample_list_val = []
for i in range(len(targets_val)):
    sample_list_val.append([])
    for j in range(len(targets_val[i][0])):
        mid = j * prec
        mid += w_len
        wnd_begin = mid - w_len
        wnd_end = mid + w_len - 1
        sample_list_val[i].append([i, j, wnd_begin, wnd_end])

# Predict on validation set
print("=====================")
print("Predicting on validation set...")
val_y_ = []
val_y = []
loss_val_tmp = []

for j in range(len(data_val)):
    f = files_val[j]
    print(f"Processing {f}...")
    generator_val = my_generator(data_val, targets_val, sample_list_val[j], shuffle=False)
    scores = model.evaluate(generator_val, steps=int(math.ceil((len(sample_list_val[j],) + 0.0) / batch_size)))
    generator_val = my_generator(data_val, targets_val, sample_list_val[j], shuffle=False)
    y_pred = model.predict(generator_val, steps=int(math.ceil((len(sample_list_val[j],) + 0.0) / batch_size)))
    
    loss_val_tmp.append(scores[0])
    val_y_.extend(np.argmax(y_pred, axis=1).flatten())
    val_y.extend(targets_val[j][0])

# Calculate metrics
loss_val = np.mean(loss_val_tmp)
kappa_per_class = kappa_metric(val_y, val_y_, n_cl)
kappa_overall = cohen_kappa_score(val_y, val_y_)
accuracy = accuracy_score(val_y, val_y_)
f1 = f1_score(val_y, val_y_, average='binary')
conf_matrix = confusion_matrix(val_y, val_y_)

print("=====================")
print("Validation Results:")
print(f"Loss: {loss_val:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Cohen's Kappa (overall): {kappa_overall:.4f}")
print(f"Cohen's Kappa per class: {kappa_per_class}")
print("Confusion Matrix:")
print(conf_matrix)

# Save results
spio.savemat('./validation_results.mat',
             mdict={'val_y': val_y, 'val_y_': val_y_, 'loss_val': loss_val,
                    'accuracy': accuracy, 'f1_score': f1, 'kappa_overall': kappa_overall,
                    'kappa_per_class': kappa_per_class, 'confusion_matrix': conf_matrix,
                    'files_val': files_val})

print("=====================")
print("Results saved to validation_results.mat")

