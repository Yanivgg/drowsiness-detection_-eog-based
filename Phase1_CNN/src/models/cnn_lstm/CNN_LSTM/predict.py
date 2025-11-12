"""
General prediction script for CNN-LSTM model
Can be used for any dataset
"""

import os
import keras
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

# Paths - can be modified as needed
data_dir = './../../data/files/'
model_path = './model.h5'
file_sets_path = './../../data/file_sets.mat'

# Load test files dynamically from file_sets.mat
print("Loading test set from file_sets.mat...")
file_sets = spio.loadmat(file_sets_path)
files_test = [f[0] for f in file_sets['files_test'].flatten()]
files_to_predict = files_test
print(f"Test files: {files_to_predict}")


def my_generator(data, targets, sample_list, shuffle=False, batch_size=32):
    """Generator for prediction data"""
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
                sample_xx1 = data[f][0][b:e]  # E1
                sample_xx2 = data[f][1][b:e]  # E2
                sample_xx = np.concatenate((sample_xx1, sample_xx2), axis=2)
                sample_yy = targets[f][0][b:e]
                
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


print("=====================")
print("Loading model...")
model = load_model(model_path)

print("=====================")
print(f"Reading dataset: {files_to_predict}")
(data, targets, N_samples) = load_data(data_dir, files_to_predict, w_len)

# Create sample list
sample_list = []
for i in range(len(targets)):
    sample_list.append([])
    for j in range(len(targets[i][0]) - w_len):
        sample_list[i].append([i, j, j + w_len])

# Predict
print("=====================")
print("Making predictions...")
all_y_pred = []
all_y_true = []

for j in range(len(data)):
    f = files_to_predict[j]
    print(f"Processing {f}...")
    generator = my_generator(data, targets, sample_list[j], shuffle=False, batch_size=batch_size)
    n_steps = int(math.ceil((len(sample_list[j]) + 0.0) / batch_size / skip))
    y_pred = model.predict(generator, steps=n_steps)
    
    # Flatten predictions
    y_pred_flat = []
    for seq in y_pred:
        for pred in seq:
            y_pred_flat.append(np.argmax(pred))
    
    all_y_pred.extend(y_pred_flat[:len(targets[j][0])])
    all_y_true.extend(targets[j][0][:len(y_pred_flat)])

# Save predictions
spio.savemat('./predictions_output.mat',
             mdict={'y_true': all_y_true, 'y_pred': all_y_pred, 'files': files_to_predict})

print("=====================")
print("Predictions saved to predictions_output.mat")
print(f"Total predictions: {len(all_y_pred)}")

