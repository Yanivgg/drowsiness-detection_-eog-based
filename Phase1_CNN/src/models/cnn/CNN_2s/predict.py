"""
General prediction script for CNN 2s window model
Can be used for any dataset (train/val/test)
"""

import os
import keras
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

# Paths - can be modified as needed
data_dir = './../../../data/files/'
model_path = './model.h5'
file_sets_path = './../../../data/file_sets.mat'

# Load test files dynamically from file_sets.mat
print("Loading test set from file_sets.mat...")
file_sets = spio.loadmat(file_sets_path)
files_test = [f[0] for f in file_sets['files_test'].flatten()]
files_to_predict = files_test
print(f"Test files: {files_to_predict}")


def my_generator(data, targets, sample_list, shuffle=False):
    """Generator for prediction data"""
    if shuffle:
        random.shuffle(sample_list)
    while True:
        for batch in batch_generator(sample_list, batch_size):
            batch_data1 = []
            batch_targets = []
            for sample in batch:
                [f, s, b, e] = sample
                sample_label = targets[f][0][s]
                # Get both EOG channels (E1 and E2)
                sample_x1 = data[f][0][b:e+1]  # E1
                sample_x2 = data[f][1][b:e+1]  # E2
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
    for j in range(len(targets[i][0])):
        mid = j * prec
        mid += w_len
        wnd_begin = mid - w_len
        wnd_end = mid + w_len - 1
        sample_list[i].append([i, j, wnd_begin, wnd_end])

# Predict
print("=====================")
print("Making predictions...")
all_y_pred = []
all_y_true = []

for j in range(len(data)):
    f = files_to_predict[j]
    print(f"Processing {f}...")
    generator = my_generator(data, targets, sample_list[j], shuffle=False)
    y_pred = model.predict(generator, steps=int(math.ceil((len(sample_list[j],) + 0.0) / batch_size)))
    
    all_y_pred.extend(np.argmax(y_pred, axis=1).flatten())
    all_y_true.extend(targets[j][0])

# Save predictions
spio.savemat('./predictions_output.mat',
             mdict={'y_true': all_y_true, 'y_pred': all_y_pred, 'files': files_to_predict})

print("=====================")
print("Predictions saved to predictions_output.mat")
print(f"Total predictions: {len(all_y_pred)}")

