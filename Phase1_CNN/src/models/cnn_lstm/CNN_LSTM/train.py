"""
Training script for CNN-LSTM model
Hybrid architecture for temporal microsleep detection
Binary classification with 2 EOG channels
"""

import os
import keras
from keras.layers import concatenate
from sklearn.metrics import cohen_kappa_score

import math
import random
from keras import optimizers
import numpy as np
import scipy.io as spio
from sklearn.metrics import f1_score, accuracy_score
np.random.seed(0)

from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Layer, Dense, Dropout, Input, Activation, TimeDistributed, Reshape
from keras.layers import GRU, Bidirectional
from keras.layers import Conv1D, Conv2D, MaxPooling2D, Flatten, BatchNormalization, LSTM, ZeroPadding2D, GlobalAveragePooling2D
from keras.callbacks import History
from keras.models import Model
from collections import Counter
from sklearn.utils import class_weight

from myModel import build_model

import sys
sys.path.append("..")
from loadData import *
from utils import *

# Configuration
n_rec = 8
batch_size = 32  # Smaller batch for LSTM
n_ep = 3
fs = 200
w1 = 200  # 1 second window for each sequence element
step = 50  # 0.25 s step
w_len = (20 * 50 + 200)  # Sequence length
data_dim = w1
half_prec = 50
prec = 1
n_cl = 2  # binary classification

# Paths
data_dir = './../../data/files/'
f_set = './../../data/file_sets.mat'

create_tmp_dirs(['./models/', './predictions/'])

# Load file splits
mat = spio.loadmat(f_set)

files_train = []
files_val = []
files_test = []

skip = 10

tmp = mat['files_train']
for i in range(len(tmp)):
    file = [str(''.join(l)) for la in tmp[i] for l in la]
    files_train.extend(file)

tmp = mat['files_val']
for i in range(len(tmp)):
    file = [str(''.join(l)) for la in tmp[i] for l in la]
    files_val.extend(file)

tmp = mat['files_test']
for i in range(len(tmp)):
    file = [str(''.join(l)) for la in tmp[i] for l in la]
    files_test.extend(file)


def my_generator(data_train, targets_train, sample_list, shuffle=True, batch_size=32, class_weights=None):
    """Generator for CNN-LSTM training data with sample weights support for Keras 3.x"""
    if shuffle:
        random.shuffle(sample_list)
    sample_list_skipped = sample_list[::skip]
    
    while True:
        for batch in batch_generator(sample_list_skipped, batch_size):
            batch_data = []
            batch_targets = []
            batch_sample_weights = []
            for sample in batch:
                [f, b, e] = sample
                # Get both EOG channels (E1 and E2)
                sample_xx1 = data_train[f][0][b:e]  # E1
                sample_xx2 = data_train[f][1][b:e]  # E2
                sample_xx = np.concatenate((sample_xx1, sample_xx2), axis=2)
                sample_yy = targets_train[f][0][b:e]
                
                sample_x = []
                sample_y = []
                sample_w = []
                z = 0
                while z < len(sample_xx) - w1:
                    sample_x.append(sample_xx[z:z + w1])
                    tmp_lbl = sample_yy[z + w1 // 2 - 20:z + w1 // 2 + 20]
                    a = Counter(tmp_lbl)
                    r = a.most_common(1)[0][0]
                    sample_y.append(r)
                    if class_weights is not None:
                        sample_w.append(class_weights[int(r)])
                    z += step
                
                batch_data.append(sample_x)
                batch_targets.append(sample_y)
                if class_weights is not None:
                    batch_sample_weights.append(sample_w)
            
            batch_data = np.array(batch_data)
            batch_targets = np.array(batch_targets)
            batch_targets = np.expand_dims(batch_targets, axis=-1)
            # Store sample weights before converting targets
            if class_weights is not None:
                batch_sample_weights = np.array(batch_sample_weights)
            batch_targets = to_categorical(batch_targets, n_cl)
            # Normalization
            batch_data = batch_data / 100
            batch_data = np.clip(batch_data, -1, 1)
            # Yield with sample weights if provided
            if class_weights is not None:
                yield batch_data, batch_targets, batch_sample_weights
            else:
                yield batch_data, batch_targets


n_channels = 2  # 2 EOG channels

# Calculate class weights
st0 = classes_global(data_dir, files_train)
cls = np.arange(n_cl)
cl_w = class_weight.compute_class_weight('balanced', classes=cls, y=st0)
print("=====================")
print("class weights ")
print(cl_w)

print("=====================")
print("Reading training dataset:")
(data_train, targets_train, N_samples) = load_data(data_dir, files_train, w_len)
print('N_samples ', N_samples)
N_batches = int(math.ceil((N_samples + 0.0) / batch_size)) // skip
print('N batches ', N_batches)

print("=====================")
print("Reading validation dataset:")
(data_val, targets_val, N_samples_val) = load_data(data_dir, files_val, w_len)

# Create sample lists
sample_list = []
for i in range(len(targets_train)):
    for j in range(len(targets_train[i][0]) - w_len):
        sample_list.append([i, j, j + w_len])

sample_list_val = []
for i in range(len(targets_val)):
    sample_list_val.append([])
    for j in range(len(targets_val[i][0]) - w_len):
        sample_list_val[i].append([i, j, j + w_len])

# Build model
ordering = 'channels_last'
keras.backend.set_image_data_format(ordering)

model = build_model(data_dim, n_channels, n_cl)
Nadam = optimizers.Nadam()
model.compile(optimizer='Nadam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

history = History()

acc_val = []
acc_tr = []
loss_tr = []
loss_val = []

# Training loop
for i in range(n_ep):
    print("Epoch = " + str(i))
    generator_train = my_generator(data_train, targets_train, sample_list, batch_size=batch_size, class_weights=cl_w)
    model.fit(generator_train, steps_per_epoch=N_batches, epochs=1, verbose=1, callbacks=[history], initial_epoch=0)

    acc_tr.append(history.history['accuracy'])
    loss_tr.append(history.history['loss'])

    # Validation
    val_y_ = []
    val_y = []
    for j in range(len(data_val)):
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

    kappa = cohen_kappa_score(val_y, val_y_)
    acc_val.append(kappa)
    print("K val = ", kappa)

    model.save('./models/model_ep' + str(i) + '.h5')
    spio.savemat('./predictions/predictions_ep' + str(i) + '.mat',
                 mdict={'val_y': val_y, 'val_y_': val_y_, 'files_val': files_val,
                        'acc_tr': acc_tr, 'loss_tr': loss_tr, 'acc_val': acc_val})

model.save('./model.h5')
spio.savemat('./predictions.mat',
             mdict={'val_y': val_y, 'val_y_': val_y_, 'files_val': files_val,
                    'acc_tr': acc_tr, 'loss_tr': loss_tr, 'acc_val': acc_val})

print("Training completed successfully!")

