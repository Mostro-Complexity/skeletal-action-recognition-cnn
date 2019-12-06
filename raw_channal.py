"""
这个程序直接将不同关节的不同坐标（共57维）当成不同channal做一维卷积
accuracy平均达到70%

"""

import json

import h5py
import numpy as np
import tensorflow as tf
from numpy.random import shuffle
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (Activation, Conv1D, Dense, Flatten,
                                     MaxPool1D)
from tensorflow.keras.optimizers import SGD, RMSprop
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.initializers import TruncatedNormal


def build():
    model = Sequential()

    model.add(Conv1D(filters=60, kernel_size=3, input_shape=(76, 60),
                     strides=1, padding="same", activation='relu',
                     kernel_initializer=TruncatedNormal(0, .05), name='CONV1'))

    model.add(MaxPool1D(pool_size=2, strides=1))

    model.add(Conv1D(filters=120, kernel_size=3, kernel_initializer=TruncatedNormal(0, .05),
                     strides=1, padding="same", activation='relu', name='CONV2'))

    model.add(MaxPool1D(pool_size=2, strides=1))

    model.add(Flatten())

    model.add(Dense(1024, kernel_initializer=TruncatedNormal(0, .05),
                    activation='relu', name='FC1'))
    model.add(
        Dense(20, kernel_initializer=TruncatedNormal(0, .05), name='FC2'))

    model.add(Activation('sigmoid'))

    model.compile(loss='mean_squared_error',
                  optimizer=SGD(lr=0.05), metrics=['accuracy'])

    return model


def split_dataset(features, action_labels, subject_labels, tr_subjects, te_subjects):
    tr_subject_ind = np.isin(subject_labels, tr_subjects)
    te_subject_ind = np.isin(subject_labels, te_subjects)

    tr_labels = action_labels[tr_subject_ind]
    te_labels = action_labels[te_subject_ind]

    tr_features = features[tr_subject_ind]
    te_features = features[te_subject_ind]
    return tr_features, tr_labels, te_features, te_labels


# 读取特征
f = h5py.File(
    'data/MSRAction3D/features.mat', 'r')

features = np.array([f[element]
                     for element in np.squeeze(f['features'][:])])
features = features.reshape((features.shape[0], features.shape[1], -1))

# 读取标签
f = h5py.File(
    'data/MSRAction3D/labels.mat', 'r')

subject_labels = f['subject_labels'][:, 0]
action_labels = f['action_labels'][:, 0]
action_labels = to_categorical(action_labels - 1, 20)

# 读取数据集划分方案
f = json.load(open(
    'data/MSRAction3D/tr_te_splits.json', 'r'))

tr_subjects = np.array(f['tr_subjects'])
te_subjects = np.array(f['te_subjects'])
n_tr_te_splits = tr_subjects.shape[0]


total_accuracy = np.empty(n_tr_te_splits, dtype=np.float32)

for i in range(n_tr_te_splits):
    model = build()

    tr_features, tr_labels, te_features, te_labels = split_dataset(
        features, action_labels, subject_labels, tr_subjects[i, :], te_subjects[i, :])

    model.fit(tr_features, tr_labels,
              batch_size=1, epochs=60, validation_data=(te_features, te_labels))

    pr_labels = model.predict(te_features)
    pr_labels = np.argmax(pr_labels, axis=-1) + 1

    te_labels = np.argmax(te_labels, axis=-1) + 1

    total_accuracy[i] = np.sum(pr_labels == te_labels) / te_labels.size
    print('split %d is done, accuracy:%f' %
          (i + 1, total_accuracy[i]))

    # tf.reset_default_graph()
print('all splits is done, avg accuracy:%f' %
      (total_accuracy.mean()))
