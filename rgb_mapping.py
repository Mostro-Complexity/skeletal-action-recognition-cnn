"""
这个程序直接将不同关节的三位坐标作为图像的RGB分量做2维卷积
有问题

"""

import cv2
# from scipy.interpolate import spline
# TODO:插值自己实现
import h5py
import numpy as np
import tensorflow as tf
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from numpy.random import shuffle
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.optimizers import SGD, RMSprop
from keras.initializers import TruncatedNormal, Zeros

UNIFORM_ISIZE = (60, 60)


def build(input_shape):
    model = Sequential()

    model.add(Conv2D(filters=6, kernel_size=(3, 3), input_shape=input_shape,
                     strides=1, padding="same", activation='relu',
                     kernel_initializer=TruncatedNormal(0, .01), name='CONV1'))

    # model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding="same",
    #                  activation='relu', kernel_initializer=tf.truncated_normal_initializer(0, .05),
    #                  name='CONV2'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=1))

    model.add(Conv2D(filters=12, kernel_size=(3, 3), kernel_initializer=TruncatedNormal(0, .01),
                     strides=1, padding="same", activation='relu', name='CONV3'))

    # model.add(Conv2D(filters=128, kernel_size=(3, 3), kernel_initializer=tf.truncated_normal_initializer(0, .05),
    #                  strides=1, padding="same", activation='relu', name='CONV4'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=1))

    
    # model.add(Conv2D(filters=12, kernel_size=(3, 3), kernel_initializer=TruncatedNormal(0, .01),
    #                  strides=1, padding="same", activation='relu', name='CONV4'))

    # model.add(Conv2D(filters=128, kernel_size=(3, 3), kernel_initializer=tf.truncated_normal_initializer(0, .05),
    #                  strides=1, padding="same", activation='relu', name='CONV4'))

    # model.add(MaxPooling2D(pool_size=(2, 2), strides=1))

    model.add(Flatten())

    # model.add(Dense(4096, kernel_initializer=tf.truncated_normal_initializer(0, .05),
    #                 activation='relu', name='FC1'))
    # model.add(Dropout(0.5))

    model.add(Dense(1024, activation='relu', name='FC2'))
    # model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu', name='FC3'))


    model.add(Dense(20, name='FC4'))

    model.add(Activation('sigmoid'))

    model.compile(loss='mean_squared_error',
                  optimizer=SGD(lr=0.001), metrics=['accuracy'])

    return model


def split_dataset(features, action_labels, subject_labels, tr_subjects, te_subjects):
    tr_subject_ind = np.isin(subject_labels, tr_subjects)
    te_subject_ind = np.isin(subject_labels, te_subjects)

    tr_labels = action_labels[tr_subject_ind]
    te_labels = action_labels[te_subject_ind]

    tr_features = features[tr_subject_ind]
    te_features = features[te_subject_ind]
    return tr_features, tr_labels, te_features, te_labels


def map_features(features, isize):
    rindex = np.arange(0, features.shape[2], 3)
    rmaps = features[:, :, rindex]
    gmaps = features[:, :, rindex + 1]
    bmaps = features[:, :, rindex + 2]

    rmaps_rsp = np.empty((features.shape[0],
                          isize[0], isize[1]), dtype=np.float32)
    gmaps_rsp = np.empty((features.shape[0],
                          isize[0], isize[1]), dtype=np.float32)
    bmaps_rsp = np.empty((features.shape[0],
                          isize[0], isize[1]), dtype=np.float32)

    for i in range(rmaps.shape[0]):
        rmaps[i] = MinMaxScaler().fit_transform(rmaps[i])*255
        gmaps[i] = MinMaxScaler().fit_transform(gmaps[i])*255
        bmaps[i] = MinMaxScaler().fit_transform(bmaps[i])*255

        rmaps_rsp[i] = np.floor(cv2.resize(rmaps[i], isize))
        gmaps_rsp[i] = np.floor(cv2.resize(gmaps[i], isize))
        bmaps_rsp[i] = np.floor(cv2.resize(bmaps[i], isize))

    features = np.empty(
        (features.shape[0], isize[0], isize[1], 3), dtype=np.float32)
    features[:, :, :, 0] = bmaps_rsp
    features[:, :, :, 1] = gmaps_rsp
    features[:, :, :, 2] = rmaps_rsp

    features = features.astype(np.uint8)
    return features


# 读取特征
f = h5py.File(
    'MSRAction3D_experiments/absolute_joint_positions/features.mat', 'r')

features = np.array([f[element][:] for element in f['features'][0]])

# 动作的图像表示
features = map_features(features, UNIFORM_ISIZE)

# 读取标签
f = h5py.File(
    'MSRAction3D_experiments/absolute_joint_positions/labels.mat', 'r')

subject_labels = f['subject_labels'][:][0]
action_labels = f['action_labels'][:][0]
print(features.shape)
print(subject_labels.shape)
print(action_labels.shape)

# for i in range(features.shape[0]):
#     cv2.imshow('representation of an action', features[i]) 
#     print(action_labels[i])
#     cv2.waitKey()
features = features.astype(np.float32) * 10
action_labels = np_utils.to_categorical(action_labels - 1, 20)

# 读取数据集划分方案
f = h5py.File(
    'data/MSRAction3D/tr_te_splits.mat', 'r')

tr_subjects = f['tr_subjects'][:].T
te_subjects = f['te_subjects'][:].T
n_tr_te_splits = tr_subjects.shape[0]

total_accuracy = np.empty(n_tr_te_splits, dtype=np.float32)

for i in range(n_tr_te_splits):
    model = build(features.shape[1:4])

    tr_features, tr_labels, te_features, te_labels = split_dataset(
        features, action_labels, subject_labels, tr_subjects[i, :], te_subjects[i, :])

    model.fit(tr_features, tr_labels,
              batch_size=6, epochs=100, validation_data=(te_features, te_labels))

    # model.fit(tr_features[:2], tr_labels[:2], batch_size=1, epochs=10)

    pr_labels = model.predict(te_features)
    pr_labels = np.argmax(pr_labels, axis=-1) + 1

    te_labels = np.argmax(te_labels, axis=-1) + 1

    total_accuracy[i] = np.sum(pr_labels == te_labels) / te_labels.size
    print('split %d is done, accuracy:%f' %
          (i + 1, total_accuracy[i]))

    # tf.reset_default_graph()
print('all splits is done, avg accuracy:%f' %
      (total_accuracy.mean()))
