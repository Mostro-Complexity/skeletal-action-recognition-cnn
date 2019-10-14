"""
这个程序直接将不同关节的三位坐标作为图像的RGB分量做2维卷积
acc=0.60

"""
import cv2
import h5py
import keras
import numpy as np
import tensorflow as tf
from keras.initializers import TruncatedNormal, Zeros
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import plot_model, to_categorical
from numpy.random import shuffle
from skimage import transform
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from util.preprocessing import (flip_img_horizontal, random_select_patch,
                                standardize_img)

RESIZE_ISIZE = (60, 60)
INPUT_ISIZE = (52, 52)


def build(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape,
                     strides=1))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=1))

    model.add(Conv2D(64, (3, 3), activation='relu', strides=1))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=1))

    model.add(Dropout(0.25))

    model.add(Conv2D(96, (3, 3), activation='relu', strides=1))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=1))

    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(256, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(20, activation='softmax'))
    # decay=1e-6, lr=0.00002
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    # 论文中写的loss和这里不一样
    # 试试categorical_crossentropy,mean_squared_error
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd, metrics=['accuracy'])
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
    """
    """
    n_samples, n_frames, n_feat = features.shape

    features = np.reshape(features, (-1, n_frames, n_feat//3, 3))
    features = np.swapaxes(features, 1, 2)
    # 插入hip_center
    features = np.insert(features, obj=6, values=np.zeros(
        (n_samples, n_frames, 3)), axis=1)
    n_feat += 3

    # 归一化
    for i in range(n_samples):
        features[i] = standardize_img(features[i])
    features = np.uint8(features)

    # 调整空间结构
    part_config = np.array([12, 10, 8, 1, 1, 3, 2, 2, 9, 11, 13, 20, 3, 4, 7, 7,
                            5, 6, 5, 14, 16, 18, 6, 15, 17, 19])
    features = features[:, part_config-1]

    desired_features = np.empty((n_samples, isize[0], isize[0], 3), np.float32)
    for i in range(n_samples):  # resize加缩小数值
        desired_features[i] = transform.resize(features[i], isize)
        # TODO:换掉这个函数

    return desired_features


# 读取特征
f = h5py.File(
    'MSRAction3D_experiments/absolute_joint_positions/features.mat', 'r')

features = np.array([f[element][:] for element in f['features'][0]])

# 动作的图像表示
features = map_features(features, RESIZE_ISIZE)

# 读取标签
f = h5py.File(
    'MSRAction3D_experiments/absolute_joint_positions/labels.mat', 'r')

subject_labels = f['subject_labels'][:][0]
action_labels = f['action_labels'][:][0]
print(features.shape)
print(subject_labels.shape)
print(action_labels.shape)

# features = mean_remove(features)

# features = features.astype(np.float32) / 255
action_labels = to_categorical(action_labels - 1, 20)

# 读取数据集划分方案
f = h5py.File(
    'data/MSRAction3D/tr_te_splits.mat', 'r')

tr_subjects = f['tr_subjects'][:].T
te_subjects = f['te_subjects'][:].T
n_tr_te_splits = tr_subjects.shape[0]

total_accuracy = np.empty(n_tr_te_splits, dtype=np.float32)
n_tr_te_splits = 1


for i in range(n_tr_te_splits):
    model = build(features.shape[1:4])

    model.summary()

    tr_features, tr_labels, te_features, te_labels = split_dataset(
        features, action_labels, subject_labels, tr_subjects[i, :], te_subjects[i, :])

    # tr_features, tr_labels = flip_img(tr_features, tr_labels)

    model.fit(tr_features, tr_labels,
              batch_size=18, epochs=100, validation_data=(te_features, te_labels))

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
