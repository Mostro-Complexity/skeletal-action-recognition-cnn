"""
这个程序直接将不同关节的三位坐标作为图像的RGB分量做2维卷积
acc稳定在0.7到0.8之间

"""
import cv2
import h5py
import keras
import numpy as np
import scipy.stats as stat
import tensorflow as tf
from keras.initializers import TruncatedNormal, Zeros
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import SGD
from keras.regularizers import l1, l2
from keras.utils import plot_model, to_categorical
from skimage import transform
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from util.dataset_processing import rand_perm_dataset, split_dataset
from util.preprocessing import (flip_img_horizontal, random_select_patch,
                                standardize_img)
from util.training import (te_generator, tr_x_generator, trainset_generator,
                           val_x_generator)

RESIZE_ISIZE = (60, 60, 3)
INPUT_ISIZE = (52, 52, 3)


def build(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape,
                     strides=1, padding='same'))

    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

    model.add(Conv2D(32, (3, 3), activation='relu', strides=1, padding='same'))

    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Conv2D(64, (3, 3), activation='relu', strides=1, padding='same'))

    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

    model.add(Conv2D(64, (3, 3), activation='relu', strides=1, padding='same'))

    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(256, activation='relu', kernel_regularizer=l2(1.e-2)))

    # model.add(Dropout(0.5))

    model.add(Dense(20, activation='softmax', kernel_regularizer=l2(1.e-2)))
    # decay=1e-6, lr=0.00002
    sgd = SGD(lr=0.01, decay=1e-4, momentum=0.9, nesterov=True)
    # 论文中写的loss和这里不一样
    # 试试categorical_crossentropy,mean_squared_error
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd, metrics=['accuracy'])
    return model


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


def training_pipline(features, subject_labels, action_labels, tr_subjects, te_subjects):
    action_labels = to_categorical(action_labels - 1, 20)
    i = 1

    model = build(INPUT_ISIZE)

    model.summary()

    tr_features, tr_labels, te_features, te_labels = split_dataset(
        features, action_labels, subject_labels, tr_subjects[i, :], te_subjects[i, :])

    tr_features, tr_labels = rand_perm_dataset(tr_features, tr_labels)

    n_actions = np.unique(action_labels, axis=0).shape[0]
    n_tr_samples = tr_labels.shape[0]
    n_te_samples = te_labels.shape[0]

    '''-----------------------个人复现版------------------------'''
    n_group, n_samples_in_group, epochs = 7, 5, 100
    batch_size = n_group*n_samples_in_group  # mush be times of n_samples_in_group
    n_orig_samples_per_step = 7

    tr_gen = trainset_generator(tr_features, tr_labels,
                                RESIZE_ISIZE, INPUT_ISIZE,
                                n_actions, n_tr_samples,
                                n_orig_samples_per_step)

    val_gen = val_x_generator(te_features, te_labels,
                              RESIZE_ISIZE, INPUT_ISIZE,
                              n_actions, n_te_samples,
                              batch_size, n_group)

    model.fit_generator(tr_gen, steps_per_epoch=n_tr_samples,
                        epochs=epochs, validation_data=val_gen,
                        validation_steps=300)

    te_gen = te_generator(te_features, te_labels,
                          RESIZE_ISIZE, INPUT_ISIZE,
                          n_actions, n_te_samples,
                          batch_size, n_group)

    pred_list = model.predict_generator(generator=te_gen,
                                        steps=n_te_samples)

    pred_list = np.array(pred_list)
    pred_list = np.reshape(pred_list, newshape=(
        n_te_samples, 2*(batch_size//n_group), -1))

    preds = np.argmax(pred_list, axis=2) + 1  # (number, 2*n_group)

    te_labels = np.argmax(te_labels, axis=1)+1
    print(te_labels)

    # (amount_testset, 1) numpy array
    pred_result = np.squeeze(stat.mode(preds, axis=1)[0])

    print(pred_result)
    print(np.sum(te_labels == pred_result)/te_labels.size)

    return model, te_features, te_labels


def additional_tr_pipline(model, te_features, te_labels):
    n_te_samples = te_labels.shape[0]
    te_labels = to_categorical(te_labels - 1, 20)
    epochs, n_orig_samples_per_step = 1, 7
    n_actions = np.unique(action_labels, axis=0).shape[0]

    tr_gen = trainset_generator(te_features, te_labels,
                                RESIZE_ISIZE, INPUT_ISIZE,
                                n_actions, n_te_samples,
                                n_orig_samples_per_step)

    model.fit_generator(tr_gen, steps_per_epoch=n_te_samples,
                        epochs=epochs)

    return model


def classification_pipline(model, samples):
    n_actions = 20
    if np.ndim(samples) == 2:
        samples = samples[np.newaxis, :, :]
        labels = to_categorical(0, n_actions)
        labels = labels[np.newaxis, :]
        n_samples = 1
    else:
        n_samples = samples.shape[0]
        labels = to_categorical(np.zeros(n_samples), n_actions)

    te_gen = te_generator(samples, labels,
                          RESIZE_ISIZE, INPUT_ISIZE,
                          n_actions, n_samples)
    pred_list = model.predict_generator(generator=te_gen,
                                        steps=n_samples)

    if np.ndim(pred_list)==2:
        pred_list=pred_list[np.newaxis,:,:]
    preds = np.argmax(pred_list, axis=2) + 1 
    pred_result = np.squeeze(stat.mode(preds, axis=1)[0])

    return pred_result

if __name__ == "__main__":
    # 读取特征
    f = h5py.File(
        'data/MSRAction3D/features.mat', 'r')

    features = np.array([f[element]
                        for element in np.squeeze(f['features'][:])])

    # 读取标签
    f = h5py.File(
        'data/MSRAction3D/labels.mat', 'r')

    subject_labels = f['subject_labels'][:, 0]
    action_labels = f['action_labels'][:, 0]

    # 读取数据集划分方案
    f = h5py.File(
        'data/MSRAction3D/tr_te_splits.mat', 'r')

    tr_subjects = f['tr_subjects'][:].T
    te_subjects = f['te_subjects'][:].T
    n_tr_te_splits = tr_subjects.shape[0]

    model, te_features, te_labels = training_pipline(features, subject_labels,
                                                     action_labels, tr_subjects, te_subjects)

    model = additional_tr_pipline(model, te_features, te_labels)

    print(classification_pipline(model, te_features[6]))

    model.save('model/rgb_mapping.h5')
