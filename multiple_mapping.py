"""
这个程序直接将不同关节的三位坐标作为图像的RGB分量做2维卷积
将前后帧之差（time_diff）放入feature map中
acc稳定0.8

"""
import cv2
import h5py
import numpy as np
import scipy.stats as stat
import json
import tensorflow as tf
from tensorflow.keras.initializers import TruncatedNormal, Zeros
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, Input, concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l1, l2
from tensorflow import image
from tensorflow.keras.utils import plot_model, to_categorical
from skimage import transform
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from util.dataset_processing import rand_perm_dataset, split_dataset
from util.preprocessing import (flip_img_horizontal, random_select_patch,
                                standardize_img)
from util.training import (te_psbatch_generator, tr_x_generator, tr_psbatch_generator,
                           val_psbatch_generator)

RESIZE_ISIZE = (60, 60, 3)
INPUT_ISIZE = (52, 52, 3)


def build_feat_extraction_module(input_shape):
    input_layer = Input(shape=input_shape)

    time_diff = input_layer[:, :, 1:]-input_layer[:, :, :-1]
    time_diff = image.resize(
        time_diff, list(input_shape[:-1]),
        method=image.ResizeMethod.NEAREST_NEIGHBOR)

    shared_layer = concatenate([
        input_layer, time_diff], axis=-1)

    # start_time = input_layer[:, :, 0]
    # start_time = tf.tile(start_time[:, :, tf.newaxis],
    #                      [1, 1, input_shape[0]-1, 1])

    # start_time_diff = input_layer[:, :, 1:]-start_time
    # start_time_diff = image.resize(
    #     time_diff, list(input_shape[:-1]),
    #     method=image.ResizeMethod.NEAREST_NEIGHBOR)

    # shared_layer = concatenate([
    #     input_layer, time_diff, start_time_diff], axis=-1)

    features_module = Model(input_layer, shared_layer)

    features_module.summary()
    return features_module


def build_classify_module(input_shape):
    model = Sequential()

    model.add(build_feat_extraction_module(input_shape))

    model.add(Conv2D(32, (3, 3), activation='relu', strides=1, padding='same'))

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

    model.add(Dense(20, activation='softmax', kernel_regularizer=l2(1.e-2)))
    # decay=1e-6, lr=0.00002
    sgd = SGD(lr=0.01, decay=1e-4, momentum=0.9, nesterov=True)
    # 论文中写的loss和这里不一样
    # 试试categorical_crossentropy,mean_squared_error
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd, metrics=['accuracy'])
    return model


def training_pipline(features, subject_labels, action_labels, tr_subjects, te_subjects):
    action_labels = to_categorical(action_labels - 1, 20)
    i = 1

    model = build_classify_module(INPUT_ISIZE)

    model.summary()

    tr_features, tr_labels, te_features, te_labels = split_dataset(
        features, action_labels, subject_labels, tr_subjects[i, :], te_subjects[i, :])

    tr_features, tr_labels = rand_perm_dataset(tr_features, tr_labels)

    n_actions = np.unique(action_labels, axis=0).shape[0]
    n_tr_samples = tr_labels.shape[0]
    n_te_samples = te_labels.shape[0]

    '''-----------------------个人复现版------------------------'''
    epochs = 1
    # 7 orig samples is used and each orig sample gen 5 patches
    n_orig_samples_per_step, n_patches = 7, 5
    batch_size = n_orig_samples_per_step * n_patches

    tr_gen = tr_psbatch_generator(tr_features, tr_labels,
                                  RESIZE_ISIZE, INPUT_ISIZE,
                                  n_orig_samples_per_step)

    val_gen = val_psbatch_generator(te_features, te_labels,
                                    RESIZE_ISIZE, INPUT_ISIZE,
                                    n_orig_samples_per_step)

    model.fit_generator(tr_gen, steps_per_epoch=n_tr_samples,
                        epochs=epochs, validation_data=val_gen,
                        validation_steps=300)

    te_gen = te_psbatch_generator(te_features, te_labels,
                                  RESIZE_ISIZE, INPUT_ISIZE,
                                  n_orig_samples_per_step)

    pred_list = model.predict_generator(generator=te_gen,
                                        steps=n_te_samples)

    pred_list = np.array(pred_list)
    pred_list = np.reshape(pred_list, newshape=(
        n_te_samples, 2*(n_patches), -1))

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

    tr_gen = tr_psbatch_generator(te_features, te_labels,
                                  RESIZE_ISIZE, INPUT_ISIZE,
                                  n_orig_samples_per_step)

    model.fit_generator(tr_gen, steps_per_epoch=n_te_samples,
                        epochs=epochs)

    return model


def classification_pipline(model, samples):
    n_actions = 20
    if np.ndim(samples) == 3:
        samples = samples[np.newaxis, :, :]
        labels = to_categorical(0, n_actions)
        labels = labels[np.newaxis, :]
        n_samples = 1
    else:
        n_samples = samples.shape[0]
        labels = to_categorical(np.zeros(n_samples), n_actions)

    n_orig_samples_per_step = 7
    te_gen = te_psbatch_generator(samples, labels,
                                  RESIZE_ISIZE, INPUT_ISIZE,
                                  n_orig_samples_per_step)
    pred_list = model.predict_generator(generator=te_gen,
                                        steps=n_samples)

    if np.ndim(pred_list) == 2:
        pred_list = pred_list[np.newaxis, :, :]
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
    f = json.load(open(
        'data/MSRAction3D/tr_te_splits.json', 'r'))

    tr_subjects = np.array(f['tr_subjects'])
    te_subjects = np.array(f['te_subjects'])
    n_tr_te_splits = tr_subjects.shape[0]

    model, te_features, te_labels = training_pipline(features, subject_labels,
                                                     action_labels, tr_subjects, te_subjects)

    model = additional_tr_pipline(model, te_features, te_labels)

    print(classification_pipline(model, te_features[6]))

    model.save('model/multiple_mapping.h5')
