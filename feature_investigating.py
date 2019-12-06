"""
acc:0.85
过去骨架数据到图像的映射采用训练前的预处理策略
此处将骨架到图像的映射转移到网络内部
"""
import json

import cv2
import h5py
import numpy as np
import scipy.stats as stat
import tensorflow as tf
from skimage import transform
from tensorflow import image
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import plot_model, to_categorical

from util.dataset_processing import rand_perm_dataset, split_dataset
from util.modeling import (build_conv_module, build_start_time_diff_module,
                           build_time_diff_module)
from util.preprocessing import (flip_img_horizontal, random_select_patch,
                                standardize_img)
from util.training import (te_batch_migenerator, tr_batch_migenerator,
                           val_batch_migenerator, empty_batch_migenerator)

RESIZE_ISIZE = [(60, 60, 3), (60, 60, 3), (60, 60, 1)]
INPUT_ISIZE = [(52, 52, 3), (52, 52, 3), (52, 52, 1)]


def build(input_shape):
    input_1 = Input(shape=input_shape[0], name='input_1')
    time_diff = build_time_diff_module(input_1, input_shape[0])
    # start_time_diff = build_start_time_diff_module(input_1, input_shape[0])

    raw_data_score = build_conv_module(input_1)
    time_diff_score = build_conv_module(time_diff)
    # start_time_diff_score = build_conv_module(start_time_diff)

    input_2 = Input(shape=input_shape[1], name='input_2')
    input_3 = Input(shape=input_shape[2], name='input_3')
    # ll_angle_score = build_conv_module(input_2)
    lp_angle_score = build_conv_module(input_2)

    output = lp_angle_score
    # output = raw_data_score*time_diff_score * \
    # start_time_diff_score*ll_angle_score*lp_angle_score

    model = Model([input_1,input_2,input_3], output)
    # decay=1e-6, lr=0.00002
    sgd = SGD(lr=0.01, decay=1e-4, momentum=0.9, nesterov=True)
    # 试试categorical_crossentropy,mean_squared_error
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd, metrics=['accuracy'])

    return model


def training_pipline(features, subject_labels, action_labels,
                     tr_subjects, te_subjects, body_model):
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

    epochs = 50
    # 7 orig samples is used and each orig sample gen 5 patches
    n_orig_samples_per_step, n_patches = 7, 5
    batch_size = n_orig_samples_per_step * n_patches

    tr_gen = tr_batch_migenerator(tr_features, tr_labels,
                                  RESIZE_ISIZE, INPUT_ISIZE, body_model,
                                  n_orig_samples_per_step)

    val_gen = val_batch_migenerator(te_features, te_labels,
                                    RESIZE_ISIZE, INPUT_ISIZE, body_model,
                                    n_orig_samples_per_step)

    model.fit_generator(tr_gen, steps_per_epoch=n_tr_samples,
                        epochs=epochs, validation_data=val_gen,
                        validation_steps=300)

    te_gen = te_batch_migenerator(te_features, te_labels,
                                  RESIZE_ISIZE, INPUT_ISIZE, body_model,
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

    tr_gen = tr_batch_migenerator(te_features, te_labels,
                                  RESIZE_ISIZE, INPUT_ISIZE, body_model,
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
    te_gen = te_batch_migenerator(samples, labels,
                                  RESIZE_ISIZE, INPUT_ISIZE, body_model,
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

    body_model = json.load(open(
        'data/MSRAction3D/body_model.json', 'r'))

    model, te_features, te_labels = training_pipline(features, subject_labels,
                                                     action_labels, tr_subjects, te_subjects, body_model)

    # model = additional_tr_pipline(model, te_features, te_labels)

    print(classification_pipline(model, te_features[6]))

    model.save('model/feature_investigating.h5')
