import numpy as np
from keras.utils import to_categorical

from preprocessing import (corner_select_patch,
                           flip_img_horizontal,
                           map_img,
                           random_select_patch)


def tr_x_generator(tr_features, tr_labels,
                   resize_isize, input_isize,
                   n_actions, n_tr_samples,
                   batch_size=35, n_group=5):
    anchor = 0
    while True:
        ret_x = np.zeros(
            shape=(0, input_isize[0], input_isize[1], 3), dtype=np.float32)
        ret_y = np.zeros(shape=(0, n_actions), dtype=np.float32)
        if anchor > n_tr_samples-batch_size:
            anchor = 0
            continue
        # here batch_size has to be the times of n_group
        group_size = batch_size // n_group
        for n_sp_in_grp in range(group_size):
            current = anchor + n_sp_in_grp
            # rgb with shape (60, 60, 3)
            rgb_img = map_img(tr_features[current],resize_isize)
            rgb_img = flip_img_horizontal(rgb_img, flip_prob=0.6)
            # randomly flip the image horizontally with the probability of filp_prob
            patches = random_select_patch(
                rgb_img, resize_isize, input_isize)  # random select patches
            label = to_categorical(tr_labels[current]-1, n_actions)
            label = label[np.newaxis, :]
            labels = np.tile(label, reps=[n_group, 1])

            ret_x = np.concatenate((ret_x, patches), axis=0)
            ret_y = np.concatenate((ret_y, labels), axis=0)
        anchor += group_size
        yield (ret_x, ret_y)


def val_x_generator(te_features, te_labels,
                    resize_isize, input_isize, n_actions,
                    n_te_samples, n_group=5):
    '''
    here we trick and use test set as validation set since the train set is too small in cross-view exp
    :return:
    '''
    validate_max_size = n_te_samples
    while True:
        ret_x = np.zeros(shape=(2*n_group, input_isize[0],
                                input_isize[1], 3), dtype=np.float32)
        ret_y = np.zeros(
            shape=(2*n_group, n_actions), dtype=np.float32)

        # random_select = random.sample(range(validate_max_size), 1)[0]
        random_select = np.random.randint(validate_max_size)

        # rgb with shape (60, 60, 3)
        rgb_img = map_img(te_features[random_select], resize_isize)

        clips = corner_select_patch(rgb_img, resize_isize, input_isize)
        flip_clips = corner_select_patch(
            flip_img_horizontal(rgb_img, flip_prob=1.00),
            resize_isize, input_isize)

        ret_x[0:n_group] = clips
        ret_x[n_group:] = flip_clips
        label = to_categorical(
            te_labels[random_select]-1, n_actions)
        label = label[np.newaxis, :]
        labels = np.tile(label, reps=[2*n_group, 1])
        ret_y[:] = labels

        yield (ret_x, ret_y)
