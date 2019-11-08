import numpy as np
from keras.utils import to_categorical

from .preprocessing import (corner_select_patch,
                            flip_img_horizontal,
                            map_img,
                            random_select_patch)


def __init_samples_ret__(input_isize, n_actions):
    '''clear samples list for return
    '''
    ret_x = np.empty(
        shape=(0, input_isize[0], input_isize[1], 3),
        dtype=np.float32)
    ret_y = np.empty(shape=(0, n_actions), dtype=np.float32)
    return ret_x, ret_y


def trainset_generator(tr_features, tr_labels,
                       resize_isize, input_isize,
                       n_actions, n_tr_samples,
                       n_orig_samples_per_step, n_patches=5):

    while True:
        ret_x, ret_y = __init_samples_ret__(input_isize, n_actions)
        for s in range(n_tr_samples):
            if s != 0 and s % n_orig_samples_per_step == 0:
                # if n_orig_samples_per_step is reached
                yield ret_x, ret_y
                ret_x, ret_y = __init_samples_ret__(input_isize, n_actions)

            rgb_img = map_img(tr_features[s], resize_isize)
            rgb_img = flip_img_horizontal(rgb_img, flip_prob=0.6)
            # randomly flip the image horizontally with the probability of filp_prob
            patches = random_select_patch(
                rgb_img, input_isize, n_patches)  # random select patches
            label = tr_labels[s]
            label = label[np.newaxis, :]
            labels = np.tile(label, reps=[n_patches, 1])

            ret_x = np.concatenate((ret_x, patches), axis=0)
            ret_y = np.concatenate((ret_y, labels), axis=0)

            if s == n_tr_samples - 1:
                # last step
                yield ret_x, ret_y


def tr_x_generator(tr_features, tr_labels,
                   resize_isize, input_isize,
                   n_actions, n_tr_samples,
                   batch_size=35, n_group=7):
    anchor = 0
    while True:
        ret_x = np.empty(
            shape=(0, input_isize[0], input_isize[1], 3), dtype=np.float32)
        ret_y = np.empty(shape=(0, n_actions), dtype=np.float32)
        if anchor > n_tr_samples-batch_size:
            anchor = 0
            continue
        # here batch_size has to be the times of n_group
        n_samples_in_group = batch_size // n_group
        for offset in range(n_group):
            current = anchor + offset
            # rgb with shape (60, 60, 3)
            rgb_img = map_img(tr_features[current], resize_isize)
            rgb_img = flip_img_horizontal(rgb_img, flip_prob=0.6)
            # randomly flip the image horizontally with the probability of filp_prob
            patches = random_select_patch(
                rgb_img, input_isize, n_samples_in_group)  # random select patches
            label = tr_labels[current]
            label = label[np.newaxis, :]
            labels = np.tile(label, reps=[n_samples_in_group, 1])

            ret_x = np.concatenate((ret_x, patches), axis=0)
            ret_y = np.concatenate((ret_y, labels), axis=0)
        anchor += n_group
        yield (ret_x, ret_y)


def val_x_generator(te_features, te_labels,
                    resize_isize, input_isize, n_actions,
                    n_te_samples, batch_size=35, n_group=7):
    '''
    here we trick and use test set as validation set since the train set is too small in cross-view exp
    :return:
    '''
    validate_max_size = n_te_samples
    n_samples_in_group = batch_size//n_group
    while True:
        ret_x = np.zeros(shape=(2*n_samples_in_group, input_isize[0],
                                input_isize[1], 3), dtype=np.float32)
        ret_y = np.zeros(
            shape=(2*n_samples_in_group, n_actions), dtype=np.float32)

        # random_select = random.sample(range(validate_max_size), 1)[0]
        random_select = np.random.randint(validate_max_size)

        # rgb with shape (60, 60, 3)
        rgb_img = map_img(te_features[random_select], resize_isize)

        clips = corner_select_patch(rgb_img, input_isize)
        flip_clips = corner_select_patch(
            flip_img_horizontal(rgb_img, flip_prob=1.00),
            input_isize)

        ret_x[0:n_samples_in_group] = clips
        ret_x[n_samples_in_group:] = flip_clips
        label = te_labels[random_select]
        label = label[np.newaxis, :]
        labels = np.tile(label, reps=[2*n_samples_in_group, 1])
        ret_y = labels

        yield (ret_x, ret_y)


def te_generator(te_features, te_labels,
                 resize_isize, input_isize,
                 n_actions, n_te_samples,
                 batch_size=35, n_group=7):
    '''
    select five patches from the four corner and center and their horizontal flip to evaluate. Using the voting result
    as the final result
    :return: (ret_x, ret_y) with the shape ret_x ~ (10, weight, height, 3), (10, num_actions)
    '''
    anchor = 0
    n_samples_in_group = batch_size//n_group
    while True:
        ret_x = np.zeros(shape=(2*n_samples_in_group, input_isize[0],
                                input_isize[1], 3), dtype=np.float32)
        ret_y = np.zeros(
            shape=(2*n_samples_in_group, n_actions), dtype=np.float32)
        if anchor > n_te_samples-1:
            print('Test traversal has been done !')
            break
        # rgb with shape (60, 60, 3)
        rgb_img = map_img(te_features[anchor], resize_isize)

        clips = corner_select_patch(rgb_img, input_isize)
        flip_clips = corner_select_patch(
            flip_img_horizontal(rgb_img, flip_prob=1.00), input_isize)
        # flip_clips = clips

        ret_x[0:n_samples_in_group] = clips
        ret_x[n_samples_in_group:] = flip_clips
        label = te_labels[anchor]
        label = label[np.newaxis, :]
        labels = np.tile(label, reps=[2*n_samples_in_group, 1])
        ret_y[:] = labels
        anchor += 1
        # print(anchor)

        yield (ret_x, ret_y)
