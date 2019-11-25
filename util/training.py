import numpy as np
from tensorflow.keras.utils import to_categorical

from skimage import transform
from .preprocessing import (corner_select_patch,
                            flip_img_horizontal,
                            map_skeletal_img,
                            random_select_patch,
                            compute_relative_loaction,
                            compute_cross_angle,
                            compute_dot_angle,
                            standardize_img)


def __init_samples_ret__(input_isize, n_actions):
    '''clear samples list for return
    '''
    ret_x = np.empty(
        shape=(0, input_isize[0], input_isize[1], 3),
        dtype=np.float32)
    ret_y = np.empty(shape=(0, n_actions), dtype=np.float32)
    return ret_x, ret_y


def __init_features_ret__(input_isize):
    '''clear features list for return
    '''
    ret_x = np.empty(
        shape=(0, input_isize[0], input_isize[1], input_isize[2]),
        dtype=np.float32)
    return ret_x


def __init_features_list__(input_isize):
    assert type(input_isize) == list
    return [__init_features_ret__(isize) for isize in input_isize]


def tr_batch_generator(tr_features, tr_labels,
                       resize_isize, input_isize,
                       n_actions, n_tr_samples,
                       n_orig_samples_per_step, n_patches=5):
    '''feature generator designed by me.
    :n_orig_samples_per_step: how many samples in raw data is used for each iteration
    :n_patches: how many patches is genarated by each original sample
    '''
    while True:
        ret_x, ret_y = __init_samples_ret__(input_isize, n_actions)
        for s in range(n_tr_samples):
            if s != 0 and s % n_orig_samples_per_step == 0:
                # if n_orig_samples_per_step is reached
                yield ret_x, ret_y
                ret_x, ret_y = __init_samples_ret__(input_isize, n_actions)

            rgb_img = map_skeletal_img(tr_features[s], resize_isize)
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
    '''reference program code, some samples cannot be used. [deprecated]
    '''

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
            rgb_img = map_skeletal_img(tr_features[current], resize_isize)
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


def val_batch_generator(te_features, te_labels,
                        resize_isize, input_isize,
                        n_actions, n_te_samples,
                        n_orig_samples_per_step, n_patches=5):
    '''
    here we trick and use test set as validation set since the train set is too small in cross-view exp
    :return:
    '''
    validate_max_size = n_te_samples
    batch_size = n_patches * n_orig_samples_per_step
    while True:
        ret_x = np.zeros(shape=(2*n_patches, input_isize[0],
                                input_isize[1], 3), dtype=np.float32)
        ret_y = np.zeros(
            shape=(2*n_patches, n_actions), dtype=np.float32)

        # random_select = random.sample(range(validate_max_size), 1)[0]
        random_select = np.random.randint(validate_max_size)

        # rgb with shape (60, 60, 3)
        rgb_img = map_skeletal_img(te_features[random_select], resize_isize)

        clips = corner_select_patch(rgb_img, input_isize)
        flip_clips = corner_select_patch(
            flip_img_horizontal(rgb_img, flip_prob=1.00),
            input_isize)

        ret_x[0:n_patches] = clips
        ret_x[n_patches:] = flip_clips
        label = te_labels[random_select]
        label = label[np.newaxis, :]
        labels = np.tile(label, reps=[2*n_patches, 1])
        ret_y = labels

        yield (ret_x, ret_y)


def te_batch_generator(te_features, te_labels,
                       resize_isize, input_isize,
                       n_actions, n_te_samples,
                       n_orig_samples_per_step, n_patches=5):
    '''
    select five patches from the four corner and center and their horizontal flip to evaluate. Using the voting result
    as the final result
    :return: (ret_x, ret_y) with the shape ret_x ~ (10, weight, height, 3), (10, num_actions)
    '''
    anchor = 0
    batch_size = n_orig_samples_per_step*n_patches
    while True:
        ret_x = np.zeros(shape=(2*n_patches, input_isize[0],
                                input_isize[1], 3), dtype=np.float32)
        ret_y = np.zeros(
            shape=(2*n_patches, n_actions), dtype=np.float32)
        if anchor > n_te_samples-1:
            print('Test traversal has been done !')
            break
        # rgb with shape (60, 60, 3)
        rgb_img = map_skeletal_img(te_features[anchor], resize_isize)

        clips = corner_select_patch(rgb_img, input_isize)
        flip_clips = corner_select_patch(
            flip_img_horizontal(rgb_img, flip_prob=1.00), input_isize)
        # flip_clips = clips

        ret_x[0:n_patches] = clips
        ret_x[n_patches:] = flip_clips
        label = te_labels[anchor]
        label = label[np.newaxis, :]
        labels = np.tile(label, reps=[2*n_patches, 1])
        ret_y[:] = labels
        anchor += 1
        # print(anchor)

        yield (ret_x, ret_y)


def empty_batch_migenerator(input_isize, n_actions,
                            batch_size=35):
    while True:
        ret_y = to_categorical(0, n_actions)
        ret_y = ret_y[np.newaxis, :]
        ret_y = np.tile(ret_y, reps=[batch_size, 1])
        ret_x = [None, None, None]

        ret_x[0] = np.empty((batch_size, input_isize[0][0], input_isize[0][1], input_isize[0][2]),
                            dtype=np.float32)
        ret_x[1] = np.empty((batch_size, input_isize[1][0], input_isize[1][1], input_isize[1][2]),
                            dtype=np.float32)
        ret_x[2] = np.empty((batch_size, input_isize[2][0], input_isize[2][1], input_isize[2][2]),
                            dtype=np.float32)
        yield ret_x, ret_y


def tr_batch_migenerator(tr_features, tr_labels,
                         resize_isize, input_isize,
                         n_actions, n_tr_samples,
                         n_orig_samples_per_step, n_patches=5):

    while True:
        _, ret_y = __init_samples_ret__(input_isize[0], n_actions)
        ret_x = __init_features_list__(input_isize)
        for s in range(n_tr_samples):
            if s != 0 and s % n_orig_samples_per_step == 0:
                # if n_orig_samples_per_step is reached
                yield ret_x, ret_y  # TODO: 输入还有很多
                _, ret_y = __init_samples_ret__(input_isize[0], n_actions)
                ret_x = __init_features_list__(input_isize)  # TODO: 输入形状

            abs_pos_map = map_skeletal_img(tr_features[s], resize_isize[0])
            abs_pos_map = flip_img_horizontal(abs_pos_map, flip_prob=0.6)
            # randomly flip the image horizontally with the probability of filp_prob
            patches = random_select_patch(
                abs_pos_map, input_isize[0], n_patches)  # random select patches
            label = tr_labels[np.newaxis, s]
            labels = np.tile(label, reps=[n_patches, 1])

            ret_x[0] = np.concatenate((ret_x[0], patches), axis=0)
            ret_y = np.concatenate((ret_y, labels), axis=0)

            lines = np.array([[20, 3], [3, 1], [3, 4], [3, 2], [1, 8],  # for select ll angle
                              [8, 10], [10, 12], [4, 7], [7, 5], [5, 14],
                              [14, 16], [16, 18], [7, 6], [6, 15], [15, 17],
                              [17, 19], [2, 9], [9, 11], [11, 13], [20, 12],
                              [12, 18], [18, 19], [19, 13], [13, 20], [12, 13],
                              [13, 18], [18, 20], [20, 19], [19, 12], [13, 9],
                              [12, 8], [20, 4], [18, 14], [19, 15]])
            lines = np.concatenate((lines, lines[:, ::-1]), axis=0)
            lines -= 1

            rel_loc = compute_relative_loaction(tr_features[s], lines)
            cross_ang = compute_cross_angle(tr_features[s], rel_loc, lines)
            dot_ang = compute_dot_angle(rel_loc, lines)  # TODO: 输入形状

            cross_ang = np.swapaxes(cross_ang, 0, 1)
            dot_ang = np.swapaxes(dot_ang, 0, 1)
            cross_ang = np.uint8(standardize_img(cross_ang))
            dot_ang = np.uint8(standardize_img(dot_ang))

            cross_ang = transform.resize(cross_ang, resize_isize[1])
            dot_ang = transform.resize(dot_ang, resize_isize[2])

            cross_ang = flip_img_horizontal(cross_ang, flip_prob=0.6)
            dot_ang = flip_img_horizontal(dot_ang, flip_prob=0.6)

            patches = random_select_patch(cross_ang, input_isize[1])
            ret_x[1] = np.concatenate((ret_x[1], patches), axis=0)
            patches = random_select_patch(dot_ang, input_isize[2])
            ret_x[2] = np.concatenate((ret_x[2], patches), axis=0)
            if s == n_tr_samples - 1:
                # last step
                yield ret_x, ret_y


def val_batch_migenerator(te_features, te_labels,
                          resize_isize, input_isize,
                          n_actions, n_te_samples,
                          n_orig_samples_per_step, n_patches=5):

    validate_max_size = n_te_samples
    batch_size = n_patches * n_orig_samples_per_step

    ret_x = [None, None, None]
    while True:
        ret_x[0] = np.zeros(shape=tuple([2*n_patches]+list(input_isize[0])),
                            dtype=np.float32)
        ret_y = np.zeros(
            shape=(2*n_patches, n_actions), dtype=np.float32)

        # random_select = random.sample(range(validate_max_size), 1)[0]
        random_select = np.random.randint(validate_max_size)

        # rgb with shape (60, 60, 3)
        rgb_img = map_skeletal_img(te_features[random_select], resize_isize[0])

        clips = corner_select_patch(rgb_img, input_isize[0])
        flip_clips = corner_select_patch(
            flip_img_horizontal(rgb_img, flip_prob=1.00),
            input_isize[0])

        ret_x[0][0:n_patches] = clips
        ret_x[0][n_patches:] = flip_clips
        label = te_labels[np.newaxis, random_select]
        labels = np.tile(label, reps=[2*n_patches, 1])
        ret_y = labels

        ret_x[1] = np.zeros(shape=tuple([2*n_patches]+list(input_isize[1])),
                            dtype=np.float32)
        ret_x[2] = np.zeros(shape=tuple([2*n_patches]+list(input_isize[2])),
                            dtype=np.float32)
        lines = np.array([[20, 3], [3, 1], [3, 4], [3, 2], [1, 8],  # for select ll angle
                          [8, 10], [10, 12], [4, 7], [7, 5], [5, 14],
                          [14, 16], [16, 18], [7, 6], [6, 15], [15, 17],
                          [17, 19], [2, 9], [9, 11], [11, 13], [20, 12],
                          [12, 18], [18, 19], [19, 13], [13, 20], [12, 13],
                          [13, 18], [18, 20], [20, 19], [19, 12], [13, 9],
                          [12, 8], [20, 4], [18, 14], [19, 15]])
        lines = np.concatenate((lines, lines[:, ::-1]), axis=0)
        lines -= 1

        rel_loc = compute_relative_loaction(te_features[random_select], lines)
        cross_ang = compute_cross_angle(
            te_features[random_select], rel_loc, lines)
        dot_ang = compute_dot_angle(rel_loc, lines)  # TODO: 输入形状

        cross_ang = np.swapaxes(cross_ang, 0, 1)
        dot_ang = np.swapaxes(dot_ang, 0, 1)
        cross_ang = np.uint8(standardize_img(cross_ang))
        dot_ang = np.uint8(standardize_img(dot_ang))

        cross_ang = transform.resize(cross_ang, resize_isize[1])
        dot_ang = transform.resize(dot_ang, resize_isize[2])

        clips = corner_select_patch(cross_ang, input_isize[1])
        flip_clips = corner_select_patch(
            flip_img_horizontal(cross_ang, flip_prob=1.00), input_isize[1])
        ret_x[1][:n_patches] = clips
        ret_x[1][n_patches:] = flip_clips

        clips = corner_select_patch(dot_ang, input_isize[2])
        flip_clips = corner_select_patch(
            flip_img_horizontal(dot_ang, flip_prob=1.00), input_isize[2])
        ret_x[2][:n_patches] = clips
        ret_x[2][n_patches:] = flip_clips

        yield ret_x, ret_y


def te_batch_migenerator(te_features, te_labels,
                         resize_isize, input_isize,
                         n_actions, n_te_samples,
                         n_orig_samples_per_step, n_patches=5):
    ret_x = [None, None, None]
    batch_size = n_orig_samples_per_step*n_patches
    for anchor in range(n_te_samples):
        ret_x[0] = np.zeros(shape=tuple([2*n_patches]+list(input_isize[0])),
                            dtype=np.float32)
        ret_y = np.zeros(
            shape=(2*n_patches, n_actions), dtype=np.float32)

        rgb_img = map_skeletal_img(te_features[anchor], resize_isize[0])

        clips = corner_select_patch(rgb_img, input_isize[0])
        flip_clips = corner_select_patch(
            flip_img_horizontal(rgb_img, flip_prob=1.00), input_isize[0])

        ret_x[0][0:n_patches] = clips
        ret_x[0][n_patches:] = flip_clips
        label = te_labels[np.newaxis, anchor]
        ret_y = np.tile(label, reps=[2*n_patches, 1])

        ret_x[1] = np.zeros(shape=tuple([2*n_patches]+list(input_isize[1])),
                            dtype=np.float32)
        ret_x[2] = np.zeros(shape=tuple([2*n_patches]+list(input_isize[2])),
                            dtype=np.float32)
        lines = np.array([[20, 3], [3, 1], [3, 4], [3, 2], [1, 8],  # for select ll angle
                          [8, 10], [10, 12], [4, 7], [7, 5], [5, 14],
                          [14, 16], [16, 18], [7, 6], [6, 15], [15, 17],
                          [17, 19], [2, 9], [9, 11], [11, 13], [20, 12],
                          [12, 18], [18, 19], [19, 13], [13, 20], [12, 13],
                          [13, 18], [18, 20], [20, 19], [19, 12], [13, 9],
                          [12, 8], [20, 4], [18, 14], [19, 15]])
        lines = np.concatenate((lines, lines[:, ::-1]), axis=0)
        lines -= 1

        rel_loc = compute_relative_loaction(te_features[anchor], lines)
        cross_ang = compute_cross_angle(
            te_features[anchor], rel_loc, lines)
        dot_ang = compute_dot_angle(rel_loc, lines)  # TODO: 输入形状

        cross_ang = np.swapaxes(cross_ang, 0, 1)
        dot_ang = np.swapaxes(dot_ang, 0, 1)
        cross_ang = np.uint8(standardize_img(cross_ang))
        dot_ang = np.uint8(standardize_img(dot_ang))

        cross_ang = transform.resize(cross_ang, resize_isize[1])
        dot_ang = transform.resize(dot_ang, resize_isize[2])

        clips = corner_select_patch(cross_ang, input_isize[1])
        flip_clips = corner_select_patch(
            flip_img_horizontal(cross_ang, flip_prob=1.00), input_isize[1])
        ret_x[1][:n_patches] = clips
        ret_x[1][n_patches:] = flip_clips

        clips = corner_select_patch(dot_ang, input_isize[2])
        flip_clips = corner_select_patch(
            flip_img_horizontal(dot_ang, flip_prob=1.00), input_isize[2])
        ret_x[2][:n_patches] = clips
        ret_x[2][n_patches:] = flip_clips

        yield ret_x, ret_y
