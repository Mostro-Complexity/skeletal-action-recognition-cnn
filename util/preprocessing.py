import numpy as np
from skimage import transform
import cv2


def map_skeletal_img(mat, resize_isize):
    '''map skeletal action to rgb image
    :param mat: (n_frames, feat_dim)
    :return: image with resize_isize
    '''
    # mat = np.reshape(mat, newshape=(-1, 20, 3))
    mat = np.swapaxes(mat, 0, 1)  # mat: (n_feat, n_frames, n_dim)

    n_frames = mat.shape[1]
    # part_config = [25, 12, 24, 11, 10, 9, 21, 21, 5, 6, 7, 8, 22, 23,
    #                21, 3, 4, 21, 2, 1, 17, 18, 19, 20, 21, 2, 1, 13, 14, 15, 16]
    part_config = np.array([12, 10, 8, 1, 1, 3, 2, 2, 9, 11, 13, 20, 3, 4, 7, 7,
                            5, 6, 5, 14, 16, 18, 6, 15, 17, 19])
    mat = mat[part_config - 1]
    # TODO:以下代码替换
    rgb_image = np.uint8(standardize_img(mat))
    rgb_image = transform.resize(rgb_image, resize_isize)
    return rgb_image


def standardize_img(mat):
    '''standardize each pixel of this image
    '''
    local_max = np.max(mat)
    local_min = np.min(mat)
    p = 255*(mat-local_min)/(local_max-local_min)
    return p


def map_standard_img(mat, resize_isize):
    '''map general image to rgb image
    :param mat: (n_frames, feat_dim)
    :return: image with resize_isize
    '''
    mat = np.swapaxes(mat, 0, 1)  # mat: (n_feat, n_frames, n_dim)

    n_frames = mat.shape[1]

    rgb_image = np.uint8(standardize_img(mat))
    rgb_image = cv2.resize(rgb_image, resize_isize[:2])
    rgb_image = standardize_img(rgb_image)/255

    # rgb_image = transform.resize(rgb_image, resize_isize)
    return rgb_image


def flip_img_horizontal(mat, flip_prob=0.6):
    '''flip the image horizontally randomly with probability of $flip_prob$
    '''
    rand = np.random.uniform(low=0, high=1.0)
    if rand > flip_prob:
        return mat
    else:
        # flip horizontally
        return np.fliplr(mat)


def random_select_patch(mat, input_isize, number=5):
    '''randomly select $number$ patch from image
    '''
    resize_isize = mat.shape[:2]

    patches = np.zeros(
        shape=(number, input_isize[0], input_isize[1], input_isize[2]), dtype=np.float32)
    height = resize_isize[1]-input_isize[1]
    weight = resize_isize[0]-input_isize[0]
    for each in range(number):
        anchor_x = np.random.randint(weight)
        anchor_y = np.random.randint(height)
        select_patch = mat[anchor_x:anchor_x+input_isize[0],
                           anchor_y:anchor_y+input_isize[1]]
        patches[each] = select_patch
    return patches


def corner_select_patch(mat, input_isize, number=5):
    '''select the patches from four corners and center
    '''
    resize_isize = mat.shape[:2]

    patches = np.zeros(
        shape=(number, input_isize[0], input_isize[1], input_isize[2]), dtype=np.float32)
    height = resize_isize[1]-input_isize[1]
    weight = resize_isize[0]-input_isize[0]
    anchors = [[0, 0], [weight, 0], [0, height], [
        weight, height], [weight//2, height//2]]
    for each in range(number):
        anchor_x, anchor_y = anchors[each][0], anchors[each][1]
        select_patch = mat[anchor_x:anchor_x+input_isize[0],
                           anchor_y:anchor_y+input_isize[1], :]
        patches[each] = select_patch
    return patches


def compute_cross_angle(mat, lp_angle_pairs):
    '''
    :mat: absolute joint locations
    :lp_angle_pairs: indices of start points and end points of angles
    '''
    lp_angle_pairs = np.array(lp_angle_pairs) - 1
    # lp_angle_pairs = lp_angle_pairs[::2]

    n_pairs = lp_angle_pairs.shape[0]
    n_frames, _, n_dim = mat.shape

    cross = np.empty((n_frames, n_pairs, n_dim), dtype=np.float32)

    for t in range(n_frames):
        rls_1 = mat[t, lp_angle_pairs[:, 1]]-mat[t, lp_angle_pairs[:, 0]]
        rls_2 = mat[t, lp_angle_pairs[:, 2]]-mat[t, lp_angle_pairs[:, 0]]
        rls_0 = mat[t, lp_angle_pairs[:, 2]]-mat[t, lp_angle_pairs[:, 1]]
        cross[t] = np.cross(rls_1, rls_2, axis=1)
        norm = np.linalg.norm(rls_0, axis=1)
        cross[t] /= norm[:, np.newaxis]

    return cross


def compute_dot_angle(mat, ll_angle_pairs):
    '''
    :mat: absolute joint locations
    :ll_angle_pairs: indices of start points and end points of angles
    '''
    ll_angle_pairs = np.array(ll_angle_pairs) - 1
    ll_angle_pairs = ll_angle_pairs[::2]
    n_pairs = ll_angle_pairs.shape[0]
    n_frames, _, n_dim = mat.shape

    dot = np.empty((n_frames, n_pairs), dtype=np.float32)

    for t in range(n_frames):
        rls_1 = mat[t, ll_angle_pairs[:, 1]]-mat[t, ll_angle_pairs[:, 0]]
        rls_1 /= np.linalg.norm(rls_1, axis=1)[:, np.newaxis]
        rls_2 = mat[t, ll_angle_pairs[:, 3]]-mat[t, ll_angle_pairs[:, 2]]
        rls_2 /= np.linalg.norm(rls_2, axis=1)[:, np.newaxis]

        dot[t] = np.sum(rls_1*rls_2, axis=1)
        dot[t] = np.clip(dot[t], -1, 1)
        dot[t] = np.arccos(dot[t])
    return dot
