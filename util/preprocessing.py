import numpy as np
from skimage import transform


def map_skeletal_img(mat, resize_isize):
    '''map skeletal action to rgb image
    :param mat: (n_frames, feat_dim)
    :return: (60, 60, 3)
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
    rgb_image = transform.resize(
        rgb_image, (resize_isize[0], resize_isize[1], 3))
    return rgb_image


def standardize_img(mat):
    '''standardize each pixel of this image
    '''
    local_max = np.max(mat)
    local_min = np.min(mat)
    p = 255*(mat-local_min)/(local_max-local_min)
    return p


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
                           anchor_y:anchor_y+input_isize[1], :]
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


def compute_relative_loaction(mat, lines):
    n_frames, n_joints, n_dim = mat.shape
    n_lines = lines.shape[0]
    relative_locations = np.empty(
        (n_frames, n_lines, n_dim), dtype=np.float32)

    for t in range(n_frames):
        for i in range(n_lines):
            relative_locations[t, i] = mat[t, lines[i, 0]]-mat[t, lines[i, 1]]

    return relative_locations


def compute_cross_angle(abs_loc, rel_loc, lines):
    '''
    :abs_loc: joint locations
    :rel_loc: coordinates of lines(vectors)
    :lines: indices of start points and end points of lines
    '''
    n_frames, n_lines, n_dim = rel_loc.shape

    cross = np.empty(
        (n_frames, n_lines*(n_lines-1), n_dim), dtype=np.float32)

    for t in range(n_frames):
        counter = 0
        for lid_1 in range(n_lines):
            for lid_2 in range(n_lines):
                if lines[lid_1, 0] == lines[lid_2, 0] \
                        and lines[lid_1, 1] != lines[lid_2, 1]:
                    line_1 = rel_loc[t, lid_1]
                    line_2 = rel_loc[t, lid_2]
                    cross[t, counter] = np.cross(line_1, line_2) / \
                        np.linalg.norm(
                            abs_loc[t, lines[lid_1, 1]]-abs_loc[t, lines[lid_2, 1]])
                    counter += 1

    return cross[:counter]


def compute_dot_angle(rel_loc, lines):
    '''
    :abs_loc: joint locations
    :rel_loc: coordinates of lines(vectors)
    :lines: indices of start points and end points of lines
    '''
    n_frames, n_lines, n_dim = rel_loc.shape

    nloc = np.linalg.norm(rel_loc, axis=2)

    dot_ang = np.empty(
        (n_frames, n_lines*(n_lines-1)), dtype=np.float32)

    for t in range(n_frames):
        counter = 0
        for lid_1 in range(n_lines//2):  # use a half
            for lid_2 in range(n_lines//2):
                if np.any(lines[lid_1] != lines[lid_2]):
                    assert nloc[t, lid_1] > 1e-6 and nloc[t, lid_2] > 1e-6
                    uline_1 = rel_loc[t, lid_1]/nloc[t, lid_1]
                    uline_2 = rel_loc[t, lid_2]/nloc[t, lid_2]
                    d = np.clip(np.dot(uline_1, uline_2), -1, 1)
                    dot_ang[t, counter] = np.arccos(d)
                    counter += 1

    return dot_ang[:, :counter]
