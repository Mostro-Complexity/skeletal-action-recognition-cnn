import os
import sys
import numpy as np
import json
import hdf5storage


def parse_single_file(filename):
    f = open(file, 'r')
    lines = f.read().splitlines()
    elements = np.float64([each_line.split(' ') for each_line in lines])

    n_frames = elements.size//(4*20)
    video = np.array([elements[20*f:20*(f+1)] for f in range(n_frames)])
    return video[:, :, :3]


def correct_coord(mat):
    temp = mat[:, :, 1].copy()
    mat[:, :, 1] = mat[:, :, 2]
    mat[:, :, 2] = temp
    return mat


def center_hip_loc(mat, body_model):
    hci = body_model['hip_center_index']-1
    hip_loc = mat[:, hci, :]
    hip_loc = np.repeat(hip_loc[:, np.newaxis, :], 20, axis=1)
    mat -= hip_loc
    return mat


def unify_bone_len(mat, body_model):
    '''每根骨头缩放到统一的长度'''
    n_frames, n_actions = mat.shape[:2]
    bone_lengths = body_model['bone_lengths']

    # b1_joints = body_model['primary_pairs'][:, :2]
    primary_pairs = np.array(body_model['primary_pairs'])
    j_pairs = primary_pairs[:, 2:] - 1
    for t in range(n_frames):
        for k in range(len(bone_lengths)):
            unit = mat[t, j_pairs[k, 1]]-mat[t, j_pairs[k, 0]]
            unit /= np.linalg.norm(unit)
            mat[t, j_pairs[k, 1]] = bone_lengths[k] * \
                unit+mat[t, j_pairs[k, 0]]

    return mat


def valid_normlise(vec, epsilon):
    norm = np.linalg.norm(vec)
    vec = vec.astype(np.float32)
    if norm <= epsilon:
        vec = np.zeros_like(vec)
    else:
        vec /= norm
    return vec


def vrrotvec(a, b):
    epsilon = 1e-12
    a = valid_normlise(a, epsilon)
    b = valid_normlise(b, epsilon)
    ax = valid_normlise(np.cross(a, b), epsilon)
    angle = np.arccos(min([np.dot(a, b), 1]))

    if not np.any(ax):
        absa = np.abs(a)
        mind = np.argmin(absa)
        c = np.zeros(3)
        c[mind] = 1
        ax = valid_normlise(np.cross(a, c), epsilon)

    return np.append(ax, angle)


def vrrotvec2mat(r):
    epsilon, ax, angle = 1e-12, r[:3], r[-1]
    s, c = np.sin(angle), np.cos(angle)
    t = 1-c

    n = valid_normlise(ax, epsilon)
    x, y, z = n[0], n[1], n[2]
    return np.array([
        [t*x*x + c,    t*x*y - s*z,  t*x*z + s*y],
        [t*x*y + s*z,  t*y*y + c,    t*y*z - s*x],
        [t*x*z - s*y,  t*y*z + s*x,  t*z*z + c]
    ])


def compute_relative_angles(mat, body_model):
    n_frames = mat.shape[0]

    primary_pairs = np.array(body_model['primary_pairs']) - 1
    bone1_joints = primary_pairs[:, :2]
    bone2_joints = primary_pairs[:, 2:]

    assert mat.shape[2] == 3, 'skeletons are expected to be 3 dimensional'
    n_angles = bone1_joints.shape[0]

    rot_mats = np.empty((n_frames, n_angles), dtype=object)
    for j in range(n_frames):
        for i in range(n_angles):
            bone1_global = None
            if bone1_joints[i, 1] != -1:
                bone1_global = mat[j, bone1_joints[i, 1]] - \
                    mat[j, bone1_joints[i, 0]]
            else:
                bone1_global = np.array([1, 0, 0])-mat[j, bone1_joints[i, 0]]

            bone2_global = mat[j, bone2_joints[i, 1]] - \
                mat[j, bone2_joints[i, 0]]

            if np.all(bone1_global == np.zeros(3)) or np.all(bone2_global == np.zeros(3)):
                rot_mats[j, i] = None
            else:
                R = vrrotvec2mat(vrrotvec(bone1_global, np.array([1, 0, 0])))
                rot_mats[j, i] = vrrotvec2mat(
                    vrrotvec(np.matmul(R, bone1_global), np.matmul(R, bone2_global)))

    return rot_mats


def reconstruct_locations_per_frame(R, bone1_joints, bone2_joints, bone_lengths):
    n_angles = R.shape[0]
    n_joints = n_angles + 1
    joint_locations = np.zeros((n_joints, 3))

    joint_locations[bone1_joints[0, 0], :] = [0, 0, 0]
    joint_locations[bone2_joints[0, 1], :] = bone_lengths[0] * \
        np.matmul(R[0], np.array([1, 0, 0]))

    for k in range(1, n_angles):
        bone1_global = joint_locations[bone1_joints[k, 1]] - \
            joint_locations[bone1_joints[k, 0]]
        rmat = vrrotvec2mat(vrrotvec(np.array([1, 0, 0]), bone1_global))
        bone2_global = np.matmul(np.matmul(rmat, R[k]), np.array([1, 0, 0]))
        joint_locations[bone2_joints[k, 1], :] = bone_lengths[k] * \
            bone2_global + joint_locations[bone2_joints[k, 0], :]

    return joint_locations


def reconstruct_locations(R, body_model):
    bone_lengths = body_model['bone_lengths']
    primary_pairs = np.array(body_model['primary_pairs']) - 1
    bone1_joints = primary_pairs[:, :2]
    bone2_joints = primary_pairs[:, 2:]

    n_frames, n_angles = R.shape[:2]
    normalized_locations = np.empty(
        (n_frames, n_angles + 1, 3), dtype=np.float32)
    frame_validity = np.ones(n_frames)

    for j in range(n_frames):
        reconstruct = True
        for i in range(body_model['n_primary_angles']):
            if R[j, i] is None:
                reconstruct = False
        if reconstruct:
            normalized_locations[j] = reconstruct_locations_per_frame(
                R[j], bone1_joints, bone2_joints, bone_lengths)
        else:
            frame_validity[j] = 0

    frame_validity = frame_validity == 1
    normalized_locations = normalized_locations[frame_validity]
    # time_stamps = np.arange(n_frames[frame_validity])
    # time_stamps += 1
    return normalized_locations


if __name__ == "__main__":
    n_actions = 20
    n_subjects = 10
    n_instances = 3

    dataset_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(dataset_dir)

    uniform_skeletal_data = np.empty(
        (n_actions, n_subjects, n_instances), dtype=object)
    original_skeletal_data = uniform_skeletal_data.copy()

    skeletal_data_validity = np.zeros(
        (n_actions, n_subjects, n_instances), dtype=np.int32)

    raw_dataset_dir = 'MSRAction3DSkeletonReal3D'

    body_model = json.load(open('body_model.json', 'r'))

    for a in range(n_actions):
        for s in range(n_subjects):
            for e in range(n_instances):
                file = raw_dataset_dir + \
                    '/a%02i_s%02i_e%02i_skeleton3D.txt' % (a+1, s+1, e+1)

                if os.path.exists(file):
                    skeletal_data_validity[a, s, e] = 1

                    joint_locations = parse_single_file(file)
                    joint_locations = correct_coord(joint_locations)
                    joint_locations = center_hip_loc(
                        joint_locations, body_model)

                    original_skeletal_data[a, s, e] = joint_locations

                    uniform_skeletal_data[a, s, e] = unify_bone_len(
                        joint_locations, body_model)

    hdf5storage.savemat('uniform_skeletal_data', {
                        u'skeletal_data': uniform_skeletal_data,
                        u'skeletal_data_validity': skeletal_data_validity
                        })

    hdf5storage.savemat('original_skeletal_data', {
                        u'skeletal_data': original_skeletal_data,
                        u'skeletal_data_validity': skeletal_data_validity
                        })
