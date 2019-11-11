import h5py
import os
import numpy as np
import json
from scipy.interpolate import interp1d
import hdf5storage


def read_skeletal_data(src_file, n_action,
                       n_subjects, n_instances):
    f = h5py.File(src_file, 'r')
    refs = np.array(f['skeletal_data'])

    skeletal_data_validity = np.array(
        f['skeletal_data_validity'])

    n_sequences = np.sum(skeletal_data_validity)

    features = np.empty(n_sequences, dtype=np.object)

    action_labels = np.empty(n_sequences, np.int32)

    subject_labels = np.empty(n_sequences, np.int32)

    instance_labels = np.empty(n_sequences, np.int32)

    count = 0
    for a in range(n_action):
        for s in range(n_subjects):
            for e in range(n_instances):
                if skeletal_data_validity[e, s, a] == 1:
                    features[count] = np.array(
                        f[refs[e, s, a]])

                    action_labels[count] = a+1
                    subject_labels[count] = s+1
                    instance_labels[count] = e+1

                    count += 1

    return features, action_labels,\
        subject_labels, instance_labels


def interpolation(sequence, body_model, n_desired_frames):
    """通过插值，统一视频长度
    注意matlab与numpy的reshape方式不同
    此处使用Fortran方式
    """
    for i in range(sequence.size):
        joint_locs = sequence[i]

        n_dim, n_joints, n_given_frames = joint_locs.shape

        joint_locs = np.delete(
            joint_locs, body_model['hip_center_index']-1, axis=1)
        n_features = (n_joints-1)*n_dim
        joint_locs = joint_locs.reshape((n_features, -1), order='F')

        valid_frame_indices = np.arange(n_given_frames)+1

        features = np.empty((n_features, n_desired_frames),
                            dtype=np.float32)

        for k in range(n_features):
            # 编辑插值函数格式
            f = interp1d(valid_frame_indices,
                         joint_locs[k, :], kind="cubic")
            # 通过相应的插值函数求得新的函数点
            features[k, :] = f(np.linspace(
                1, n_given_frames, n_desired_frames, endpoint=True))

        features = features.reshape((n_dim, n_joints-1, -1), order='F')
        features = np.insert(features, body_model['hip_center_index']-1, np.zeros(
            (n_dim, n_desired_frames), np.float32), axis=1)

        features = features.reshape((n_joints*n_dim, -1), order='F')
        sequence[i] = features

    return sequence


if __name__ == "__main__":
    dataset_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(dataset_dir)

    src_file = 'uniform_skeletal_data.mat'
    features, action_labels, subject_labels, instance_labels = read_skeletal_data(
        src_file, 20, 10, 3)

    body_model = json.load(open('body_model.json', 'r'))

    features = interpolation(features, body_model, 76)

    hdf5storage.savemat('features', {u'features': features})
    hdf5storage.savemat('labels', {
        u'action_labels': action_labels,
        u'subject_labels': subject_labels,
        u'instance_labels': instance_labels
    })
