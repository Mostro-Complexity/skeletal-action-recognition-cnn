import numpy as np
from numpy.random import shuffle


def split_dataset(features, action_labels, subject_labels, tr_subjects, te_subjects):
    tr_subject_ind = np.isin(subject_labels, tr_subjects)
    te_subject_ind = np.isin(subject_labels, te_subjects)

    tr_labels = action_labels[tr_subject_ind]
    te_labels = action_labels[te_subject_ind]

    tr_features = features[tr_subject_ind]
    te_features = features[te_subject_ind]
    return tr_features, tr_labels, te_features, te_labels


def rand_perm_dataset(features, labels):
    n_samples = features.shape[0]
    index = np.arange(n_samples)
    shuffle(index)
    features = features[index]
    labels = labels[index]

    return features, labels
