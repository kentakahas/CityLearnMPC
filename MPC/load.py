# Import database
# Modified version of Scikit-learn load_data() function
# Allows data and target to be callable like db.target
# first row is the metadata = Number of samples, data, and targets

import csv
import numpy as np


class DataFormat:
    def __init__(self, data, target, header, n_samples, n_features, n_targets):
        self.data = data
        self.target = target
        self.header = header
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_targets = n_targets


def load_data(f):
    data_file = csv.reader(f)

    meta = next(data_file)
    n_samples = int(meta[0])
    n_features = int(meta[1])
    n_targets = int(meta[2])

    fields = next(data_file)
    header = np.array(fields)

    data = np.empty((n_samples, n_features))
    target = np.empty((n_samples, n_targets))

    for i, row in enumerate(data_file):
        data[i] = row[:n_features]
        target[i] = row[-n_targets:]

    return DataFormat(data, target, header, n_samples, n_features, n_targets)


def load_csv(filename):
    with open(filename, 'r') as f:
        return load_data(f)


def load_io(io):
    with io as f:
        return load_data(f)
