import os
import h5py
import numpy as np
import tables

for file in os.listdir('data'):
    print(os.path.join('data', file))
    # f = h5py.File(os.path.join('data', file), 'r')
    f = h5py.File(os.path.join('data', file), 'r')

    # print('Keys', list(f.keys()))
    # print('Values', list(f.values()))

    X_train_dataset = f.get('X_train')
    Y_train_dataset = f.get('Y_train')

    X_train = np.array(X_train_dataset)
    Y_train = np.array(Y_train_dataset)
    print(file, "X_train shape", X_train.shape, "Y_train shape", Y_train.shape)
    print("")


# dataset = f.get('Y_train')
# print(np.array(dataset).shape)