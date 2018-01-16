import gzip
import numpy as np
import tables

X_train = np.array([
    [1, 2, 5, 8],
    [1, 4, 5, 8],
    [0, 1, 5, 8],
    [1, 2, 5, 9],
])
Y_train = np.array([1,2,3,4,6,7])

file_name = 'data/ggggg'

f = tables.openFile(file_name+'.hdf', 'w')
filters = tables.Filters(complib='blosc', complevel=5)

atom1= tables.Atom.from_dtype(X_train.dtype)
ds1 = f.createCArray(f.root, 'X_train', atom1, X_train.shape, filters=filters)
ds1[:] = X_train

atom2 = tables.Atom.from_dtype(X_train.dtype)
ds2 = f.createCArray(f.root, 'Y_train', atom2, Y_train.shape, filters=filters)
ds2[:] = Y_train
f.close()

# with gzip.open('data/file.txt.gz', 'wb') as f:
#     f.write(training_data.tostring())
#
#
# with gzip.open('data/file.txt.gz', 'r') as f:
#     data = f.read()

# print(np.fromstring(data))