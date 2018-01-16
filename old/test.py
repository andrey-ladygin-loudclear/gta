import h5py
import numpy as np
import zlib
import tables

#293
# f = h5py.File('data/training_12-31-2017-2.pkl.hdf', 'r')
f = h5py.File('/home/srivoknovskiy/Python/gta/data/training_12-31-2017-0.hdf', 'r')

print(list(f.keys()))
print(list(f.values()))


dataset = f.get('dataset_1')
print(f.items())

for i in f.get('X_train'):
    print(i)

dataset = f.get('X_train')
print(np.array(dataset).shape)
dataset = f.get('Y_train')
print(np.array(dataset).shape)
# dataset = f.get('Y_train')
# print(np.array(dataset).shape)

# print(x.dtype)
#
#
# f = tables.openFile('data/test.hdf', 'w')
# atom = tables.Atom.from_dtype(x.dtype)
# ds = f.createCArray(f.root, 'somename', atom, x.shape)
# ds[:] = x
# f.close()

# Store "all_data" in a chunked array...
# from: http://stackoverflow.com/questions/8843062/python-how-to-store-a-numpy-multidimensional-array-in-pytables

# f = tables.openFile('data/all_data.hdf', 'w')
# atom = tables.Atom.from_dtype(training_data.dtype)
# filters = tables.Filters(complib='blosc', complevel=5)
# ds = f.createCArray(f.root, 'all_data', atom, training_data.shape, filters=filters)
# # save w/o compressive filter
# #ds = f.createCArray(f.root, 'all_data', atom, all_data.shape)
# ds[:] = training_data
# f.close()



# training_data = [{
#     # 'x': np.ones((640, 480, 3)),
#     'x': np.ones((6, 4, 3)),
#     'y': [1,0,1,0]
# }]
#
# print(type(training_data[0]))
#
# training_data = np.array([np.ones((6, 4, 3)), np.ones((6, 4, 3))])
# print('training_data', training_data.shape)
#
# #np.save('data/test', np.array(training_data).tostring())
# np.save('data/test', np.array(training_data).tobytes(training_data.dtype))
# #np.dtype('b')
# data = np.load('data/test.npy')
# # data = zlib.decompress(data)
# data = np.frombuffer(data, dtype=training_data.dtype)
# print(data, training_data.dtype)
# print(data.shape)
# print(data[0])

# data/training_12-31-2017-1.pkl
# saving training data 257

# import bson
# import numpy as np
# import pickle
#
# training_data = [{
#     # 'x': np.ones((640, 480, 3)),
#     'x': np.ones((6, 4, 3)),
#     'y': [1,0,1,0]
# }]
#
# # with open('data/test.bson', 'a') as handle:
# #     data = bson.BSON.encode(training_data)
# #     handle.write(training_data)
#
#
#
# with open('data/test.bson', 'wb') as handle:
#     pickle.dump(training_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# with open('data/test.bson', 'rb') as handle:
#     b = pickle.load(handle)
#
# print(b)








# try:
#     from msvcrt import kbhit
# except ImportError:
#     import termios, fcntl, sys, os
#     def kbhit():
#         fd = sys.stdin.fileno()
#         oldterm = termios.tcgetattr(fd)
#         newattr = termios.tcgetattr(fd)
#         newattr[3] = newattr[3] & ~termios.ICANON & ~termios.ECHO
#         termios.tcsetattr(fd, termios.TCSANOW, newattr)
#         oldflags = fcntl.fcntl(fd, fcntl.F_GETFL)
#         fcntl.fcntl(fd, fcntl.F_SETFL, oldflags | os.O_NONBLOCK)
#         try:
#             while True:
#                 try:
#                     c = sys.stdin.read(1)
#                     return True
#                 except IOError:
#                     return False
#         finally:
#             termios.tcsetattr(fd, termios.TCSAFLUSH, oldterm)
#             fcntl.fcntl(fd, fcntl.F_SETFL, oldflags)
#
# kbhit()