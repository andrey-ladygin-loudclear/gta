import h5py
import numpy as np
import zlib

import sys
import tables

arr = np.random.rand(1000,600,800,3)

print(arr.dtype)
raise EOFError

#print(arr)

bytes = arr.reshape(-1).tobytes(arr.dtype)

#print(bytes)

bytes = zlib.compress(bytes)

bytes_length = sys.getsizeof(bytes)
print(bytes_length)
bytes = zlib.decompress(bytes)

new_arr = np.frombuffer(bytes, dtype=arr.dtype)
new_arr = new_arr.reshape((1000,600,800,3))
#print(new_arr)
print(new_arr == arr)

print(bytes_length)
