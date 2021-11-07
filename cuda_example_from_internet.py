import sys
import numba 
import numpy as np
from numba import cuda

print("Python version:", sys.version)
print("Numba version:", numba.__version__)
print("Numpy version:", np.__version__)

@cuda.jit
def cudakernel0(array):
    for i in range(array.size):
        array[i] += 0.5

array = np.array([0, 1], np.float32)
print('Initial array:', array)

print('Kernel launch: cudakernel0[1, 1](array)')
cudakernel0[1, 1](array)

print('Updated array:',array)


array = np.array([0, 1], np.float32)
print('Initial array:', array)

gridsize = 1024
blocksize = 1024
print("Grid size: {}, Block size: {}".format(gridsize, blocksize))

print("Total number of threads:", gridsize * blocksize)

print('Kernel launch: cudakernel0[gridsize, blocksize](array)')
cudakernel0[gridsize, blocksize](array)

print('Updated array:',array)