from MPS_lib import SVD
import numpy as np
from numpy.linalg import norm

# define matrix
M = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])

# do SVD
U, S, Vh = SVD(M)

# test
M_reconstructed = U @ S @ Vh
assert norm(M_reconstructed-M) < 1e-10