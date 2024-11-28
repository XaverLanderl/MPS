# imports
from gaussian_packet import *
from matplotlib import pyplot as plt

# set parameters
L = 51              # length of spin chain
j_0 = int(L/2+1)    # centre of wave packet (default = middle)
sigma = 10          # standard deviation of wave packet
k_0 = np.pi/2       # momentum of wave packet
chi = 30            # MPS matrix size

lambdas, Gammas = get_canonical_MPS(L, j_0, sigma, k_0, chi, True)