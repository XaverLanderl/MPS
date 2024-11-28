# imports
from gaussian_packet import *
from MPS_lib import *

# set parameters
L = 51              # length of spin chain
j_0 = int(L/2+1)    # centre of wave packet (default = middle)
sigma = 3           # standard deviation of wave packet
k_0 = np.pi/2       # momentum of wave packet
params = {
    'L'                 : L,
    'chi'               : 30,
    'tau'               : 0.1,
    'J_z'               : 1.0,
    'J_xy'              : 1.0,
    'trunc_tol'         : 1,
    'show_S_z'          : True,
    'show_entropy'      : False,
    'show_progress'     : True,
    'show_disc_weights' : False
}

# initialize solver
solver = MPS_solver(**params)
lambdas, Gammas = get_canonical_MPS(L, j_0, sigma, k_0, params['chi'], True)
solver.lambdas = lambdas
solver.Gammas = Gammas
solver.run();