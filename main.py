# imports
from MPS_lib import *

# parameters
L = 51
params = {
    'L'                 : L,
    'chi'               : int(np.log(L) + 1),
    'tau'               : 0.1,
    'J_z'               : 1.0,
    'J_xy'              : 1.0,
    'trunc_tol'         : 1,
    'show_disc_weights' : True
}

# initialize solver
solver = MPS_solver(**params)
solver.initialize_state([int(L/2+1)])
J, T, RES, ENT, disc_weights = solver.run()