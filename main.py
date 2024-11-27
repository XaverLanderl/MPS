# imports
from MPS_lib import *

# parameters
L = 51
params = {
    'L'                 : L,
    'chi'               : int(np.log(L) + 1),
    'tau'               : 0.1,
    'J_z'               : 1.0,
    'J_xy'              : -1.0,
    'trunc_tol'         : 1,
    'show_S_z'          : True,
    'show_entropy'      : False,
    'show_progress'     : True,
    'show_disc_weights' : False
}

# initialize solver
solver = MPS_solver(**params)
solver.initialize_entangled_2spin_state(eta=1j)
solver.initialize_product_state([int(L/2+1)])
results = solver.run()