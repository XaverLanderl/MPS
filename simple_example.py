# imports
from MPS_lib import *

# parameters
params = {
    'L'                 : 50,
    'chi'               : 4,
    'tau'               : 0.1,
    'J_z'               : 15.0,
    'J_xy'              : 1.0,
    'trunc_tol'         : 1,
    'show_S_z'          : True,
    'show_entropy'      : False,
    'show_progress'     : True,
    'show_disc_weights' : False
}

# initialize solver
solver = MPS_solver(**params)

# make a wall
WALL = [1]
solver.initialize_product_state(WALL)

# run solver
solver.run();