# imports
from MPS_lib import *

# parameters
params = {
    'L'                 : 50,
    'chi'               : 10,
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
WALL = [21,22,23,24,25,26,27,28,29,30,49]
solver.initialize_product_state(WALL)

# run solver
solver.run();