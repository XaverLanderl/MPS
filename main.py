# imports
from MPS_lib import *

# parameters
L = 50
chi = 2*int(np.log(L) + 1)
tau = 0.1
J_z = 1.0
J_xy = 1.0
trunc_tol = 1

# initial state
list_of_spins_down = [int(L/2), int(L/2+1)]

# initialize solver
solver = MPS_solver(L=L, chi=chi, tau=tau, J_z=J_z, J_xy=J_xy, trunc_tol=trunc_tol, show_disc_weights=True)

results = solver.run([int(L/4), int(3*L/4)], t_max=3*L)