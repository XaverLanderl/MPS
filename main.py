# imports
from MPS_lib import *

# parameters
L = 51
chi = int(np.log(L) + 1)
tau = 0.1
J_z = 1.0
J_xy = 1.0
trunc_tol = 1

# initialize solver
solver = MPS_solver(L=L, chi=chi, tau=tau, J_z=J_z, J_xy=J_xy, trunc_tol=trunc_tol)
solver.initialize_state([int(L/2),int(L/2+1)])
results = solver.run()