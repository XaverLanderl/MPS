# imports
from MPS_lib import *
import numpy as np

# parameters
L = 50
chi = 30
tau = 0.1
J_z = 1.0
J_xy = 1.0
trunc_tol = 1e-10

# initialize solver
solver = MPS_solver(L=L, chi=chi, tau=tau, J_z=J_z, J_xy=J_xy, trunc_tol=trunc_tol)

# initialize state
spins_up = [25,26]

for j in range(solver.L+1):
    solver.lambdas[j][0,0] = 1.0
    
for j in range(1,solver.L+1):
    # choose whether to assign up or down
    if True in (pos == j for pos in spins_up):
        solver.Gammas[j][1,0,0] = 1.0
    else:
        solver.Gammas[j][0,0,0] = 1.0

S_x = np.array([[0, 1] , [1 , 0]])
S_y = np.array([[0,-1j], [1j, 0]])
S_z = np.array([[1, 0] , [0 ,-1]])
S_x_exp = solver.single_site_expectation_value(S_x).real
S_y_exp = solver.single_site_expectation_value(S_y).real
S_z_exp = solver.single_site_expectation_value(S_z).real

# time evolution
O = time_evolution_matrix_elements(tau=tau, J_z=J_z, J_xy=J_xy)

U,S,Vh = solver.apply_two_site_operator(O, 10)


