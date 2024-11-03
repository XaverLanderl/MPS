# imports
from MPS_lib import *
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm

# parameters
L = 51
chi = int(np.log(L) + 1)
tau = 0.01
J_z = 1.0
J_xy = 1.0
trunc_tol = 1e-10

# initialize solver
solver = MPS_solver(L=L, chi=chi, tau=tau, J_z=J_z, J_xy=J_xy, trunc_tol=trunc_tol)

# initialize state
spins_up = [int(L/2+1)]

for j in range(solver.L+1):
    solver.lambdas[j][0,0] = 1.0
    
for j in range(1,solver.L+1):
    # choose whether to assign up or down
    if True in (pos == j for pos in spins_up):
        solver.Gammas[j][1,0,0] = 1.0
    else:
        solver.Gammas[j][0,0,0] = 1.0

S_z = np.array([[1 , 0],
                [0 ,-1]])

# time evolution
num_steps = int(L/tau + 1)
RES = np.zeros(shape=(num_steps+1,L))
RES[0,:] = solver.single_site_expectation_value(S_z).real
time = [0.0]

for step in range(num_steps):
    
    # print out progress
    if step % 10 == 0:
        print('step = ' + str(step) + '/' + str(num_steps))
    
    # perform time-evolution step
    solver.apply_time_evolution()

    # measure observables
    RES[step+1,:] = solver.single_site_expectation_value(S_z).real
    time.append(time[-1]+tau)

# visualize result
t = np.array(time)
j = np.arange(1,L+1)
J,T = np.meshgrid(j,t)
plt.figure()
# Create the custom colormap
cmap = LinearSegmentedColormap.from_list("blue_white",["darkblue","white"])
plt.pcolormesh(J, T, RES, cmap=cmap, norm=PowerNorm(gamma=2))
plt.xlabel('j')
plt.ylabel('t')
plt.colorbar()