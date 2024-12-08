# imports
from gaussian_packet import *
from MPS_lib import *

# wave packet parameters
L = 51              # length of spin chain
j_0 = int(L/2+1)    # centre of wave packet (default = middle)
sigma = 4           # standard deviation of wave packet
k_0 = np.pi/2       # momentum of wave packet

# solver parameters
params = {
    'L'                 : L,
    'chi'               : 20,
    'tau'               : 0.1,
    'J_z'               : 1.0,
    'J_xy'              : 1.0,
    'trunc_tol'         : 1,
    'show_S_z'          : True,
    'show_entropy'      : True,
    'show_progress'     : True,
    'show_disc_weights' : False
}

wave_packet_params = {
    'L'     :   params['L'],
    'j_0'   :   int(params['L']/2+1),
    'sigma' :   2,
    'k_0'   :   np.pi/2,
    'chi'   :   params['chi'],
    'test'  :   True
}

# initialize solver
solver = MPS_solver(**params)
lambdas, Gammas, coeffs = get_canonical_MPS(**wave_packet_params)
solver.lambdas = lambdas
solver.Gammas = Gammas

# plot initial expectation value
plt.figure();
plt.plot(solver.single_site_expectation_value(solver.S_z).real)
plt.ylim([-1,1])
plt.xlabel('j');
plt.title('$\\langle S_{j}^{z}\\rangle, t = 0$');

# run the simulation
solver.run();