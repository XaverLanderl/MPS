from exact_diag import *
from MPS_lib import visualize_result

# physical parameters
L = 51
J_z = 1
J_xy = 1

# calculation parameters
tau = 0.1
t_max = L
num_steps = int(t_max/tau) + 1

# initial state
gaussian = False
j_0 = int(L/2)+1    # centre of spin chain
sigma = 3
k_0 = -np.pi/2

# define initial state
if gaussian == False:
    init_state = np.zeros(shape=(L,1), dtype=complex)
    init_state[j_0] = 1.0
else:
    init_state = gaussian_coeff(L=L, j_0=j_0, sigma=sigma, k_0=k_0)

# calculate Hamiltonian
H = get_ith_block(L=L, k=1, J_z=J_z, J_xy=J_xy, do_timing=False)

# diagonalize H
d, P = np.linalg.eigh(H)

# get time evolution operator
U = P @ np.diag(np.exp(-1j*tau*d)) @ P.conj().T

# check
assert np.linalg.norm(P @ np.diag(d) @ P.conj().T - H) <= 1e-10

# exponentiate matrix
exp_iD = np.exp(-1j*d)
print(exp_iD.shape)

# set up results
t = [0.0]
RES = np.zeros(shape=(num_steps+1,L))

# get first expectation value
RES[0,:] = expectation_value(init_state)

# do the time evolution
state = init_state
for step in range(num_steps):

    # print progress
    print(str(step+1) + '/' + str(num_steps))

    # get time evolution operator
    state = U @ state

    # get expectation value
    RES[step+1,:] = expectation_value(state)

    # new time
    t.append(t[-1] + tau)

# visualization
J, T = np.meshgrid(np.arange(1,L+1),np.array(t))
visualize_result(J, T, RES)