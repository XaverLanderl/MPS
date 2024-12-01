### imports
import numpy as np
from matplotlib import pyplot as plt
import time

### some functions
def bin_sum(n):
    """
    Returns the number of ones in the binary representation of n.

    Parameters
    ----------
    int n   : natural number (including 0)

    Returns
    -------
    int     : number of ones in the binary representation of n
    """

    # initialize result
    result = 0

    # go over all numbers in the binary representation
    for k in bin(n)[2:]:
        result += int(k)

    # return result
    return result

def factorial(n):
    """
    Returns the factorial of n.

    Parameters
    ----------
    int n   : natural number (including 0)

    Returns
    -------
    int     : factorial of n
    """

    if n == 0:
        return 1
    else:
        return n * factorial(n-1)
    
def choose(n,k):
    """
    Returns the binomial coefficient n choose k.

    Parameters
    ----------
    int n   : number of elements
    int k   : number of hits

    Returns
    -------
    int     : n choose k
    """

    return int(factorial(n) / (factorial(k) * factorial(n - k)))

def swap_adjacent_spins(spin_config, ind):
    """
    Exchanges the characters in the passed string at positions ind and ind+1.

    Parameters
    ----------
    string spin_config  : spin configuration, passed as a string
    int ind             : index at which to swap elements

    Returns
    -------
    string              : spin configuration with swapped elements
    """

    # get length of config
    len_config = len(spin_config)

    # convert string to lost
    char_list = list(spin_config)

    # swap elements
    if ind != len_config-1:     # two adjacent elements
        char_list[ind], char_list[ind+1] = char_list[ind+1], char_list[ind]
    elif ind == len_config-1:   # first and last element
        char_list[-1], char_list[0] = char_list[0], char_list[-1]

    # transform back to a string
    result = ''.join(char_list)

    # return result
    return result

def generate_subspaces(L):
    """
    Sorts all numbers between 0 and 2**L - 1 (max L binary digits).
    Returnes L + 1 lists, each containing numbers with a fixed binary sum.

    Parameters
    ----------
    int L   : number of binary digits

    Returns
    -------
    list    : list of lists containing numbers
    """

    # initialize result
    list_of_lists = []
    list_of_dicts = []

    # L + 1 possible binary sums
    for k in range(L+1):
        list_of_lists.append([])
        list_of_dicts.append({})

    # go over all numbers and add
    for num in range(2**L):
        list_of_lists[bin_sum(num)].append(bin(num)[2:].zfill(L))

    # write to dictionaries
    for num_spins_up, subspace in enumerate(list_of_lists):
        state_index = 0
        for configuration in subspace:
            list_of_dicts[num_spins_up].update({configuration : state_index})
            state_index += 1

    # return result
    return list_of_lists, list_of_dicts

def get_block_hamiltonians(L, J_z, J_xy, do_timing=True):
    """
    Calculates the first L/2 blocks of the Hamiltonian.

    Parameters
    ----------
    int L           : length of the spin chain
    double J_z      : longitudinal coupling
    double J_xy     : transverse coupling
    bool do_timing  : time the procedure?

    Returns
    -------
    list of length int((L+1)/2)
    """

    if do_timing == True:
        time1 = time.time()

    # generate subspaces and bases within subspaces
    SUBSPACES, BASES = generate_subspaces(L)

    # initialize list of Hamiltonians
    list_of_hamiltonians = []

    # go over all subspaces
    for num_spins_up, subspace in enumerate(SUBSPACES):

        # need only half the blocks, abort loop if done
        if num_spins_up > int(L/2):
            break

        # initialize this subspaces's block-hamiltonian
        block_hamiltonian = np.diag(np.zeros(len(subspace)))

        # go over all configurations in the subspace
        for basis_state, configuration in enumerate(subspace):

            # create a numpy-array containing the actual spin values of the configuration
            spin_config = np.array([int(x) - 1/2 for x in configuration])
            
            # add diagonal part from longitudinal interaction 
            block_hamiltonian[basis_state, basis_state] += J_z*np.sum(spin_config*np.roll(spin_config,shift=1))

            # go over the current configuration to get non-diagonal contributions
            for site in range(L):
                
                # check if adjacent spins have opposite spin
                if configuration[site] != configuration[(site+1)%L]:

                    # get basis state index of contributing state
                    ind_cont = BASES[num_spins_up][swap_adjacent_spins(configuration, site)]
                    
                    # add contribution to the corresponding basis element
                    block_hamiltonian[basis_state, ind_cont] += J_xy/2

        # append to list of hamiltonians
        list_of_hamiltonians.append(block_hamiltonian)

    if do_timing == True:
        time2 = time.time()
        print('Generating the Hamiltonian took ' + str(round(time2-time1,4)) + 's.')

    # return results
    return list_of_hamiltonians

def calculate_partition_sums(list_of_hamiltonians, L, beta, h, do_timing=True):
    """
    Calculates the partition sums of the blocks and adds the magnetic field.

    Parameters
    ----------
    list_of_hamiltonians    : list containing the block-Hamiltonians
    int L                   : length of spin chain
    double beta             : inverse temperature
    double h                : magnetic field in z-direction
    bool do_timing          : time the procedure?

    Returns
    list_of_Z               : list of partition functions (length L+1)
    """

    if do_timing == True:
        time1 = time.time()
    
    # check input
    if not len(list_of_hamiltonians) == int(L/2+1):
        raise IndexError('There must be int(L/2+1) Hamiltonians!')
    
    # initialize results
    list_of_Z_1 = []
    list_of_Z_2 = []
    list_of_E_1 = []
    list_of_E_2 = []

    # go over all blocks
    for num_spins_up, block in enumerate(list_of_hamiltonians):

        # diagonalize the Hamiltonian
        eigvals = np.linalg.eigvalsh(block)

        # get magnetization
        mag = -L/2 + num_spins_up

        # get Z0
        Z0 = np.sum(np.exp(-beta*eigvals))

        E0_1 = np.sum((eigvals - h*mag)*np.exp(-beta*eigvals))
        E0_2 = np.sum((eigvals + h*mag)*np.exp(-beta*eigvals))

        # get Z with h
        list_of_Z_1.append(np.exp(beta*h*mag)*Z0)
        list_of_Z_2.append(np.exp(-beta*h*mag)*Z0)

        # get trace of He^-betaH in block
        list_of_E_1.append(np.exp(beta*h*mag)*E0_1)
        list_of_E_2.append(np.exp(-beta*h*mag)*E0_2)

    # add equivalent contributions
    if L%2 == 1:    # case L odd
        list_of_Z = list_of_Z_1 + list_of_Z_2[::-1]
        list_of_E = list_of_E_1 + list_of_E_2[::-1]
    elif L%2 == 0:  # case L even
        list_of_Z = list_of_Z_1 + list_of_Z_2[-2::-1] # must omit middle term
        list_of_E = list_of_E_1 + list_of_E_2[-2::-1]

    # check result
    if not len(list_of_Z) == L+1:
        raise IndexError('There must be L+1 blocks!')
    
    # get total Z and average M
    Z = 0
    M_average = 0
    for num_spins_up, Z_block in enumerate(list_of_Z):
        Z += Z_block
        M_average += (-L/2+num_spins_up)*Z_block
    M_average *= 1/Z

    # get average E
    E_average = 0
    for E_block in list_of_E:
        E_average += E_block
    E_average *= 1/Z

    if do_timing == True:
        time2 = time.time()
        print('Calculating Z took ' + str(round(time2-time1,4)) + 's.')

    # return results
    return list_of_Z, E_average, M_average

# parameters
L = 2           # length of spin chain
beta = 1        # inverse temperature
J_z = 2         # longitudinal coupling
J_xy = 2*J_z    # transverse coupling (default = isotropic Heisenberg model)
h = J_z/2       # magnetic field

list_of_hamiltonians = get_block_hamiltonians(L, J_z, J_xy, False)
list_of_Z, E_average, M_average = calculate_partition_sums(list_of_hamiltonians, L, beta, h, False)
print(E_average)

# free case
if False:
    h_list = np.linspace(-7,7,50)
    m_list = np.zeros(shape=h_list.shape)

    list_of_hamiltonians = get_block_hamiltonians(L, 0, 0, False)

    for ind, h in enumerate(h_list):
        list_of_Z, E_average, M_average = calculate_partition_sums(list_of_hamiltonians, L, beta, h, False)
        m_list[ind] = M_average/L

    plt.plot(h_list, m_list, label='simulation')
    plt.plot(h_list, 1/2*np.tanh(h_list/2/beta),'--', label='exact')
    plt.xlabel('h');
    plt.ylabel('m');

    plt.legend();
    plt.title('No coupling = free paramagnet');