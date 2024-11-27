### FUNCTIONS USED TO IMPLEMENT TEBD FOR MPS ###

# imports
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Define the colors based on the observed color scale in the image
list_of_colors = ["white", "lightblue", "blue", "cyan", "yellow", "red"]
num_cols = len(list_of_colors) - 1
colors = [(ind/num_cols, col) for ind, col in enumerate(list_of_colors)]


# Create the colormap
custom_cmap = LinearSegmentedColormap.from_list("custom_colormap", colors)

# time evolution operator
def time_evolution_nonzero_elements(tau, J_z, J_xy):
    """
    Returns the 3 distinct non-zero matrix elements of the time-evolution operator h_{j,j+1}.
    <s_1, s_i+1| e^{-i*tau*h_(i,i+1)} |s'_i, s'_i+1>

    Parameters
    ----------
    tau     :   time step
    J_z     :   longitudinal coupling
    J_xy    :   transverse coupling
    """

    # commonly used expressions
    exp_plus = np.exp(1j*tau*J_z/4)
    cos_ = np.cos(tau/2 * J_xy)
    sin_ = (-1j) * np.sin(tau/2 * J_xy)

    # non-zero matrix elements
    u_diag = np.exp(-1j*tau*J_z/4)
    u_off_diag = exp_plus * cos_
    u_flip = exp_plus * sin_

    # return results
    return u_diag, u_off_diag, u_flip

def visualize_result(J, T, RES, cmap=None):
    """
    Plots the results.
    
    Parameters
    ----------
    J, T    :   x- and y-grids
    RES     :   expectation values
    """

    plt.figure()
    if cmap == None:
        plt.pcolormesh(J, T, RES, cmap=custom_cmap)
    else:
        plt.pcolormesh(J, T, RES, cmap=cmap)
    plt.xlabel('j')
    plt.ylabel('t')
    plt.title('$\\langle S_{j}^{z} \\rangle $')
    plt.colorbar()

def visualize_entropy(J, T, ENT, cmap=None):
    """
    Plots the entanglement entropy.

    Parameters
    ----------
    J, T    :   x- and y-grids
    ENT     :   entanglement entropies
    """

    plt.figure()
    if cmap == None:
        plt.pcolormesh(J[:,:-1], T[:,:-1], ENT, cmap=custom_cmap)
    else:
        plt.pcolormesh(J[:,:-1], T[:,:-1], cmap=cmap)
    plt.xlabel('j')
    plt.ylabel('t')
    plt.title('Entanglement Entropy')
    plt.colorbar()

class MPS_solver:
    """
    Class that does an TEBD calculation on a given MPS
    """

    def __init__(self, L, chi, tau, J_z, J_xy, trunc_tol, show_S_z=True, show_entropy=True, show_progress=True, show_disc_weights=False):
        """
        Class initializer.

        Parameters
        ----------
        L           :   length of the spin chain
        chi         :   matrix size
        tau         :   time step
        J_z         :   longitudinal coupling
        J_xy        :   transverse coupling
        trunc_tol   :   threshold below which singular values are set to zero
        show_S_z    :   plot expectation value of S_z?
                    :   default = True
        show_entropy:   plot entanglement entropy?
                    :   default = True
        show_progress   print out progress?
                    :   default = True
        show_disc_weights   :   plot discarded weights?
                    :   default = False
        """

        # set parameters
        self.L = L
        self.chi = chi
        self.tau = tau
        self.J_z = J_z
        self.J_xy = J_xy
        self.trunc_tol = trunc_tol
        self.show_S_z = show_S_z
        self.show_entropy = show_entropy
        self.show_progress = show_progress
        self.show_disc_weights = show_disc_weights

        # S_z-operator
        self.S_z = np.array([[1 , 0],
                             [0 ,-1]])

        # initialize time evolution operators
        self.u_diag_even, self.u_off_diag_even, self.u_flip_even = time_evolution_nonzero_elements(tau/2, J_z, J_xy)
        self.u_diag_odd, self.u_off_diag_odd, self.u_flip_odd = time_evolution_nonzero_elements(tau, J_z, J_xy)

    def initialize_product_state(self, list_of_spins_up=[]):
        """
        Initializes the state as a product state of spins down, except at the entries given in list_of_spins_up.

        Parameters
        ----------
        self                :   self
        list_of_spins_up    :   list of spins initialized as up
                            :   default: all spins down

        Returns
        -------
        Initializes self.lambdas and self.Gammas   
        """

        # Check input
        for site in list_of_spins_up:
            if site > self.L:
                raise IndexError("Site index is out of bounds!")

        ### initialize with empty matrices
        # empty matrix
        empt_l = np.zeros(shape=(self.chi,self.chi), dtype=float)   # lambdas are real
        empt_G = np.zeros(shape=(2,self.chi,self.chi), dtype=complex)

        # initialize matrixes
        self.lambdas = [np.zeros_like(empt_l)]
        self.Gammas = [None]    # there is no Gamma^0

        # fill list
        for j in range(1, self.L+1):
            self.lambdas.append(np.zeros_like(empt_l))
            self.Gammas.append(np.zeros_like(empt_G))

        ### initialize state
        for j in range(self.L+1):
            self.lambdas[j][0,0] = 1.0
            
        for j in range(1,self.L+1):
            # choose whether to assign up or down
            if True in (pos == j for pos in list_of_spins_up):
                self.Gammas[j][0,0,0] = 1.0     # 0: up
            else:                               # 1: down
                self.Gammas[j][1,0,0] = 1.0

    def initialize_entangled_1spin_state(self, site=None, eta=1.0+0.0j):
        """
        Initializes the state as an entangled state.
        |psi> = 1/sqrt(2)*(c_(j-1)^+ + eta*c_j^+)

        Parameters
        ----------
        self                :   self
        site                :   site at which to place the entangled spin
                            :   default: middle of chain
        eta                 :   relative phase
                            :   default: 1

        Returns
        -------
        Initializes self.lambdas and self.Gammas   
        """

        # default: middle of chain
        if site == None:
            site = int(self.L/2 + 1)
        
        # check input
        if site > self.L:
            raise IndexError("Site index is out of bounds!")
        if np.abs(eta) != 1.0:
            raise ValueError("Phase must have magnitude 1!")

        ### initialize with empty matrices
        # empty matrix
        empt_l = np.zeros(shape=(self.chi,self.chi), dtype=float)   # lambdas are real
        empt_G = np.zeros(shape=(2,self.chi,self.chi), dtype=complex)

        # initialize matrixes
        self.lambdas = [np.zeros_like(empt_l)]
        self.Gammas = [None]    # there is no Gamma^0

        # fill list
        for j in range(1, self.L+1):
            self.lambdas.append(np.zeros_like(empt_l))
            self.Gammas.append(np.zeros_like(empt_G))

        ### initialize state
        for j in range(self.L+1):
            if j != (site - 1):
                self.lambdas[j][0,0] = 1.0
            elif j == (site - 1):
                self.lambdas[j][0,0] = 1/np.sqrt(2)
                self.lambdas[j][1,1] = 1/np.sqrt(2)
            
        for j in range(1,self.L+1):
            # choose whether to assign up or down
            if (j != (site - 1)) and (j != site):
                # assign spin down, index 1 = down
                self.Gammas[j][1,0,0] = 1.0
            elif j == (site - 1):
                self.Gammas[j][0,0,0] = 1.0
                self.Gammas[j][1,0,1] = 1.0
            elif j == site:
                self.Gammas[j][0,1,0] = eta
                self.Gammas[j][1,0,0] = 1.0

    def initialize_entangled_2spin_state(self, site=None, eta=1.0+0.0j):
        """
        Initializes the state as an entangled state.
        |psi> = 1/sqrt(2)*(c_(j-1)^+ + eta*c_j^+)

        Parameters
        ----------
        self                :   self
        site                :   site at which to place the entangled spin
                            :   default: middle of chain
        eta                 :   relative phase
                            :   default: 1

        Returns
        -------
        Initializes self.lambdas and self.Gammas   
        """

        # default: middle of chain
        if site == None:
            site = int(self.L/2 + 1)
        
        # check input
        if site > self.L:
            raise IndexError("Site index is out of bounds!")
        if np.abs(eta) != 1.0:
            raise ValueError("Phase must have magnitude 1!")

        ### initialize with empty matrices
        # empty matrix
        empt_l = np.zeros(shape=(self.chi,self.chi), dtype=float)   # lambdas are real
        empt_G = np.zeros(shape=(2,self.chi,self.chi), dtype=complex)

        # initialize matrixes
        self.lambdas = [np.zeros_like(empt_l)]
        self.Gammas = [None]    # there is no Gamma^0

        # fill list
        for j in range(1, self.L+1):
            self.lambdas.append(np.zeros_like(empt_l))
            self.Gammas.append(np.zeros_like(empt_G))

        ### initialize state
        for j in range(self.L+1):
            if (j != (site - 1)) and (j != site):
                self.lambdas[j][0,0] = 1.0
            elif (j == (site - 1)) or (j == site):
                self.lambdas[j][0,0] = 1/np.sqrt(2)
                self.lambdas[j][1,1] = 1/np.sqrt(2)
            
        for j in range(1,self.L+1):
            # choose whether to assign up or down
            if (j != (site - 1)) and (j != site) and (j != site + 1):
                # assign spin down, index 1 = down
                self.Gammas[j][1,0,0] = 1.0
            elif j == (site - 1):
                self.Gammas[j][0,0,0] = 1.0
                self.Gammas[j][1,0,1] = 1.0
            elif j == site:
                self.Gammas[j][0,0,0] = np.sqrt(2)
                self.Gammas[j][0,1,1] = np.sqrt(2)
            elif j == (site + 1):
                self.Gammas[j][0,1,0] = eta
                self.Gammas[j][1,0,0] = 1.0

    def run(self, t_max=None):
        """
        Runs the entire calculation.
        The state must have been initialized.

        Parameters
        ----------
        self                :   self
        list_of_spins_down  :   list of spins initialized as down
        t_max               :   maximal time reached
        """

        # do time evolution
        J, T, RES, ENT, disc_weights = self.perform_time_evolution(t_max)

        # plot results
        if self.show_S_z:
            visualize_result(J, T, RES)
        if self.show_entropy:
            visualize_entropy(J, T, ENT)

        # plot discarded weights
        if self.show_disc_weights:
            plt.figure()
            plt.plot(disc_weights)
            plt.xlabel('time step')
            plt.title('Discarded Weights')

        # return results
        return J, T, RES, ENT, disc_weights
        
    def single_site_expectation_value(self, O):
        """
        Returns the expectation values of single-site operators.

        Parameters
        ----------
        self    :   self
        O       :   single-site operator
                :   numpy.ndarray, shape=(2,2)

        Returns
        -------
        Expectation values  :    numpy.ndarray, shape=(L,)
        """

        # initialize result
        O_exp = np.zeros(shape=(self.L,), dtype=complex)

        # go over all sites
        for j in range(1, self.L+1):
            
            # get local state matrix
            M = np.zeros(shape=(2,self.chi,self.chi), dtype=complex)

            # fill with values
            for spin in range(2):
                M[spin,:,:] = self.lambdas[j-1] @ self.Gammas[j][spin] @ self.lambdas[j]

            # calculate expectation value
            for spin1 in range(2):
                for spin2 in range(2):
                    O_exp[j-1] += O[spin1,spin2] * np.trace((M[spin1].conj().T) @ M[spin2])
            
        # return result
        return O_exp
    
    def apply_two_site_operator(self, O, j):
        """
        Applies a general two-site operator of the form O_{j,j+1} to the state.
        We start counting sites at 1.

        Parameters
        ----------
        self    :   self
        O       :   two-site operator
                :   numpy.ndarray, shape=(2,2,2,2)
        j       :   operator acts on sites j & j+1
                :   j must be in the interval [1, L-1]

        Returns
        -------
        Updated lambda[j] and Gammas[j] & [j+1])
        Cannot act on the last site as there is no site L+1.
        """

        # get relevant matrices
        l_jm1 = self.lambdas[j-1]
        l_j = self.lambdas[j]
        l_jp1 = self.lambdas[j+1]
        G_j = self.Gammas[j]
        G_jp1 = self.Gammas[j+1]

        # initialize results
        l_j_new = l_j.copy()
        G_j_new = G_j.copy()
        G_jp1_new = G_jp1.copy()

        # initialize theta
        Theta = np.zeros(shape=(2,2,self.chi,self.chi), dtype=complex)

        # fill theta
        for s1 in range(2):
            for s2 in range(2):
                Theta[s1,s2,:,:] = l_jm1 @ G_j[s1,:,:] @ l_j @ G_jp1[s2,:,:] @ l_jp1

        # apply operator
        Theta_new = np.zeros(shape=(2,2,self.chi,self.chi), dtype=complex)

        # perform spin sums
        for sp1 in range(2):
            for sp2 in range(2):
                for s1 in range(2):
                    for s2 in range(2):
                        Theta_new[sp1,sp2,:,:] += O[sp1,sp2,s1,s2] * Theta[s1,s2,:,:]

        # write as 2x2 matrix
        Theta_temp = np.zeros(shape=(2*self.chi,2*self.chi), dtype=complex)

        # assign blocks
        Theta_temp[:self.chi,:self.chi]  = Theta_new[0,0,:,:] 
        Theta_temp[:self.chi,self.chi:] = Theta_new[0,1,:,:]
        Theta_temp[self.chi:,:self.chi]  = Theta_new[1,0,:,:]
        Theta_temp[self.chi:,self.chi:] = Theta_new[1,1,:,:]

        # perform SVD
        U, s, Vh = np.linalg.svd(Theta_temp, full_matrices=True)

        # trunctate matrices and renormalize S
        disc_weight_j = np.sum(s[self.chi:]**2)
        # check discarded weight:
        if disc_weight_j > self.trunc_tol:
            raise ValueError("Trunctation tolerance exceeded. Discarded weight = " + str(disc_weight_j))
        l_j_new = 1 / np.sqrt(1 - disc_weight_j) * np.diag(s[:self.chi])

        # split off lambdas
        l_jml_inv = np.linalg.pinv(l_jm1)
        l_jp1_inv = np.linalg.pinv(l_jp1)

        # and assign new Gammas
        G_j_new[0,:,:] = l_jml_inv @ U[:self.chi,:self.chi]
        G_j_new[1,:,:] = l_jml_inv @ U[self.chi:,:self.chi]
        G_jp1_new[0,:,:] = Vh[:self.chi,:self.chi] @ l_jp1_inv
        G_jp1_new[1,:,:] = Vh[:self.chi,self.chi:] @ l_jp1_inv

        # return results
        return G_j_new, l_j_new, G_jp1_new, disc_weight_j
    
    def apply_two_site_time_evolution(self, u_diag, u_off_diag, u_flip, j):
        """
        Applies the two-site time-evolution operator at site j.

        Parameters
        ----------
        self        :   self
        u_diag      :   diagonal elements of U
        u_off_diag  :   off-diagonal elements of U
        u_flip      :   "spin-flip" elements of U
        j           :   operator acts on sites j & j+1
                    :   j must be in the interval [1, L-1]

        Returns
        -------
        Updated lambda[j] and Gammas[j] & [j+1])
        Cannot act on the last site as there is no site L+1.
        """

        # get relevant matrices
        l_jm1 = self.lambdas[j-1]
        l_j = self.lambdas[j]
        l_jp1 = self.lambdas[j+1]
        G_j = self.Gammas[j]
        G_jp1 = self.Gammas[j+1]

        # initialize results
        l_j_new = l_j.copy()
        G_j_new = G_j.copy()
        G_jp1_new = G_jp1.copy()

        # initialize theta
        Theta = np.zeros(shape=(2,2,self.chi,self.chi), dtype=complex)

        # fill theta
        for s1 in range(2):
            for s2 in range(2):
                Theta[s1,s2] = l_jm1 @ G_j[s1] @ l_j @ G_jp1[s2] @ l_jp1

        # write new Theta as 2x2 matrix
        Theta_new = np.zeros(shape=(2*self.chi,2*self.chi), dtype=complex)

        # assign blocks of new Theta
        Theta_new[:self.chi,:self.chi] = u_diag * Theta[0,0]
        Theta_new[self.chi:,self.chi:] = u_diag * Theta[1,1]
        Theta_new[:self.chi,self.chi:] = u_off_diag * Theta[0,1] + u_flip * Theta[1,0]
        Theta_new[self.chi:,:self.chi] = u_flip * Theta[0,1] + u_off_diag * Theta[1,0]
        
        # perform SVD
        U, s, Vh = np.linalg.svd(Theta_new, full_matrices=True)

        # trunctate matrices and renormalize S
        disc_weight_j = np.sum(s[self.chi:]**2)
        # check discarded weight:
        if disc_weight_j > self.trunc_tol:
            raise ValueError("Trunctation tolerance exceeded. Discarded weight = " + str(disc_weight_j))
        l_j_new = 1 / np.sqrt(1 - disc_weight_j) * np.diag(s[:self.chi])

        # split off lambdas
        l_jml_inv = np.linalg.pinv(l_jm1)
        l_jp1_inv = np.linalg.pinv(l_jp1)

        # and assign new Gammas
        G_j_new[0,:,:] = l_jml_inv @ U[:self.chi,:self.chi]
        G_j_new[1,:,:] = l_jml_inv @ U[self.chi:,:self.chi]
        G_jp1_new[0,:,:] = Vh[:self.chi,:self.chi] @ l_jp1_inv
        G_jp1_new[1,:,:] = Vh[:self.chi,self.chi:] @ l_jp1_inv

        # return results
        return G_j_new, l_j_new, G_jp1_new, disc_weight_j
    
    def apply_time_evolution_step(self):
        """
        Performs a time-evolution step on the entire MPS.

        Parameters
        ----------
        self        : self

        Returns
        -------
        updates lambdas and Gammas.
        """

        disc_weight_max = 0

        # first, we must apply a half-time step to every even site.
        for j in range(1, self.L, 2):  # start counting at 1, only even sites

            # get new Gamma[j], lambda[j], Gamma[j+1]
            G_j_new, l_j_new, G_jp1_new, disc_weight_j = self.apply_two_site_time_evolution(self.u_diag_even, self.u_off_diag_even, self.u_flip_even, j)
            self.Gammas[j] = G_j_new
            self.lambdas[j] = l_j_new
            self.Gammas[j+1] = G_jp1_new

            # check accuracy
            if disc_weight_j > disc_weight_max:
                disc_weight_max = disc_weight_j
        
        # then, we must apply a full-time step to every odd site.
        for j in range(2, self.L, 2):  # start counting at 1, only odd sites

            # get new Gamma[j], lambda[j], Gamma[j+1]
            G_j_new, l_j_new, G_jp1_new, disc_weight_j =  self.apply_two_site_time_evolution(self.u_diag_odd, self.u_off_diag_odd, self.u_flip_odd, j)
            self.Gammas[j] = G_j_new
            self.lambdas[j] = l_j_new
            self.Gammas[j+1] = G_jp1_new

            # check accuracy
            if disc_weight_j > disc_weight_max:
                disc_weight_max = disc_weight_j

        # finally, we must apply a half-time step to every even site.
        for j in range(1, self.L, 2):  # start counting at 1, only even sites

            # get new Gamma[j], lambda[j], Gamma[j+1]
            G_j_new, l_j_new, G_jp1_new, disc_weight_j =  self.apply_two_site_time_evolution(self.u_diag_even, self.u_off_diag_even, self.u_flip_even, j)
            self.Gammas[j] = G_j_new
            self.lambdas[j] = l_j_new
            self.Gammas[j+1] = G_jp1_new

            # check accuracy
            if disc_weight_j > disc_weight_max:
                disc_weight_max = disc_weight_j

        # return result to monitor data
        return disc_weight_max
    
    def perform_time_evolution(self, t_max=None):
        """
        Evolves the MPS to t_max = L.

        Parameters
        ----------
        self        :   self
        t_max       :   maximal time reached
                    :   defaults to L

        Returns
        -------
        np.arrays containing j-axis, t-axis and expectation values
        """
        
        # get t_max
        if t_max == None:
            t_max = self.L
        
        # number of time-steps to reach t_max = L
        num_steps = int(t_max/self.tau + 1)

        # initialize result of expectation values
        RES = np.zeros(shape=(num_steps+1,self.L))
        RES[0,:] = self.single_site_expectation_value(self.S_z).real    # hermition operator: real expectation values
        
        # initialize list of times
        time = [0.0]

        # initialize list of maximimal discarded weights per time step
        disc_weights = []

        # initialize array for entanglement entropy
        ENT = np.zeros(shape=(num_steps+1,self.L-1))
        ENT[0,:] = self.calculate_entanglement_entropy()

        for step in range(num_steps):
            
            # print out progress every 100 time steps
            if (step % 100 == 0) and self.show_progress :
                print('step = ' + str(step) + '/' + str(num_steps))
            
            # perform time-evolution step
            disc_weight_max = self.apply_time_evolution_step()
            disc_weights.append(disc_weight_max)

            # measure observables
            RES[step+1,:] = self.single_site_expectation_value(self.S_z).real
            time.append(time[-1]+self.tau)
            ENT[step+1,:] = self.calculate_entanglement_entropy()

        # make x- and y-grids
        J,T = np.meshgrid(np.arange(1,self.L+1),np.array(time))

        # return results
        return J, T, RES, ENT, disc_weights
    
    def calculate_entanglement_entropy(self):
        """
        Calculates the entanglement entropy at every site.

        Parameters
        ----------
        self                    :   self

        Returns
        -------
        Entanglement entropy    :    numpy.ndarray, shape=(L-1,)
        """

        # initialize result
        entanglement_entropy = np.zeros(shape=(self.L-1))

        # go over all sites with left and right side
        for site in range(1,self.L):

            # perform the sum
            entropy = 0
            for i in range(self.chi):
                lambda_i_2 = self.lambdas[site][i,i]**2
                if lambda_i_2 != 0:
                    entropy += lambda_i_2 * np.log(lambda_i_2)
            
            # don't forget the minus sign
            entanglement_entropy[site-1] = -entropy
        
        # return result
        return entanglement_entropy
