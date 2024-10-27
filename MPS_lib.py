### FUNCTIONS USED TO IMPLEMENT TEBD FOR MPS ###

# imports
import numpy as np

# time evolution operator
def time_evolution_matrix_elements(tau, J_z, J_xy):
    """
    Returns the matrix elements of the 2-site time evolution operator
    <s_1, s_i+1| e^{-i*tau*h_(i,i+1)} |s'_i, s'_i+1>

    Parameters
    ----------
    tau     :   time step
    J_z     :   longitudinal coupling
    J_xy    :   transverse coupling
    """

    # initialize result
    result = np.zeros(shape=(2,2,2,2), dtype=complex)

    # commonly used values
    exp_min = np.exp(-1j*tau*J_z/4)
    exp_plus = np.exp(1j*tau*J_z/4)
    cos_ = np.cos(tau/2 * np.abs(J_xy))
    sin_ = (-1j) * np.sin(tau/2 * np.abs(J_xy))

    # diagonal diagonal terms
    result[0,0,0,0] = exp_min
    result[1,1,1,1] = exp_min

    # opposite diagonal tersm
    result[0,1,0,1] = exp_plus * cos_
    result[1,0,1,0] = exp_plus * cos_

    # off diagonal terms
    result[0,1,1,0] = exp_plus * sin_
    result[1,0,0,1] = exp_plus * sin_

    # return result
    return result

# svd
def SVD(M):
    U, S, Vh = np.linalg.svd(M, full_matrices=False)
    return U, np.diag(S), Vh

class MPS_solver:
    """
    Class that does an TEBD calculation on a given MPS
    """

    def __init__(self, L, chi, tau, J_z, J_xy):
        """
        Class initializer.

        Parameters
        ----------
        L       :   length of the spin chain
        chi     :   matrix size
        tau     :   time step
        J_z     :   longitudinal coupling
        J_xy    :   transverse coupling
        """

        # set parameters
        self.L = L
        self.chi = chi
        self.tau = tau
        self.J_z = J_z
        self.J_xy = J_xy

        # empty matrix
        empt_l = np.zeros(shape=(self.chi,self.chi), dtype=complex)
        empt_G = np.zeros(shape=(2,self.chi,self.chi), dtype=complex)
            # first index:  0 = spin down
            #               1 = spin up

        # initialize matrixes
        self.lambdas = [np.zeros_like(empt_l)]
        self.Gammas = [None]    # there is no Gamma^0

        # fill list
        for j in range(1, self.L+1):
            self.lambdas.append(np.zeros_like(empt_l))
            self.Gammas.append(np.zeros_like(empt_G))

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
                M[spin,:,:] = self.lambdas[j-1] * self.Gammas[j][spin,:,:] * self.lambdas[j]

            # calculate expectation value
            for spin1 in range(2):
                for spin2 in range(2):
                    O_exp[j-1] += O[spin1,spin2] * np.trace(M[spin1,:,:].conjugate()*M[spin2,:,:])
            
        # return result
        return O_exp
    
    def apply_two_site_operator(self, O, j):
        """
        Applies a two-site operator O_{j,j+1} to the state.

        Parameters
        ----------
        self    :   self
        O       :   two-site operator
                :   numpy.ndarray, shape=(2,2,2,2)
        j       :   operator acts on sites j & j+1

        Returns
        -------
        Updated lambdas (j-1, j & j+1) and Gammas (j & j+1)
        """

        # get relevant matrices
        l1 = self.lambdas[j-1]
        l2 = self.lambdas[j]
        l3 = self.lambdas[j+1]
        G1 = self.Gammas[j]
        G2 = self.Gammas[j+1]

        # initialize theta
        Theta = np.zeros(shape=(2,2,self.chi, self.chi), dtype=complex)

        # fill theta
        for s1 in range(2):
            for s2 in range(2):
                Theta[s1,s2,:,:] = l1 @ G1[s1,:,:] @ l2 @ G2[s2,:,:] @ l3

        # apply operator
        Theta_new = np.zeros(shape=(2,2,self.chi, self.chi), dtype=complex)

        # perform spin sums
        for sp1 in range(2):
            for sp2 in range(2):
                for s1 in range(2):
                    for s2 in range(2):
                        Theta_new[sp1,sp2,:,:] += O[sp1,sp2,s1,s2] * Theta[s1,s2,:,:]
        
        # write as 2x2 matrix
        Theta_temp = np.zeros(shape=(2*self.chi,2*self.chi), dtype=complex)

        # assign blocks
        Theta_temp[:self.chi , :self.chi]  = Theta_new[0,0,:,:] 
        Theta_temp[:self.chi , self.chi:] = Theta_new[0,1,:,:]
        Theta_temp[self.chi: , :self.chi]  = Theta_new[1,0,:,:]
        Theta_temp[self.chi: , self.chi:] = Theta_new[1,1,:,:]

        # perform SVD
        U_new, S_new, Vh_new = SVD(Theta_temp)

        return U_new, S_new, Vh_new