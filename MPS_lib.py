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
        empt_l = np.zeros(shape=(self.chi,self.chi))
        empt_G = np.zeros(shape=(2,self.chi,self.chi))
            # first index:  0 = spin down
            #               1 = spin up

        # initialize matrixes
        self.lambdas = [np.zeros_like(empt_l)]
        self.Gammas = [None]    # there is no Gamma^0

        # fill list
        for j in range(1, self.L+1):
            self.lambdas.append(np.zeros_like(empt_l))
            self.Gammas.append(np.zeros_like(empt_G))

    def expectation_value(self, O):
        """
        Returns the expectation values of single-site operators.

        Parameters
        ----------
        self    :   self
        O       :   operator that acts on a single site
                :   numpy.ndarray, shape=(2,2)

        Returns
        -------
        Expectation values  :    numpy.ndarray, shape=(L,)
        """

        # initialize result
        O_exp = np.zeros(shape=(self.L,))

        # go over all sites
        for j in range(1, self.L+1):
            
            # get local state matrix
            M = np.zeros(shape=(2,self.chi,self.chi))

            # fill with values
            for spin in range(2):
                M[spin,:,:] = self.lambdas[j-1] * self.Gammas[j][spin,:,:] * self.lambdas[j]

            # calculate expectation value
            for spin1 in range(2):
                for spin2 in range(2):
                    O_exp[j-1] += O[spin1,spin2] * np.trace(M[spin1,:,:].conjugate()*M[spin2,:,:])
            
        # return result
        return O_exp