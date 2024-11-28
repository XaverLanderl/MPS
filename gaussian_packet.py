### FUNCTIONS TO IMPLEMENT THE GAUSSIAN WAVE PACKET IN MPS

# imports
import numpy as np
import time

def get_A_matrices(L, j_0, sigma, k_0):
    """
    Calculates the non-canonical MPS representation of the wave packet.

    Parameters
    ----------
    L               : length of the spin chain
    j_0             : centre of the wave packet
    sigma           : standard deviation of the wave packet
    k_0             : momentum of the wave packet

    Returns
    -------
    A_up_matrices   : list containing the non-canonical A_up-matrices
    A_down_matrices : list containing the non-canonical A_down-matrices
    coeffs          : np.array containing the state expansion coefficients
    """

    # set up coefficients of state expansion
    coeffs = np.zeros(shape=(L,), dtype=complex)
    for j in range(L):
        coeffs[j] = np.exp(-(j - j_0)**2/(2*sigma**2)) * np.exp(1j*k_0*(j - j_0))

    # initialize list of A-matrix-diagonals
    A_up_diags = []
    A_down_diags = []

    # calculate the A-matrix-diagonals
    for j in range(L):
        A_new = np.zeros(shape=(L,), dtype=complex)
        A_new[j] = 1.0
        A_up_diags.append(A_new)
        A_down_diags.append(1.0 - A_new)

    # add expansion coefficients to the last diag
    A_up_diags[-1] *= coeffs
    A_down_diags[-1] *= coeffs

    # transform to matrices, initialize lists
    A_up_matrices = []
    A_down_matrices = []

    # first element is a row
    A_up_1 = np.zeros(shape=(L,L), dtype=complex)
    A_up_1[0,:L] = A_up_diags[0]
    A_down_1 = np.zeros(shape=(L,L), dtype=complex)
    A_down_1[0,:L] = A_down_diags[0]
    A_up_matrices.append(A_up_1)
    A_down_matrices.append(A_down_1)

    # middle elements are diagonal matrices
    for j in range(1,L-1):
        A_up_j = np.zeros(shape=(L,L), dtype=complex)
        A_up_j[:L,:L] = np.diag(A_up_diags[j])
        A_down_j = np.zeros(shape=(L,L), dtype=complex)
        A_down_j[:L,:L] = np.diag(A_down_diags[j])

        A_up_matrices.append(A_up_j)
        A_down_matrices.append(A_down_j)

    # last element is a column
    A_up_L = np.zeros(shape=(L,L), dtype=complex)
    A_up_L[:L,0] = A_up_diags[-1]
    A_down_L = np.zeros(shape=(L,L), dtype=complex)
    A_down_L[:L,0] = A_down_diags[-1]
    A_up_matrices.append(A_up_L)
    A_down_matrices.append(A_down_L)

    # check result
    coeffs_reconstr = np.zeros(shape=coeffs.shape, dtype=complex)
    for spin_at in range(L):

        X = np.diag(np.ones(L))

        for k in range(L):
            if k != spin_at:
                X = X @ A_down_matrices[k]
            else:
                X = X @ A_up_matrices[k]
        
        coeffs_reconstr[spin_at] = X[0,0]

    assert np.linalg.norm(coeffs_reconstr - coeffs) <= 1e-10

    # return result
    return A_up_matrices, A_down_matrices, coeffs

def perform_left_right_sweep(A_up_matrices, A_down_matrices, coeffs, test=False):
    """
    Performs a left-to-right SVD sweep to calculate the left-normalized B-matrices.

    Parameters
    ----------
    A_up_matrices   : list containing the non-canonical A_up-matrices
    A_down_matrices : list containing the non-canonical A_down-matrices
    coeffs          : np.array containing the state expansion coefficients
    test            : check results?; default = False

    Returns
    -------
    B_up_matrices   : list containing the left-canonical, normalized B_up-matrices
    B_down_matrices : list containing the left-canonical, normalized B_down-matrices
    """

    time1 = time.time()

    # get length of spin chain
    L = len(A_up_matrices)

    # initialize lists of B-matrices
    B_up_matrices = []
    B_down_matrices = []

    # initialize diagonal S-matrix and Vd-matrix
    S = np.diag(np.ones(L))
    Vd = np.diag(np.ones(L))

    for k in range(0,L):

        # define new M-matrices
        M_tilde_up = S @ Vd @ A_up_matrices[k]
        M_tilde_down = S @ Vd @ A_down_matrices[k]

        # combine M-matrices into a single matrix
        M_tilde = np.concatenate((M_tilde_up, M_tilde_down), axis=0)

        # SVD
        B, s, Vd = np.linalg.svd(M_tilde, full_matrices=False)
        S = np.diag(s)

        # save B-matrices
        m = int(B.shape[0]/2)
        B_up_matrices.append(B[:m,:])
        B_down_matrices.append(B[m:,:])

    # get sign of last matrix right
    sign = np.sign((S@Vd)[0,0]).real
    B_up_matrices[-1] *= sign
    B_down_matrices[-1] *= sign

    time2 = time.time()

    # test
    if test == True:
        coeffs_reconstr_norm = coeffs.copy()
        coeffs_reconstr_orig = coeffs.copy()

        for spin_at in range(L):
            
            X = np.diag(np.ones(L))

            for k in range(L):
                if k != spin_at:
                    X = X @ B_down_matrices[k]
                else:
                    X = X @ B_up_matrices[k]

            coeffs_reconstr_norm[spin_at] = X[0,0]     
            coeffs_reconstr_orig[spin_at] = sign*(S@Vd@X)[0,0]  # we changed the sign!

        # Compare WITH normalization    
        norm_coeffs = np.linalg.norm(coeffs)
        norm_diff_norm = np.linalg.norm(coeffs_reconstr_norm - 1/norm_coeffs*coeffs)
        if norm_diff_norm <= 1e-10:
            print('Normalized coefficients correctly reproduced! norm_diff = ' + str(norm_diff_norm))
        else:
            raise ValueError('Normalized coefficients NOT correctly reconstructed! norm_diff = ' + str(norm_diff_norm))
        
        # Compare WITHOUT normalization
        norm_diff_orig = np.linalg.norm(coeffs_reconstr_orig - coeffs)
        if norm_diff_orig <= 1e-10:
            print('Original coefficients correctly reproduced! norm_diff = ' + str(norm_diff_orig))
        else:
            print(sign)
            raise ValueError('Original coefficients NOT correctly reconstructed! norm_diff = ' + str(norm_diff_orig))

        # check left-normalization of Bs
        for k in range(L):
            uni = B_up_matrices[k].conj().T @ B_up_matrices[k] + B_down_matrices[k].conj().T @ B_down_matrices[k]
            uni_dist = np.linalg.norm(uni - np.diag(np.ones(L)))
            if  uni_dist <= 1e-10:
                passed = True
            else:
                passed = False
                break
        if passed == True:
            print('B-matrices are all left-normalized! uni_dist = ' + str(uni_dist))
        else:
            raise ValueError('B-matrices must are NOT left-normalized! uni_dist = ' + str(uni_dist))
        print()

        time3 = time.time()
        print('runtime calculation = ' + str(time2-time1) + 's')
        print('runtime checks = ' + str(time3-time2) + 's')
        print()

    # return results
    return B_up_matrices, B_down_matrices

def perform_right_left_sweep(B_up_matrices, B_down_matrices, coeffs, test=False):
    """
    Performs a right-to-left SVD sweep to calculate the right-normalized C-matrices.
    Extracts the lambda-matrices.

    Parameters
    ----------
    B_up_matrices   : list containing the left-canonical, normalized B_up-matrices
    B_down_matrices : list containing the left-canonical, normalized Bs_down-matrices
    coeffs          : np.array containing the state expansion coefficients
    test            : check results?; default = False

    Returns
    -------
    C_up_matrices   : list containing the right-canonical, normalized C_up-matrices
    C_down_matrices : list containing the right-canonical, normalized C_down-matrices
    lambdas         : list containing the lambda-matrices
    """

    time1 = time.time()

    # get length of spin chain
    L = len(B_up_matrices)

    # initialize lists of B-matrices
    C_up_matrices = []
    C_down_matrices = []

    # initialize final lambda
    lambda_L = np.zeros(shape=(L,L))
    lambda_L[0,0] = 1.0

    # initialize diagonal S-matrix and U-matrix
    U = lambda_L.copy()
    S = lambda_L.copy()

    # initialize list of lambda-matrices
    lambdas = [lambda_L]    # put last element in already

    for k in list(range(L))[::-1]:  # right-to-left!

        # define new M-matrices
        M_tilde_up = B_up_matrices[k] @ U @ S
        M_tilde_down = B_down_matrices[k] @ U @ S

        # combine M-matrices into a single matrix
        M_tilde = np.concatenate((M_tilde_up, M_tilde_down), axis=1)

        # SVD
        U, s, C = np.linalg.svd(M_tilde, full_matrices=False)
        S = np.diag(s)

        # save C-matrices
        m = int(C.shape[1]/2)
        C_up_matrices.append(C[:,:m])
        C_down_matrices.append(C[:,m:])

        # save lambda-matrix
        lambdas.append(S)

    # the lists are all the wrong way around
    C_up_matrices = C_up_matrices[::-1]
    C_down_matrices = C_down_matrices[::-1]
    lambdas = lambdas[::-1]

    # get sign of first matrix right
    sign = np.sign((U@S)[0,0]).real
    C_up_matrices[0] *= sign
    C_down_matrices[0] *= sign

    # check that first lambda is equal to last lambda
    if np.linalg.norm(lambdas[0] - lambdas[-1]) <= 1e-10:
        if test == True:
            print('First and last lambda consistent!')
        lambdas[0] = lambda_L
    else:
        raise ValueError('First and last lambda must be equal!')

    time2 = time.time()

    # test
    if test == True:
        coeffs_reconstr_norm = coeffs.copy()

        for spin_at in range(L):
            
            X = np.diag(np.ones(L))

            for k in range(L):
                if k != spin_at:
                    X = X @ C_down_matrices[k]
                else:
                    X = X @ C_up_matrices[k]

            coeffs_reconstr_norm[spin_at] = X[0,0]

        # Compare WITH normalization    
        norm_coeffs = np.linalg.norm(coeffs)
        norm_diff_norm = np.linalg.norm(coeffs_reconstr_norm - 1/norm_coeffs*coeffs)
        if norm_diff_norm <= 1e-10:
            print('Normalized coefficients correctly reproduced! norm_diff = ' + str(norm_diff_norm))
        else:
            print(np.linalg.norm(coeffs_reconstr_norm + 1/norm_coeffs*coeffs))
            raise ValueError('Normalized coefficients NOT correctly reconstructed! norm_diff = ' + str(norm_diff_norm))
        
        # check right-normalization of Bs
        for k in range(L):
            uni = C_up_matrices[k] @ C_up_matrices[k].conj().T + C_down_matrices[k] @ C_down_matrices[k].conj().T
            uni_dist = np.linalg.norm(uni - np.diag(np.ones(L)))
            if uni_dist <= 1e-10:
                passed = True
            else:
                passed = False
                break
        if passed == True:
            print('C-matrices are all right-normalized! uni_dist = ' + str(uni_dist))
        else:
            raise ValueError('C-matrices must are NOT right-normalized! uni_dist = ' + str(uni_dist))
        print()

        # check traces of lambdas
        for k in range(L+1):
            tr = np.trace(lambdas[k]**2)
            tr_dist = np.abs(tr - 1.0)
            if tr_dist <= 1e-10:
                passed = True
            else:
                passed = False
                break
        if passed == True:
            print('lambdas are all correctly normalized!')
        else:
            raise ValueError('lambdas are NOT correctly normalized! tr_dist = ' + str(tr_dist))

        time3 = time.time()
        print('runtime calculation = ' + str(time2-time1) + 's')
        print('runtime checks = ' + str(time3-time2) + 's')
        print()

    # return results
    return C_up_matrices, C_down_matrices, lambdas

def get_Gammas(C_up_matrices, C_down_matrices, lambdas, chi, coeffs, test=False):
    """
    Calculates the Gamma-matrices from the C-matrices and the lambdas.

    Parameters
    ----------
    C_up_matrices   : list containing the right-canonical, normalized B_up-matrices
    C_down_matrices : list containing the right-canonical, normalized Bs_down-matrices
    chi             : MPS matrix size
    coeffs          : np.array containing the state expansion coefficients
    test            : check results?; default = False

    Returns
    -------
    lambdas         : list containing the lambda-matrices (trunctated to chi x chi)
    Gammas          : list containing the spin-resolved Gamma-matrices (trunctated to chi x chi)
    """

    time1 = time.time()

    # get length of spin chain
    L = len(C_up_matrices)

    # check consistency
    if L == len(lambdas) - 1:
        pass
    else:
        raise TypeError('len(lambdas) must be len(C_up_matrices) + 1!')
    
    # initialize Gamma-list, None as first entry
    Gammas = [None]

    for k in range(1,L+1):

        # define spin-resolved Gamma-matrix
        Gamma_k = np.zeros(shape=(2,L,L), dtype=complex)

        # get pseudo-inverse of lambda(k-1)
        l_pinv = np.linalg.pinv(lambdas[k-1])

        # get Gamma-matrices with pseudo-inverse
        Gamma_k[0] = l_pinv @ C_up_matrices[k-1]
        Gamma_k[1] = l_pinv @ C_down_matrices[k-1]
        # Note: need lambda[k-1]^-1 * C[k], but due to indexing
        # lambda[k] = lambdas[k]
        # C_matrices[k] = C_matrices[k-1]
        # this leads to consistent indexing of Gammas and lambdas!

        # append to list
        Gammas.append(Gamma_k)

    # check that trunctation size is not too large
    if chi >= L:
        raise ValueError('Matrices are already maximum size!')
    
    # check that no large singular values are discarded
    max_val = 0
    for k in lambdas:
        large_singulars = np.sum(np.diag(k) >= 1e-10)
        if large_singulars > max_val:
            max_val = large_singulars
    if max_val >= chi:
        raise ValueError('Cannot discard large singular values!')
    
    # trunctate Gammas and lambdas
    lambdas_trunct = []
    Gammas_trunct = [None]  # don't forget the indexing convention!
    for l in lambdas:
        lambdas_trunct.append(l[:chi,:chi])
    for ind, G in enumerate(Gammas):
        if ind != 0:    # first element is none for consistent indexing!
            Gammas_trunct.append(G[:,:chi,:chi])
    
    # re-label
    lambdas = lambdas_trunct
    Gammas = Gammas_trunct

    time2 = time.time()

    # test
    if test == True:
        coeffs_reconstr_norm = coeffs.copy()

        for spin_at in range(L):

            X = lambdas[0]

            for k in range(1,L+1):
                if k != spin_at + 1:
                    X = X @ Gammas[k][1]
                else:
                    X = X @ Gammas[k][0]
                X = X @ lambdas[k]
            
            coeffs_reconstr_norm[spin_at] = X[0,0]

        # Compare WITH normalization    
        norm_coeffs = np.linalg.norm(coeffs)
        norm_diff_norm = np.linalg.norm(coeffs_reconstr_norm - 1/norm_coeffs*coeffs)
        if norm_diff_norm <= 1e-10:
            print('Normalized coefficients correctly reproduced! norm_diff = ' + str(norm_diff_norm))
        else:
            print(np.linalg.norm(coeffs_reconstr_norm + 1/norm_coeffs*coeffs))
            raise ValueError('Normalized coefficients NOT correctly reconstructed! norm_diff = ' + str(norm_diff_norm))

        time3 = time.time()
        print('runtime calculation = ' + str(time2-time1) + 's')
        print('runtime checks = ' + str(time3-time2) + 's')
        print()

    # return results
    return lambdas, Gammas

def get_canonical_MPS(L, j_0, sigma, k_0, chi, test=False):
    """
    Calculates the canonical MPS representation of a Gaussian wave packet.

    Parameters
    ----------
    L               : length of the spin chain
    j_0             : centre of the wave packet
    sigma           : standard deviation of the wave packet
    k_0             : momentum of the wave packet
    chi             : MPS matrix size
    test            : check results?; default = False

    Returns
    -------
    lambdas         : list containing the lambda-matrices
    Gammas          : list containing the spin-resolved Gamma-matrices
    """

    # get non-canonical representation
    non_canon = get_A_matrices(L, j_0, sigma, k_0)
    A_up_matrices, A_down_matrices, coeffs = non_canon

    # left-right sweep
    print('##### ----- B-matrices ----- #####')
    left_canon = perform_left_right_sweep(A_up_matrices, A_down_matrices, coeffs, test)
    B_up_matrices, B_down_matrices = left_canon

    # right-left sweep
    print('##### ----- C-matrices ----- #####')
    right_canon = perform_right_left_sweep(B_up_matrices, B_down_matrices, coeffs, test)
    C_up_matrices, C_down_matrices, lambdas = right_canon

    # extract Gammas
    print('##### ----- Gamma-matrices ----- #####')
    canon = get_Gammas(C_up_matrices, C_down_matrices, lambdas, chi, coeffs, test=True)

    # return results
    return canon