# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 14:12:34 2024

@author: dexta
"""
import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.linalg import svd

#%% calculate Gauss-State
#clear plots
plt.close('all')

# Parameters
sigma = 4
factor = 2
size = 2*sigma + 1
k_0 = np.pi / 2
L = 50
x_0 = L/2  # Center of the Gaussian
chi = 20

x_values = np.linspace(-factor*sigma, factor*sigma, size) + x_0 # Exactly N points

N = L #len(x_values)  # Total number of sites 


def generate_configurations(num = size):
    # Generate all possible configurations of spins (0 for down, 1 for up)
    return list(itertools.product([0, 1], repeat=num))

def multiply_configuration(matrix, config, shift, num = size):
    
    result = np.diag(np.ones((num), dtype = complex))
    for site, spin in enumerate(config):
        #print(site, spin)
        result = np.copy(np.dot(result, matrix[site+shift][spin]))
        
    
    return result

def checkState(matrix, matrix_size,shift, num = size ):
    coeffs = []
    nonzero_config = []
    configurations = generate_configurations(num)
    for config in configurations:
        product = multiply_configuration(matrix, config, shift, matrix_size)
        coeffs.append(product[0][0])
        if not product[0][0] == 0:
            nonzero_config.append(config)
        #print(product)
    return coeffs, nonzero_config

coefficients = []
# Compute the wavefunction coefficients
for x in x_values:
    #print(np.exp(-(x - x_0)**2/ (2 * sigma**2)))
    #print(np.exp(1j * (x - x_0) * k_0))
    coefficients.append( np.exp(-(x - x_0)**2 / (2 * sigma**2)) * np.exp(1j * (x - x_0) * k_0))
    

# Normalize the coefficients
norm = np.sqrt(np.sum(np.abs(coefficients)**2))
normalized_coefficients = coefficients / norm
real_Coeffs = np.abs(normalized_coefficients)
shift = int(L/2 - len(coefficients)/2)

# plt.figure(figsize=(12, 8))
# plt.plot(range(len(normalized_coefficients)), normalized_coefficients)
# plt.plot(range(len(real_Coeffs)), real_Coeffs)


# Function to construct full NxN tensors
def construct_A(coeffs):
    tensors = np.full((L,2), np.array)
    blockTensor = np.full(L, np.array)
    
    for k in range(len(coeffs)):
        A_up = np.zeros((chi, chi), dtype=complex)  
        A_down = np.zeros((chi,chi), dtype=complex)
        A_down_diag = np.ones(size)
        np.fill_diagonal(A_down, A_down_diag)
        
        if k == 0:
            A_down = np.zeros((chi, chi), dtype=complex)
            A_down[0,1:] = 1
            A_up[0][0] = 1
            
        elif k == len(coeffs) - 1:  # last tensor
            A_down = np.zeros((chi, chi), dtype=complex)
            A_up[-1][0] = coeffs[k]
            A_down[:size,0] = coeffs
            A_down[-1][0] = 0
            
        else:  # Middle tensors
            A_down[k][k] = 0
            A_up[k][k] = 1
    
        tensors[int(L/2 -len(coeffs)/2) + k][0] = np.copy(A_up) 
        tensors[int(L/2 -len(coeffs)/2) + k][1] = np.copy(A_down)

        blockTensor[int(L/2 -len(coeffs)/2) + k] = np.copy(np.block([[A_up], [A_down]]))
    
    for site in range(int(L/2 -len(coeffs)/2)):
        trivial_A_up = np.zeros((chi,chi), dtype=complex)
        trivial_A_down = np.zeros((chi,chi), dtype=complex)
        trivial_A_down[0][0] = 1
        tensors[site][0] = trivial_A_up
        tensors[site][1] = trivial_A_down
        
        blockTensor[site] = np.block([[trivial_A_up],[trivial_A_down]])
    
    for site in range(int(L/2 +len(coeffs)/2), L):
        trivial_A_up = np.zeros((chi,chi), dtype=complex)
        trivial_A_down = np.zeros((chi,chi), dtype=complex)
        trivial_A_down[0][0] = 1
        tensors[site][0] = trivial_A_up
        tensors[site][1] = trivial_A_down
        
        blockTensor[site] = np.block([[trivial_A_up],[trivial_A_down]])
    
    
    return tensors, blockTensor

# Construct the MPS tensors
full_mps_tensors, mpsBlockTensor = construct_A(normalized_coefficients)
#full_mps_tensors, mpsBlockTensor = construct_A(coefficients)

mpsBlockTensor_reshaped = [[site[:chi, :], site[chi:, :]] for site in mpsBlockTensor]

check_initial_State, nonzero_config = checkState(full_mps_tensors, chi, shift)

plt.figure(figsize=(12, 8))
plt.plot(range(len(check_initial_State)), np.array(check_initial_State).real)
plt.plot(range(len(check_initial_State)), np.array(check_initial_State).imag)

#print(mpsBlockTensor[-2])

def checkIdentity(M1,M2):
    U_dagger_U = np.copy(M1 @ M2)
    comp = np.equal(U_dagger_U,np.identity(chi)) 

    if np.all(comp):
        print(i)
    else:
        diff = U_dagger_U - np.identity(chi)
        print(np.linalg.norm(diff, ord='fro'), np.linalg.norm(diff, ord=2))

#print(mpsBlockTensor)
#SVD block following below:
S_leftSweep = np.full(N, np.array)
A_leftSweep = np.full(N, np.array)
V_leftSweep = np.full(N, np.array)

check_Ms_LTR = np.full((L,2), np.array)
check_Ms_LTR_I = np.full((L,2), np.array)
check_Ms_LTR_Others = np.full((L,2), np.array)


checkI = True
numOfChecks = 2


for i in range(N):
    
    if i == 0:
        
        #A,S,V = np.linalg.svd(mpsBlockTensor[i], full_matrices=False)
        AI,SI,VI = svd(mpsBlockTensor[i],full_matrices=False)
        
        #!!!try here
        S_fullI = np.zeros((chi,chi), dtype=complex)  # Matrix mit Nullen in der richtigen Größe
        np.fill_diagonal(S_fullI, SI)  # Diagonale mit den Singularwerten füllen
        #S_full[0,:] = S
        #print(S)
        
        S_leftSweep[i] = np.copy(S_fullI)
        A_leftSweep[i] = np.copy(AI)
        V_leftSweep[i] = np.copy(VI)
        
        #print(S_full)
        
        if checkI:
            for ic in range(N):
                if ic == 0:
                    check_M_I = np.copy(np.dot(A_leftSweep[ic], np.dot(S_leftSweep[ic], V_leftSweep[ic])))
                    check_Ms_LTR_I[ic][0] = np.copy(check_M_I[:chi, :])
                    check_Ms_LTR_I[ic][1] = np.copy(check_M_I[chi:, :])
                else:
                    #print(full_mps_tensors[ic][0])
                    check_Ms_LTR_I[ic][0] = np.copy(full_mps_tensors[ic][0])
                    check_Ms_LTR_I[ic][1] = np.copy(full_mps_tensors[ic][1])
            
            check_LTR_StateI, nonzero_config_LTRI = checkState(check_Ms_LTR_I, chi, shift)

            plt.figure(figsize=(12, 8))
            plt.plot(range(len(check_initial_State)), np.array(check_LTR_StateI).real)
            plt.plot(range(len(check_initial_State)), np.array(check_LTR_StateI).imag)
    else:
        
        # if checkOthers:
        #     print(mpsBlockTensor[i], mpsBlockTensor[i][N:,:])
        
        #m_temp_up = np.copy(np.dot(S_leftSweep[i-1], np.dot(V_leftSweep[i-1], mpsBlockTensor[i][:chi,:])))
        #m_temp_down = np.copy(np.dot(S_leftSweep[i-1], np.dot(V_leftSweep[i-1], mpsBlockTensor[i][chi:,:])))

        v_temp_up = np.copy(V_leftSweep[i-1])
        v_temp_down = np.copy(V_leftSweep[i-1])
        s_temp_up = np.copy(S_leftSweep[i-1])
        s_temp_down = np.copy(S_leftSweep[i-1])
        mps_temp_up = np.copy(mpsBlockTensor[i][:chi, :])
        mps_temp_down = np.copy(mpsBlockTensor[i][chi:, :])
        
        m_temp_up = s_temp_up @ v_temp_up @ mps_temp_up
        m_temp_down = s_temp_down @ v_temp_down @ mps_temp_down

        m_temp = np.copy(np.block([[m_temp_up],[m_temp_down]]))
        
        #m_temp = np.concatenate((m_temp_up,m_temp_down), axis = 0)
        
        # if checkOthers:
        #     print(m_temp, m_temp_up, m_temp_down)
            
        #A,S,V = np.linalg.svd(m_temp, full_matrices=False)
        A,S,V = svd(m_temp,full_matrices=False)
        
        #checkIdentity(np.conj(A.T), A)
        
        #!!! try here 
        S_full = np.zeros((chi,chi), dtype=complex)  # Matrix mit Nullen in der richtigen Größe
        np.fill_diagonal(S_full, S)  # Diagonale mit den Singularwerten füllen
        #S_full[:,0] = S
                
        # m_test = np.copy(np.dot(A, np.dot(S_full,V)))
        # diff = m_temp - m_test
        # print(np.linalg.norm(diff, ord='fro'), np.linalg.norm(diff, ord=2))

        # comp = np.equal(m_test,m_temp)
        # print(comp)
        
        S_leftSweep[i] = np.copy(S_full)
        A_leftSweep[i] = np.copy(A)
        V_leftSweep[i] = np.copy(V)
        
        if i > 0:
            for ic in range(N):
                ############################################################################################
                if ic < i:
                    #print(ic)
                    check_M_O = A_leftSweep[ic]
                    check_Ms_LTR_Others[ic][0] = np.copy(check_M_O[:chi, :])
                    check_Ms_LTR_Others[ic][1] = np.copy(check_M_O[chi:, :])
                elif ic == i:
                    check_M_O = np.copy(np.dot(A_leftSweep[ic], np.dot(S_leftSweep[ic], V_leftSweep[ic])))
                    check_Ms_LTR_Others[ic][0] = np.copy(check_M_O[:chi, :])
                    check_Ms_LTR_Others[ic][1] = np.copy(check_M_O[chi:, :])
                ############################################################################################
                else:
                    #print(full_mps_tensors[ic][0])
                    check_Ms_LTR_Others[ic][0] = np.copy(full_mps_tensors[ic][0])
                    check_Ms_LTR_Others[ic][1] = np.copy(full_mps_tensors[ic][1])
            
            check_LTR_StateO, nonzero_config_LTRO = checkState(check_Ms_LTR_Others, chi, shift)

            plt.figure(figsize=(12, 8))
            plt.plot(range(len(check_initial_State)), np.array(check_LTR_StateO).real)
            plt.plot(range(len(check_initial_State)), np.array(check_LTR_StateO).imag)
            plt.title(i)
        checkOthers = False
    
    #print(check_M.shape)

### account for missing sign ###################################################
TEST_END = S_full @ V
discarded_sign = np.sign(TEST_END[0,0])
A_leftSweep[-1] *= discarded_sign

for i in range(N):
    check_M = A_leftSweep[i]
    check_Ms_LTR[i][0] = np.copy(check_M[:chi, :])
    check_Ms_LTR[i][1] = np.copy(check_M[chi:, :])

check_LTR_State, nonzero_config_LTR = checkState(check_Ms_LTR, chi, shift)

plt.figure(figsize=(12, 8))
plt.plot(range(len(check_initial_State)), np.array(check_LTR_State).real)
plt.plot(range(len(check_initial_State)), np.array(check_LTR_State).imag)

################################################################################    
assert False

S_rightSweep = np.full(N, np.array)
U_rightSweep = np.full(N, np.array)
B_rightSweep = np.full(N, np.array) 

check_Ms_RTL = np.full((N,2), np.array)       

for i in range(N-1, -1, -1):
    if i == N-1:
        vec_A = np.copy(A_leftSweep[i][:,0])     #need to transpose A to get from shape 18x9 to 9x18
        temp_A = np.zeros((2*chi,chi), dtype = complex)
        #temp_A[0,0] = np.copy(A_leftSweep[i][0,0])
        temp_A[:,0] = np.copy(vec_A)
        temp_A2 = np.copy(A_leftSweep[i].T)
        
        U,S,B = np.linalg.svd(temp_A.T, full_matrices=False)
        #U,S,B = np.linalg.svd(temp_A2, full_matrices=False)
        
        #!!! try here
        S_full = np.zeros((chi,chi), dtype=complex)  # Matrix mit Nullen in der richtigen Größe
        np.fill_diagonal(S_full, S)  # Diagonale mit den Singularwerten füllen
        #S_full[:,0] = S
        
        S_rightSweep[i] = np.copy(S_full)
        U_rightSweep[i] = np.copy(U)
        B_rightSweep[i] = np.copy(B)
        
    else:
        
        #print(B_rightSweep[i+1].shape)
        temp_A = np.copy(A_leftSweep[i].T)
        
        m_temp_up = np.copy(np.dot(np.dot( temp_A[:,:chi], U_rightSweep[i+1]), S_rightSweep[i+1]))
        m_temp_down = np.copy(np.dot(np.dot(temp_A[:,chi:], U_rightSweep[i+1]), S_rightSweep[i+1]))
        
        m_temp = np.copy(np.block([[m_temp_up, m_temp_down]]))
        U,S,B = np.linalg.svd(m_temp, full_matrices=False)
        
        #checkIdentity(B, np.conj(B.T))
        
        #!!! try here
        S_full = np.zeros((chi,chi), dtype=complex)  # Matrix mit Nullen in der richtigen Größe
        np.fill_diagonal(S_full, S)  # Diagonale mit den Singularwerten füllen
        #S_full[:,0] = S
        #print(S)
        
        # m_test = np.copy(np.dot(U, np.dot(S_full,B)))
        # diff = m_temp - m_test
        # print(np.linalg.norm(diff, ord='fro'), np.linalg.norm(diff, ord=2))
        
        S_rightSweep[i] = np.copy(S_full)
        U_rightSweep[i] = np.copy(U)
        B_rightSweep[i] = np.copy(B)
        
    check_M = np.copy(np.dot(U_rightSweep[i], np.dot(S_rightSweep[i], B_rightSweep[i])))
    #print(check_M.shape)
    check_Ms_RTL[i][0] = np.copy(check_M[:, :chi])
    check_Ms_RTL[i][1] = np.copy(check_M[:, chi:])
    
check_RTL_State, nonzero_config_RTL = checkState(check_Ms_RTL, chi, shift)

plt.figure(figsize=(12, 8))
plt.plot(range(len(check_RTL_State)), check_RTL_State)
