from MPS_lib import *

# parameters
L = 50

# set up all four cases
cases_J_z = []
cases_J_xy = []

# first case
cases_J_z.append(0.0)
cases_J_xy.append(1.0)

# second case
cases_J_z.append(0.0)
J_xy_s = np.ones(shape=L-1)
J_xy_s[int(L/2):] = 0.5
cases_J_xy.append(J_xy_s)

# third case
cases_J_z.append(0.0)
J_xy_s = np.ones(shape=L-1)
J_xy_s[int(L/2)-1] = 0.5
cases_J_xy.append(J_xy_s)

# fourth case
cases_J_z.append(1.0)
cases_J_xy.append(1.0)



# go over all cases
for ind, J_z in enumerate(cases_J_z):

    # get parameters for this case
    params = {
        'L'                 : L,
        'chi'               : 4,
        'tau'               : 0.1,
        'J_z'               : J_z,
        'J_xy'              : cases_J_xy[ind],
        'trunc_tol'         : 1,
        'show_S_z'          : True,
        'show_entropy'      : True,
        'show_progress'     : True,
        'show_disc_weights' : True
    }

    # initialize solver
    solver = MPS_solver(**params)

    # initial state
    init_state = [i for i in range(1,int(solver.L/2)+1)]
    solver.initialize_product_state(list_of_spins_up=init_state)

    # run solver
    result = solver.run()

    # save result
    for ind_res, res in enumerate(result):
        file_name = 'case_' + str(ind+1) + '_res_' + str(ind_res+1) + '.csv'
        np.savetxt(file_name, res, delimiter=',')

# plots of results --> later
if False:
    # get list of times
    t = result[1][:,0]
    # get list of entanglements
    entanglement = result[3][:,int(solver.L/2)-1]

    # plot entanglement over time
    plt.figure()
    plt.plot(t, entanglement)
    plt.title('Entanglement at middle bond, linear plot')
    plt.xlabel('t')
    plt.ylabel('S')

    plt.figure()
    plt.semilogx(t,entanglement)
    plt.title('Entanglement at middle bond, logarithmic plot')
    plt.xlabel('ln(t)')
    plt.ylabel('S')