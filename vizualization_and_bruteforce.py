"""
\********************************************************************************
* Copyright (c) 2024 the Qrisp authors
*
* This program and the accompanying materials are made available under the
* terms of the Eclipse Public License 2.0 which is available at
* http://www.eclipse.org/legal/epl-2.0.
*
* This Source Code may also be made available under the following Secondary
* Licenses when the conditions for such availability set forth in the Eclipse
* Public License, v. 2.0 are satisfied: GNU General Public License, version 2
* with the GNU Classpath Exception which is
* available at https://www.gnu.org/software/classpath/license.html.
*
* SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
********************************************************************************/
"""

from zander_preparation_parametrized import *
from qrisp.quantum_backtracking import OHQInt
from qrisp import QuantumArray
from classical_cost_func import new_cost_function, format_coeffs
import matplotlib.pyplot as plt
from phase_polynomial import *

from encode_BMW_problem import PBS_graph, N, M, cost_coeff
#from encode_random_problem import PBS_graph, N, M, cost_coeff

def vizualition_and_bruteforce(res, values, cost_function_qubo =None):
    """
    vizualization and brute force solution turned to a function
    """
    tot_coeff = format_coeffs(cost_coeff, N , set_same_site_coeffs=True)
    ################ RENES STUFF
    qtype = OHQInt(N)
    q_array_2 = QuantumArray(qtype = qtype, shape = (M))
    uniform_state = prepare_pbs_state(PBS_graph, 0, N, q_array_2)
    meas_res = uniform_state.get_measurement()
    cl_cost = new_cost_function(tot_coeff,M,N,PBS_graph)

    solutions = {}
    for k,v in meas_res.items():
        c = cl_cost({k:1})  
        solutions[k] = c
    sorted_solutions = sorted(solutions.items(), key=lambda item: item[1])
    min_cost = sorted_solutions[0][1]

    print('###  Best assignment (brute force): ###')
    print('   Cost:  ',sorted_solutions[0][1])
    print('   State: ',sorted_solutions[0][0])
    print('   Number of admissible states: ',len(sorted_solutions))


    # Create a figure and a 2x2 grid of subplots
    fig, axs = plt.subplots(2)
    fig.subplots_adjust(hspace=0.5) 

    ####################

    axs[0].plot([0,len(values)],[min_cost,min_cost], color='red')
    x = list(range(len(values)))
    y = values
    # Create a scatter plot
    axs[0].scatter(x, y, color='black', s=10)
    # Add labels and title
    axs[0].set_xlabel("Iterations", fontsize = 14)
    axs[0].set_ylabel("Cost", fontsize = 14)

    ####################
    if cost_function_qubo is not None:
        cl_cost = cost_function_qubo
    results = list(set([round(cl_cost({item[0]:1.0}),2) for item in res.items()]))
    probs = [0]*len(results)
    for item in res.items():
        cost = round(cl_cost({item[0]:1.0}),2)
        probs[results.index(cost)] += item[1]

    axs[1].plot([min_cost,min_cost],[0,1], color='red')
    axs[1].bar(results, probs, width=0.1)
    # Add labels and title
    axs[1].set_xlabel("Cost", fontsize = 14)
    axs[1].set_ylabel("Probabilities", fontsize = 14)

    plt.show()