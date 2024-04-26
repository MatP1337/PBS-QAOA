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
from classical_cost_func import cost_function, format_coeffs, new_cost_function
import numpy as np
import matplotlib.pyplot as plt
from sympy import Symbol
from qrisp import *
from scipy.optimize import minimize


from encode_BMW_problem import PBS_graph, N, M, cost_coeff
#from encode_random_problem import PBS_graph, N, M, cost_coeff

#np.random.seed(11122)

tot_coeff = format_coeffs(cost_coeff, N )

qtype = OHQInt(N)
q_array = QuantumArray(qtype = qtype, shape = (M))

params,symbols = init_params_symb(PBS_graph, 0, N)

state = prepare_pbs_state(PBS_graph, 0, N, q_array, params)
qc=state.qs.compile() #parameterized compiling

# Initial parameters for uniform superposition
params_uniform = init_params(PBS_graph, 0, N, uniform=True)
init_uniform = []
for k,v in params_uniform.items():
    init_uniform.extend(v)

#classical cost function
values = []
cl_cost = new_cost_function(tot_coeff,M,N,PBS_graph,values) # Store intermediate values
cl_cost_2 = new_cost_function(tot_coeff,M,N,PBS_graph)


####################
# Compute optimal solution via unstructured search (brute force method)
####################

qtype = OHQInt(N)
q_array_2 = QuantumArray(qtype = qtype, shape = (M))

uniform_state = prepare_pbs_state(PBS_graph, 0, N, q_array_2)
meas_res = uniform_state.get_measurement()

solutions = {}
for k,v in meas_res.items():
    c = cl_cost_2({k:1})  
    solutions[k] = c
sorted_solutions = sorted(solutions.items(), key=lambda item: item[1])
min_cost = sorted_solutions[0][1]

print('###  Best assignment (brute force): ###')
print('   Cost:  ',sorted_solutions[0][1])
print('   State: ',sorted_solutions[0][0])
print('   Number of admissible states: ',len(sorted_solutions))

####################
####################


def optimization_wrapper(theta, qc, symbols, qarg):

    subs_dic = {symbols[i] : theta[i] for i in range(len(symbols))}

    res_dic = qarg.get_measurement(subs_dic = subs_dic, precompiled_qc = qc)
    return cl_cost(res_dic)

optimization_method='COBYLA'
#optimization_method='SLSQP'

store_dicts={}
max_iter=20
N_exp=5

# Running 10 times the optimization.
for exp in range(N_exp):
  print('Experiment #',exp)

  # random initial point
  if exp==0:  
    init_point = init_uniform    
  else:
    init_point=2*np.pi * np.random.rand(len(symbols))

  res_sample = minimize(optimization_wrapper,
                              init_point,
                              method=optimization_method,
                              options={'maxiter':max_iter},
                              args = (qc, symbols, q_array))  
    
  subs_dic = {s : res_sample.x[i] for i,s in enumerate(symbols)}
  res_dic = q_array.get_measurement(subs_dic = subs_dic, precompiled_qc = qc)
  best_result= list(res_dic.keys())[0]

  print('###  Best assignments: ###')
  print('   Cost:  ',cl_cost_2({best_result:1.0}))
  print('   State: ',best_result[0])
  print('   Prob:  ',list(res_dic.values())[0])
  store_dicts[exp]=res_dic

####################
# Visualize results
####################
  
# Create a figure and a 2x2 grid of subplots
fig, axs = plt.subplots(2, N_exp)

for exp in range(N_exp):
    # Optimal solution  
    axs[0, exp].plot([0,len(values)/N_exp],[min_cost,min_cost], color='red')

    x = list(range(max_iter))
    y = values[exp*max_iter:(exp+1)*max_iter]
    # Create a scatter plot
    axs[0, exp].scatter(x, y, color='black', s=10)

    # Add labels and title
    #axs[0, 0].xlabel("Iterations", fontsize = 14)
    #axs[0, 0].ylabel("Cost", fontsize = 14)
    #plt.show()
####################

for exp in range(N_exp): 
    # Optimal solution  
    axs[1, exp].plot([min_cost,min_cost],[0,1], color='red')

    results = [cl_cost_2({res:1.0}) for res in list(store_dicts[exp].keys())]
    probs = list(store_dicts[exp].values())
    # Create a scatter plot
    axs[1, exp].bar(results, probs, width=0.1)

    # Add labels and title
    #axs[1, 0].xlabel("Cost", fontsize = 14)
    #axs[1, 0].ylabel("Probability", fontsize = 14)

plt.show()