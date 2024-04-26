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
from classical_cost_func import cost_function, format_coeffs, new_cost_function, cost_symp
import numpy as np
import matplotlib.pyplot as plt
from phase_polynomial import *
from scipy.optimize import minimize

from encode_BMW_problem import PBS_graph, N, M, cost_coeff
#from encode_random_problem import PBS_graph, N, M, cost_coeff

#np.random.seed(11122)

tot_coeff = format_coeffs(cost_coeff, N )

####################
# Compute optimal solution via unstructured search (brute force method)
####################

qtype = OHQInt(N)
q_array_2 = QuantumArray(qtype = qtype, shape = (M))

uniform_state = prepare_pbs_state(PBS_graph, 0, N, q_array_2)
meas_res = uniform_state.get_measurement()

cl_cost = cost_function(tot_coeff,M,N,PBS_graph)

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

####################
####################

qtype = OHQInt(N)
q_array = QuantumArray(qtype = qtype, shape = (M))

#classical cost function
values = []
cl_cost = cost_function(tot_coeff,M,N,PBS_graph,values)

#cost op
P,S = cost_symp(tot_coeff,M,N,PBS_graph)
ord_symbs=list(S.values())

def cost_op(q_array, gamma):
    app_sb_phase_polynomial(q_array, -gamma*P, ord_symbs)

#init func
init_func = pbs_state_init(PBS_graph, 0 , N)

#mixer
#mixer workaround
from mixer_workaround import * 
mixer_op = custom_mixer(PBS_graph,0,N)


params,symbols = init_params_symb(PBS_graph, 0, N)

prepare_pbs_state(PBS_graph, 0, N, q_array)

depth = 3
gamma = [Symbol("gamma_" + str(i)) for i in range(depth)]
beta = [Symbol("beta_" + str(i) + str(j)) for i in range(depth) for j in range(M)]
symbols = gamma + beta

for k in range(depth):
    cost_op(q_array,gamma[k])
    mixer_op(q_array,[beta[M*k+l] for l in range(M)])

qc=q_array.qs.compile() #parameterized compiling

def optimization_wrapper(theta, qc, symbols, qarg):

    subs_dic = {symbols[i] : theta[i] for i in range(len(symbols))}

    res_dic = qarg.get_measurement(subs_dic = subs_dic, precompiled_qc = qc)
    return cl_cost(res_dic)

optimization_method='COBYLA'
#optimization_method='SLSQP'
max_iter=100

init_point=2*np.pi * np.random.rand(len(symbols))


res_sample = minimize(optimization_wrapper,
                            init_point,
                            method=optimization_method,
                            options={'maxiter':max_iter},
                            args = (qc, symbols, q_array))  
    
subs_dic = {s : res_sample.x[i] for i,s in enumerate(symbols)}
res_dic = q_array.get_measurement(subs_dic = subs_dic, precompiled_qc = qc)
best_result= list(res_dic.keys())[0]

#
# Evaluations
#

cl_cost = cost_function(tot_coeff,M,N,PBS_graph)

print('###  Best assignments: ###')
print('   Cost:  ',cl_cost({best_result:1.0}))
print('   State: ',best_result[0])
print('   Prob:  ',list(res_dic.values())[0])

# print the results
#for k,v in res.items():
    #print("res")
    #print(k , cl_cost({k:1}), v)

####################
# Visualize results
####################
    
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

results = list(set([round(cl_cost({item[0]:1.0}),2) for item in res_dic.items()]))
probs = [0]*len(results)
for item in res_dic.items():
    cost = round(cl_cost({item[0]:1.0}),2)
    probs[results.index(cost)] += item[1]

axs[1].plot([min_cost,min_cost],[0,1], color='red')
axs[1].bar(results, probs, width=0.1)
# Add labels and title
axs[1].set_xlabel("Cost", fontsize = 14)
axs[1].set_ylabel("Probabilities", fontsize = 14)

####################

plt.show()
