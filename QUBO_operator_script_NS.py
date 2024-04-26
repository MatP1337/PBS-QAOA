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
from qrisp import QuantumArray
from classical_cost_func import format_coeffs, get_Psi
import numpy as np
from phase_polynomial import *

from encode_BMW_problem import PBS_graph, N, M, cost_coeff, qubo_lambda_array

""""
Niklas' script for the QUBO approach
also includes benchmarking, and saving of all results
"""

#stuff i want to benchmark
valid_states_array = []
approx_ratio_array = []
cost_valid_array = []
five_best_states_array = []

tot_coeff = format_coeffs(cost_coeff, N , set_same_site_coeffs=True)

qtype = QuantumVariable(N)
q_array = QuantumArray(qtype = qtype, shape = (M))

#init func
init_func = pbs_state_init(PBS_graph, 0 , N)
def mixer_op(q_array, beta):
    for qv in q_array:
        rx(beta,qv)


from QUBO_quantum_cost_op import create_QUBO_cost_operator_constrained
cost_op = create_QUBO_cost_operator_constrained(tot_coeff,PBS_graph,0,N, qubo_lambda_array)

from classical_cost_func_QUBO import cost_for_QUBO_constrained, cost_for_QUBO_constrained_only_obj, cost_for_QUBO_only_constraints
PBS_compl = get_Psi(PBS_graph)

# seperated to evaluate the objective function and constraint performances of the algo individually0
#set values to empty list for plotting
values = []
#values= None
cl_cost = cost_for_QUBO_constrained(tot_coeff,M,N,PBS_graph,PBS_compl,qubo_lambda_array,values = values)
cl_cost_obj = cost_for_QUBO_constrained_only_obj(tot_coeff,M,N,PBS_graph)
cl_cost_constr = cost_for_QUBO_only_constraints(tot_coeff,M,N,PBS_graph,PBS_compl,qubo_lambda_array)

# run the QAOA
from qrisp.qaoa import *
qaoaPBS = QAOAProblem(cost_operator=cost_op ,mixer=mixer_op, cl_cost_function=cl_cost)
qaoaPBS.set_init_function(init_func)
depth = 3
res = qaoaPBS.run(q_array, depth,mes_kwargs={"shots": 5000} , max_iter = 100)
#
print("res")

from benchmark_functionality import approximation_ratio_PBS, valid_states_full_qubo
dict_valid, sum_valid, cost_valid =  valid_states_full_qubo(res)
approx_rat = approximation_ratio_PBS(res, cost_for_QUBO_constrained,  type = "full_qubo", kwargs = [tot_coeff,M,N,PBS_graph,PBS_compl,qubo_lambda_array])

approx_ratio_array.append(approx_rat)
valid_states_array.append(sum_valid)
cost_valid_array.append(cost_valid)

index=0 

name = 'res_dicts\saved_dictionary' + str(index) +'.pkl'
with open(name, 'wb') as f:
    pickle.dump(res, f)


name = r'res_dicts\valids\saved_dictionary' + str(index) +'.pkl'
with open(name, 'wb') as f:
    pickle.dump(dict_valid, f)
    


np.savetxt('scores_fullqubo.csv', [p for p in zip(approx_ratio_array, valid_states_array, cost_valid_array#, five_best_states_array
                                                  )], delimiter=',', fmt='%s')


# call function below with res from above. benchmark loop should be not used i guess. 
# also set values = []
from vizualization_and_bruteforce import vizualition_and_bruteforce
vizualition_and_bruteforce(res, values, cl_cost_obj)