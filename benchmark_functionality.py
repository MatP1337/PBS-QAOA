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
from classical_cost_func import cost_function, format_coeffs
from classical_cost_func_QUBO import get_Psi, cost_for_QUBO_constrained_only_obj, cost_for_QUBO_only_constraints
from phase_polynomial import *

from encode_BMW_problem import PBS_graph, N, M, cost_coeff, qubo_lambda_array
tot_coeff = format_coeffs(cost_coeff, N)

"""
BELOW:
Hardcoded routine to find minimal and maximal objective frunction values for all valid states
""" 
qtype = OHQInt(N)
q_array_2 = QuantumArray(qtype = qtype, shape = (M))
uniform_state = prepare_pbs_state(PBS_graph, 0, N, q_array_2)
meas_res = uniform_state.get_measurement()
old_cl_cost = cost_function(tot_coeff,M,N,PBS_graph)
solutions = {}
for k,v in meas_res.items():
    c = old_cl_cost({k:1})  
    solutions[k] = c
sorted_solutions = sorted(solutions.items(), key=lambda item: item[1])
min_cost = sorted_solutions[0][1]
max_cost = sorted_solutions[-1][1]


def valid_states_full_qubo(res_dic):
    """
    helper function for Niklas' QUBO implementation 
    --> get the valid states (i.e. non contraint violating) for the approx_ratio
    """
    #the cost functions
    cl_cost_obj = cost_for_QUBO_constrained_only_obj(tot_coeff,M,N,PBS_graph)
    cl_cost_constr = cost_for_QUBO_only_constraints(tot_coeff,M,N,PBS_graph,get_Psi(PBS_graph),qubo_lambda_array)
    dic_copy = copy.deepcopy(res_dic)
    # share of valid states
    sum_valid = 1
        
    for k,v in res_dic.items():
        
        if not (min_cost <= cl_cost_obj({k:1}) <= max_cost) :
            dic_copy.pop(k)
            sum_valid -=v
             
        elif not (cl_cost_constr({k:1}) <=0) :#
            #-len(k)*qubo_lambda_array[0]
            dic_copy.pop(k)
            sum_valid -=v

    return dic_copy, sum_valid, cl_cost_obj(dic_copy)


# defintion for this guy : see the slack paper
def approximation_ratio_PBS(res_dic, cost_function,  type = "default", kwargs = []):
    """
    this the approximation ratio, as defined for the protfolio optimization. (See the slack channel)
    """
    # this is the default case, i.e. the constrained mixer, no qubo case
    cl_cost = cost_function(* kwargs)
    approx_ratio = 0
    if type == "default":
        # nothing much to do here, since we only have valid states
        cost_val = cl_cost(res_dic)
        approx_ratio = (max_cost - cost_val)/ (max_cost-min_cost)
    
    #not valid for the sympy substitution approach
    elif type == "full_qubo":
        new_dic, sum_valid, cost_valid = valid_states_full_qubo(res_dic)
        #divide by sum_valid to "normalize"
        cost_val = cost_valid/sum_valid
        #multiply by sum_valid --> this then respects the approx_ratio = 0 part for the share of non valid states, which is (1-sum_valid)
        approx_ratio = sum_valid*(max_cost - cost_val)/ (max_cost-min_cost)
        
        
    else:
        raise Exception("Approx ratio for this type not implemented yet")
    

    return approx_ratio
