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

from sympy import Symbol
from qrisp import *
from zander_preparation_parametrized import *
from itertools import combinations

def get_Psi(Phi_graph):
    """
    the alternative graph, which defines same level nodes
    """
    Psi = []
    for node in Phi_graph.nodes():
        predecessors = list(Phi_graph.predecessors(node))
        pairs = list(combinations(predecessors, 2))
        Psi.extend(pairs)

    return Psi


def cost_symp(tot_coeff,n_parts,n_sites,Phi_graph,lambda_list,constraints = True):
    """
    returns the sympy cost function and the dictionary for the variables
    """
    #print(type(tot_coeff))
    combinations=list(tot_coeff[1].keys())

    x = {str(r)+str(i): Symbol(f"x{r}{i}") for r in range(n_parts) for i in range(n_sites)}
    #x = {(r,i): Symbol(f"x{r}{i}") for r in range(n_parts) for i in range(n_sites)}
    
    cost=sum([ sum([ tot_coeff[r][(i,j)]*x[str(r)+str(i)]*x[str(s)+str(j)] for (i,j) in combinations]) for r,s in Phi_graph.edges()])

    # C1 (One and ony one assignment of the site per part)
    C1 = 0
    for r in range(n_parts):
        curr = 0
        for i in range(n_sites):
            curr += x[str(r)+str(i)]
        C1 += (curr-1)**2

    # C2 (Origin and destination of each transport must be different)
    C2 = sum([ sum([ x[str(r)+str(i)]*x[str(s)+str(i)] for i in range(n_sites)]) for r,s in Phi_graph.edges()])

    # C3 (The origins of 2 sub-parts of a common part must be different)
    Psi = get_Psi(Phi_graph)
    C3 = sum([ sum([ x[str(r)+str(i)]*x[str(s)+str(i)] for i in range(n_sites)]) for r,s in Psi])

    if constraints:
        cost += lambda_list[0]*C1 + lambda_list[1]*C2 + lambda_list[2]*C3

    cost = cost.expand()
    
    return cost,x

# The following methods are used to move from the one hot encoding of the states
# to a binary one, in order to evaluate the cost function above.
# This can be done surely better with some qrisp feature and it will be changed.
def concatenate(outcome_array):
    binary_string = ''
    for element in outcome_array:
        conv=''.join(element)
        binary_string += conv#[::-1]
    
    return binary_string


def translate_encoding(outdic, N):
    return {concatenate(k):v  for k,v in outdic.items()}


def cost_function(tot_coeff,M,N,G,values=None):
    """
    This method returns a dictionary containing a list of abstract parameters for each node.

    Parameters
    ----------
    tot_coeff : dict(dict)
        A dictionary of dictionaries, describing te pbs problem optimization targets. See example for formatting
    M : int
        the number of PBS parts
    N : int
        The number of sites.
    G : networkx.DiGraph
        The directed graph representing the PBS.

    Returns
    -------
    cost_operator : function
        The classical cost function of the problem

    """
    #print(type(tot_coeff))
    qubo,symb_dic=cost_symp(tot_coeff,M,N,G)
    ord_symbs=list(symb_dic.values())
    vars_ = ord_symbs
    def classical_cost(res_dic):
        b_dic=translate_encoding(res_dic, N)
        cost=0
        for k,v in b_dic.items():
            qval = qubo.subs({var:s for var, s in zip(vars_, k)})
            cost +=  qval * v
        
        #print(cost)
        if values is not None:
            values.append(cost)

        return cost
    
    return classical_cost


"""
Below:
cost functions for NIklas' QUBO implementation

"""

def cost_for_QUBO_constrained_only_obj(tot_coeff,M,N,G,values=None):
    """
    classical cost function -- for calculation of objective function
    """
    def classical_cost_obj(res_dic):
        cost=0
        for sites,v in res_dic.items():
            for edge in G.edges():
                for i in range(N):
                    for j in range(N):
                        cost += v *  int(sites[edge[0]][i]) * int(sites[edge[1]][j]) * tot_coeff[edge[0]][(i,j)]

        return cost
    
    return classical_cost_obj



def cost_for_QUBO_only_constraints(tot_coeff,M,N,G,G_complement,lambda_list,values=None):
    """
    classical cost function -- for calculation of constraints
    """
    def classical_cost_constr(res_dic):
        lamb1, lamb2, lamb3 = lambda_list[0], lambda_list[1], lambda_list[2]

        def cost_cs1(res_dic):
            cost1 = 0
            for sites,v in res_dic.items():
                for a in range(len(G.nodes())):
                    summat = (sum([int(item) for item in list(sites[a])]) -1)
                    cost1 += v* (pow(summat, 2) )                
            return lamb1* cost1
                
        def cost_cs2(res_dic):
            cost2 = 0
            for sites,v in res_dic.items():
                for edge in G.edges():
                    for i in range(N):
                        cost2 += v *  int(sites[edge[0]][i])* int(sites[edge[1]][i])
            return lamb2* cost2

        def cost_cs3(res_dic):
            # the complementing graph from Rene's function is just a list
            cost3 = 0
            for sites,v in res_dic.items():
                for edge in G_complement:
                    for i in range(N):
                        cost3 += v *  int(sites[edge[0]][i])* int(sites[edge[1]][i])
                        
            return lamb3* cost3

        return cost_cs1(res_dic) + cost_cs2(res_dic) + cost_cs3(res_dic)
    
    return classical_cost_constr


def cost_for_QUBO_constrained(tot_coeff,M,N,G,G_complement,lambda_list,values=None):
    """
    classical cost function -- objective function and constraints
    """
    cost_func_obj = cost_for_QUBO_constrained_only_obj(tot_coeff,M,N,G,values=None)
    cost_func_constr = cost_for_QUBO_only_constraints(tot_coeff,M,N,G,G_complement,lambda_list,values=None)

    def classical_cost_full(res_dic):

        cost_obj= cost_func_obj(res_dic)
        cost_constr = cost_func_constr(res_dic) 
        cost =  cost_constr + cost_obj
        if values is not None:
            values.append(cost_obj)
        
        return cost
    
    return classical_cost_full



