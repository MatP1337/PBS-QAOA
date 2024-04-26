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

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sympy import Symbol
from qrisp import *
from qrisp.quantum_backtracking import OHQInt
from zander_preparation_parametrized import *
from scipy.optimize import minimize

def format_coeffs(coeff_dict, N, set_same_site_coeffs = False):
    tot_coeff = {}
    for k,v in coeff_dict.items():
        tot_coeff[k]={}
        for rs,c in v.items():
            i,j=rs[0],rs[1]
            if i > N-1 or j > N-1:
                continue
            tot_coeff[k][(i,j)]=c
            tot_coeff[k][(j,i)]=c
            #avoid movements from a destination to the same
            if set_same_site_coeffs: 
                tot_coeff[k][(i,i)]=1000 #cost suggested by the challenge text
                tot_coeff[k][(j,j)]=1000
            else:
                tot_coeff[k][(i,i)]=0  #cost suggested by the challenge text
                tot_coeff[k][(j,j)]=0

    return tot_coeff

def init_params_symb(graph, root, N):
    """
    This method returns a dictionary containing a list of abstract parameters for each node.

    Parameters
    ----------
    graph : networkx.DiGraph
        The directed graph representing the PBS.
    root : int
        The root of the graph.
    N : int
        The number sites.

    Returns
    -------
    params : dict
        A dictionary containing a list of abstract parameters for each node.
    symbols : list
        List of symbols associated to the abstract parameters

    """

    params = {}
    symbols=[Symbol("theta_"+str(root)+"_"+ str(i)) for i in range(N-1)]

    params[root] = [Symbol("theta_"+str(root)+"_"+ str(i)) for i in range(N-1)]

    recursive_init_params_symb(graph, root, N, params, symbols)

    return params,symbols

def recursive_init_params_symb(graph, node, N, params, symbols):
    """
    A recursive method for initializing a dictionary of abstract parameters for each node.

    Parameters
    ----------
    graph : networkx.DiGraph
        The directed graph representing the PBS.
    node : int
        The current node.
    N : int
        The number sites.
    q_array : QuantumArray
        The QuantumArray representing the PBS superposition state.
    params : dict
        A dictionary containing a list of angles for each node.

    """
    predecesors = list(graph.predecessors(node))
    m = len(predecesors)
    if(N<m+1):
        raise Exception(
                "Insufficient number of sites N"
        )

    for i in range(m):
        x=predecesors[i]
        params[predecesors[i]] = [ Symbol("theta_"+str(x)+"_"+ str(i))  for i in range(N-2-i)]

    for i in range(m):
      x=predecesors[i]
      for i in range(N-2-i):
        symbols.append(Symbol("theta_"+str(x)+"_"+ str(i)))


    # Recursivley add list of angles for predecessors
    for pred in predecesors:
        recursive_init_params_symb(graph, pred, N, params,symbols)


def cost_symp(tot_coeff,n_parts,n_sites,Phi_graph):
    """
    returns the sympy cost function and the dictionary for the variables
    """
    #print(type(tot_coeff))
    combinations=list(tot_coeff[1].keys())

    x = {str(r)+str(i): Symbol(f"x{r}{i}") for r in range(n_parts) for i in range(n_sites)}
    
    cost=sum([ sum([ tot_coeff[r][(i,j)]*x[str(r)+str(i)]*x[str(s)+str(j)] for (i,j) in combinations]) for r,s in Phi_graph.edges()])
    
    return cost,x

# The following methods are used to move from the one hot encoding of the states
# to a binary one, in order to evaluate the cost function above.
# This can be done surely better with some qrisp feature and it will be changed.
def convert_to_binary(outcome_array,N):
    binary_string = ''
    for element in outcome_array:
        conv=''.join('1' if i == element else '0' for i in range(N))
        binary_string += conv#[::-1]
    
    return binary_string


def translate_encoding(outdic, N):
    return {convert_to_binary(k, N):v  for k,v in outdic.items()}


def cost_function(tot_coeff,M,N,G):
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

        return cost
    
    return classical_cost


def new_cost_function(tot_coeff,M,N,G,values=None):

    def classical_cost(res_dic):
        cost=0
        for sites,v in res_dic.items():
            for edge in G.edges():
                cost += v*tot_coeff[edge[0]][(sites[edge[0]],sites[edge[1]])]

        if values is not None:
            values.append(cost)

        return cost
    
    return classical_cost

from itertools import combinations
def get_Psi(Phi_graph):
        Psi = []
        for node in Phi_graph.nodes():
            predecessors = list(Phi_graph.predecessors(node))
            pairs = list(combinations(predecessors, 2))
            Psi.extend(pairs)

        return Psi



