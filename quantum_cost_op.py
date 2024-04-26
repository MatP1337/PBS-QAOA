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

from qrisp import *
from qrisp.quantum_backtracking import OHQInt
from zander_preparation import *
import numpy as np
import networkx as nx
from qrisp import *

def create_cost_operator(C, G, root, N):
    """
    A function for creating the quantum cost operator for the PBS problem

    Parameters
    ----------
    C : dict(dict)
        A dictionary of dictionaries, describing te pbs problem optimization targets. See example for formatting
    G : networkx.DiGraph
        The directed graph representing the PBS.
    root : int
        the root node.
    N : int
        The number of sites.

    Returns:
    --------
    cost_operator : function
        The cost operator of the problem
    """

    # quantum cost operator exp(-i*gamma*H)
    # $H = sum_{(r,s)\in\Phi}\sum_{i,j=0}^{N-1}C_{ij}^r*(1-Z_{ri}-Z_{sj}-Z_{ri}Z_{sj})
    def cost_operator(q_array, gamma):
        gamma = 2* gamma
        global_phase = 0
        phases = [0]*N

        recursive_create_cost_operator(q_array, gamma, C, G, root, N, phases, global_phase)
        gphase(-gamma/4*global_phase,q_array[0][0])
    
    return cost_operator

# recursive method for creating the cost operator
def recursive_create_cost_operator(q_array, gamma, C, G, node, N, phases, global_phase):
    """
    A recursive function for creating the quantum cost operator, based on the structure of the initial function call

    Parameters
    ----------
    q_array : QuantumArray(OHQInt)
        The datastructure to solve the problem on
    gamma : int
        the QAOA parameter for this cost layer
    C : dict(dict)
        A dictionary of dictionaries, describing te pbs problem optimization targets. See example for formatting
    G : networkx.DiGraph
        The directed graph representing the PBS.
    node : int
        the current node.
    N : int
        The number of sites.
    phases : int
        the phases to be applied and handed over in the recursive steps
    global_phase : int
        the global phase
    
    
    """
    predecessors = list(G.predecessors(node))
    pred_phases = {} # phases to pass to the cost operator for each predecessor of the current node
    for pred in predecessors:
        pred_phases[pred]= [0]*N

    for i in range(N):
        phase = 0
        for pred in predecessors:
            #print(C[pred])
            summe = 0
            for k, v in C[pred].items():
                if k[1] == i:
                    summe += v
            phase += summe # sum coefficients of Z_{sj} for all edges (r,s) in the graph (nodes r are predecessors of s)
            pred_phases[pred][i] = C[pred][(i,i)] # sum coefficients of Z_{ri}


        global_phase += phase

        # Z_{sj}
        rz(-gamma/2*(phase+phases[i]), q_array[node][i])

        for j in range(N):
            for pred in predecessors:
                if C[pred][(i,j)] != 0:
                    # Z_{ri}Z_{sj}
                    rzz(gamma/2*C[pred][(i,j)], q_array[pred][i], q_array[node][j])

    for pred in predecessors:
        recursive_create_cost_operator(q_array, gamma, C, G, pred, N, pred_phases[pred], global_phase)
