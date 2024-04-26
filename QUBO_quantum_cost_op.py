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

from quantum_cost_op import * 
from zander_preparation import *
from qrisp import *
from phase_polynomial import app_sb_phase_polynomial
from classical_cost_func_QUBO import cost_symp


def create_QUBO_cost_operator_constrained(C, G, root, N, lambda_array):
    """
    Niklas' QUBO implementation
    QUBO quantum cost operator for the PBS problem, including the constraints translated into gates. Operates on QuantumVariable-based array, not OHQInt.

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

    lamb1 = lambda_array[0]
    lamb2 = lambda_array[1]
    lamb3 = lambda_array[2]

    
    def cost_operator(q_array, gamma):
        gamma = 2* gamma
        global_phase = 0
        phases = [0]*N
        # Rene's new cost_op implementation
        # this then solves the unconstrained optimization
        P,S = cost_symp(C,len(G.nodes),N,G, constraints= False)
        ord_symbs=list(S.values())
        app_sb_phase_polynomial(q_array, -gamma*P, ord_symbs)


        # first constraint
        for a in range(len(G.nodes())):
            for i in range(N):

                rz( 2* lamb1 * gamma,q_array[a][i])
                for j in range(N):
                    #with control (q_array[a] == j):
                        # phase gate for I^2 ? 
                    if i==j : 
                        rz( -lamb1 * gamma,q_array[a][i])
                        continue
                    rzz(-lamb1 * gamma , q_array[a][i], q_array[a][j])
                    # Two rz for the -Z_ai, -Z_aj
                    rz(- lamb1 * gamma,q_array[a][i])
                    rz(- lamb1 * gamma,q_array[a][j]) 

        # second constraint
        for edge in G.edges():
            for i in range(N):
          
                cp(-lamb2 * gamma,q_array[edge[0]][i], q_array[edge[1]][i])

        # third constraint
        """ alt_edges = []
        edgeset = copy.deepcopy(list(G.edges()))
        edgeset2 = copy.deepcopy(list(G.edges()))
        for edge4 in edgeset:
            for edge2 in edgeset2:
                if edge2[1] == edge4[1]:
                    if not edge2[0] == edge4[0]:
                        if edge2[0] > edge4[0]:
                            alt_edges.append((edge2[0], edge4[0]))
                        else: 
                            alt_edges.append((edge4[0], edge2[0])) """
        
        from classical_cost_func_QUBO import get_Psi

        alt_edge_list = get_Psi(G)
        for edge3 in alt_edge_list:
            for i in range(N):
                cp(-lamb3 * gamma,q_array[edge3[0]][i], q_array[edge3[1]][i])

    
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





    #TODO
    # third constraint: figure out hot to loop over same height pairs
    #fk the first constraint for now, since it its fullfilled by the encoding anyways
