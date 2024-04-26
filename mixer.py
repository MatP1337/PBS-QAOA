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

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 17:20:27 2024

@author: sea
"""

import networkx as nx
from zander_preparation_parametrized import prepare_pbs_state
from qrisp import *
from qrisp.quantum_backtracking import OHQInt


def inv_prepare(G, root, N, q_array):
    with invert():
        prepare_pbs_state(G, root, N,q_array)

def mixer(G, root, N):
    """
    A function for creating the constrained mixer for the PBS problem

    Parameters
    ----------
    G : networkx.DiGraph
        The directed graph representing the PBS.
    root : int
        the root node.
    N : int
        The number of sites.

    Returns:
    --------
    apply_mixer : function
        The mixer operator of the problem
    """

    def apply_mixer(q_array, beta):
        with conjugate(inv_prepare)(G, root, N, q_array):
            for i in range(len(q_array)):
                mcp(beta, q_array[i], ctrl_state = 0)
        #return q_array
    
    return apply_mixer







""" PBS_graph = nx.DiGraph([(1,0),(2,0), (3,0)])

M = PBS_graph.number_of_nodes()
N = 4
qtype = OHQInt(N)
q_array = QuantumArray(qtype = qtype, shape = (M))
prepare_pbs_state(PBS_graph, 0, N, q_array)

inital_state = q_array.duplicate(init = True)
print(inital_state)
beta = 1*np.pi/len(PBS_graph)
#beta = 1
#mixer(beta, PBS_graph, 0, N, q_array)
mixer_op = mixer(PBS_graph, 0, N)
#q_array2 = 
mixer_op(q_array,beta)
print(q_array)

initial_vs_mixed = multi_measurement([inital_state, q_array])

print(initial_vs_mixed)   """