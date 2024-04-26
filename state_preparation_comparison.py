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
Created on Tue Feb 20 19:50:54 2024

@author: sea
"""

from zander_preparation import *

import networkx as nx
import random

def generate_random_tree_graph(num_nodes):
    # Create an empty directed graph
    G = nx.DiGraph()

    # Generate a list of nodes
    nodes = list(range(num_nodes))

    # Add nodes to the graph
    for node in nodes:
        G.add_node(node)

    # Add edges to form a tree
    for i in range(1, num_nodes):
        # Choose a random parent node
        parent = random.choice(nodes[:i])
        # Add an edge from the parent to the current node
        G.add_edge(i, parent)

    return G

from qrisp.quantum_backtracking import OHQInt

M = 4
N = 4
qtype = OHQInt(N)

for i in range(10):
    
    G = generate_random_tree_graph(M)
    q_array = QuantumArray(qtype = qtype, shape = (M))
    
    prepare_pbs_state(G, 0, N, q_array)
    
    meas_res = q_array.get_measurement()

    for oar in meas_res.keys():
        for n0, n1 in G.edges():
            assert oar[n0] != oar[n1]
    
    compiled_qc = q_array.qs.compile()
    print("======")
    print("QC depth: ", compiled_qc.depth())
    print("CNOT count: ", compiled_qc.depth())
    
    
    
