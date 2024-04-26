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

from Steinmann_GraphColorStructure import *
from qrisp.qaoa import QAOAProblem, apply_XY_mixer
from qrisp import QuantumArray, QuantumVariable
import networkx as nx
from operator import itemgetter
import numpy as np
import random

products = 7
sites = n_colors =4

#pbs = [(2, 1),(3, 1),(4, 1),(5, 2),(6, 2),(7, 2),(8, 3),(9, 4),(10, 4)]
pbs = [(0,1),(0,2), (1,3), (2,6), (2,5), (1,4)]
pbs_subparts = [(1,2),(5,6), (3,4)]#
def run_graphColorBMW(n_products, pbs,pbs_subparts,color_list,depth):
    G = nx.Graph()
    G.add_nodes_from(list(range(products)))
    G.add_edges_from(pbs)
    G.add_edges_from(pbs_subparts)
    num_nodes = len(G.nodes)
    color_list = ["red", "blue", "yellow"]

    #needs to be a matrix?
    #transport_matrix = np.ndarray(10,7,7)
    #site_allocations = np.ndarray()



    depth = 5

    # choose different functions for coloring operator, see structure file
    coloring_operator = create_coloring_operator_adjacent_singular(G)
    #coloring_operator = create_coloring_operator(G)
    #coloring_operator = create_coloring_operator_adjacent_all(G)


    use_quantum_array = True

    # Define quantum argument as a QuantumArray of QuantumColors
    qarg = QuantumArray(qtype = QuantumColor(color_list,one_hot_enc=True), shape = num_nodes) 


    # Define the initial state, which is a random coloring of all nodes
    init_state = [color_list[i % len(color_list) ] for i in range(len(G))]
    #init_state = [random.choice(color_list) for _ in range(len(G))]


    def initial_state_mkcs(qarg):

        # Set all elements in qarg to initial state
        qarg[:] = init_state

        # Return updated quantum argument
        return qarg

    from qrisp.interface import VirtualQiskitBackend
    # Set default backend for QAOA

    cl_cost_function = create_coloring_cl_cost_function(G)
    coloring_instance = QAOAProblem(coloring_operator, apply_XY_mixer, cl_cost_function) 
    coloring_instance.set_init_function(initial_state_mkcs) 
    res = coloring_instance.run(qarg, depth,)

    # Get the best solution and print it
    best_coloring, best_solution = min([(mkcs_obj(quantumcolor_array,G),quantumcolor_array) for quantumcolor_array in res.keys()], key=itemgetter(0))
    print(f"Best string: {best_solution} with coloring: {-best_coloring}")


    # Get final solution with optimized gamma and beta angle parameter values and print it
    best_coloring, res_str = min([(mkcs_obj(quantumcolor_array,G),quantumcolor_array) for quantumcolor_array in list(res.keys())[:5]], key=itemgetter(0))
    print("QAOA solution: ", res_str)
    print(res[res_str])
    print("top5")
    for key in list(sorted(res, key=res.get, reverse=True)[:5]):
        print(key)
        print(res[key])
    best_coloring, best_solution = (mkcs_obj(res_str,G),res_str)

    counts_good = 0
    counts_bad = 0
    for key,val in res.items():
        #if 0.05 <= val:
            #print(key)
            #print(val)
        is_sol = True
        for edge in G.edges():
            if not key[edge[0]] != key[edge[1]]:
                is_sol = False
        
        if is_sol:
            counts_good += val
        else:
            counts_bad += val

    print("split")
    print("good - bad")
    print(counts_good)
    print(counts_bad)
        
    print(len(res.keys()))
    for item, val in list(res.items()):
        if val > 0.002:
            print(item , float(val))

    # Draw graph with node colors specified by final solution
    nx.draw(G, node_color=res_str, with_labels=True)

    import matplotlib.pyplot as plt
    # Show plot
    plt.show()

            # Draw graph with node colors specified by final solution
    nx.draw(G, node_color=res_str, with_labels=True)

    import matplotlib.pyplot as plt
    # Show plot
    plt.show()

    return res

print(run_graphColorBMW(
    products,
    pbs = [(0,1),(0,2), (1,3), (2,6), (2,5), (1,4)],
    pbs_subparts = [(1,2),(5,6), (3,4)],
    color_list = ["red", "blue", "yellow"],
    depth = 5
))