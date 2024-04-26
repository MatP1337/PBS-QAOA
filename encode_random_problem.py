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

np.random.seed(11222)

# DOES WORK:   
Phi = [(1,0),(2,0),(3,1), (4,2), (5,4)]
N = 4

PBS_graph = nx.DiGraph()
PBS_graph.add_edges_from(Phi)
M = PBS_graph.number_of_nodes()
#nx.draw(PBS_graph, with_labels = True)
#plt.show()

# Define the problem instance 

cost_coeff={} 
for i in range(1, len(PBS_graph.nodes)+1):
    cost_coeff[i] = {}
    for i2 in range(N):
        for i3 in range(i2,N):
            if not (i2,i3) in list(cost_coeff[i].keys()):
                random_num = round(np.random.uniform(0.5, 10), 2)
                cost_coeff[i].setdefault((i2,i3), random_num)

#print(cost_coeff)
                

