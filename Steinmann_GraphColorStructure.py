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

from qrisp import p, cp, cx, mcp, QuantumVariable
import numpy as np

class QuantumColor(QuantumVariable):
    """
    The QuantumColor is a custom QuantumVariable implemented with tackling the Max-k-Colorable-Subgraph problem
    and other coloring optimization problems in mind. It provides flexibility in choosing encoding methods and 
    leverages efficient data structures like QuantumArrays to enhance computational performance.

    The QuantumColor class takes as input a list of colors and a flag indicating the preferred encoding 
    method - binary or one-hot encoding. The choice of encoding method has implications for how colors are 
    represented in the quantum computation.

    In binary encoding, each color is represented by a unique binary number. For instance, if there are four 
    colors, Red, Green, Blue, and Yellow, they can be represented, for example, as [0,0], [0,1], [1,0], and [1,1] 
    respectively.

    In contrast, one-hot encoding represents each color as an array where only one element is 1 and the rest are 0. 
    Using the same four-color example, red can be represented as [1,0,0,0], green as [0,1,0,0], blue as [0,0,1,0], 
    and yellow as [0,0,0,1].

    Another key feature of the QuantumColor class is its use of QuantumArrays. QuantumArrays are data structures designed for efficient quantum computation. 
    They allow for compact representation and manipulation of quantum states and operators.

    Parameters.
    ----------
    list_of_colors : list
        The list of colors to be used in the quantum coloring problem.
    one_hot_enc : bool, optional
        The flag to indicate whether to use one-hot encoding. If False, binary encoding is used. We use the one-hot encoding by default.

    Attributes
    ----------
    list_of_colors : list
        The list of colors to be used in the quantum coloring problem.
    one_hot_enc : bool
        The indicator which tells the program whether to use one-hot encoding. If False, binary encoding is used.

    Methods
    -------
    decoder(i)
        Decode the color from the given index for both binary and one-hot encoding.
    """

    def __init__(self, list_of_colors, one_hot_enc = True): 
        """
        Initialize the QuantumColor with a list of colors and a flag indicating whether to use one-hot encoding.

        Parameters
        ----------
        list_of_colors : list
            The list of colors to be used in the coloring problem instance.
        one_hot_enc : bool, optional
            The flag to indicate whether to use one-hot encoding. If False, binary encoding is used. Default is True.

        """
        self.list_of_colors = list_of_colors
        self.one_hot_enc = one_hot_enc

        # If one-hot encoding is used, the size of QuantumVariable is the number of colors
        if one_hot_enc:
            QuantumVariable.__init__(self, size = len(list_of_colors)) 

        # If binary encoding is used, the size of QuantumVariable is the maximal value of log2 for the number of colors
        else:
            QuantumVariable.__init__(self, size = int(np.ceil(np.log2(len(list_of_colors)))))

    def decoder(self, i):
        """
        Decode the color from the given index i.

        Parameters
        ----------
        i : int
            The index to be decoded into a color.

        Returns
        -------
        str
            The decoded color if it exists, otherwise "undefined".
        
        """
        if not self.one_hot_enc:
            # Binary encoding: Each color is represented by a binary number.

            # For example, with four colors Red, Green, Blue and Yellow:
            #Red:   [0,0]
            #Green: [0,1]
            #Green: [1,0]
            #Yellow:[1,1]
            return self.list_of_colors[i]

        else:
            #One hot encoding: Each color is represented by an array where only one element is 1 and rest are 0.

            # For example, with four colors Red, Green, Blue and Yellow:     
            #Red:   [1,0,0,0]
            #Green: [0,1,0,0]
            #Yellow:[0,0,1,0]
            #Blue:  [0,0,0,1]

            is_power_of_two = ((i & (i-1) == 0) and i != 0)

            if is_power_of_two:
                return self.list_of_colors[int(np.log2(i))]

            else:
                return "undefined"
            
def initial_state_mkcs(qarg):
    """
    The initial_state_mkcs function provides the correct initial state of qubits in
    the system on which we run the optimization. In the case of the Max-k-Colorable 
    Subgraph problem, the initial state of the systemis simply any random coloring 
    of nodes of the graph.

    Parameters
    ----------
    qarg : QuantumArray
        A QuantumArray consisting of the color values for each node of graph G.


    Returns
    -------
    qarg : QuantumArray
        The quantum argument (in our case this is a QuantumArray) adapted to include 
        the information of the initial state of the system.
          
    """

    # Set all elements in qarg to initial state
    qarg[:] = init_state

    # Return updated quantum argument
    return qarg

def apply_phase_if_eq(qcolor_0, qcolor_1, gamma):
    """
    Applies a phase if the colors of the two arguments are matching.

    Parameters
    ----------
    qcolor_0 : QuantumColor
        The color of the first argument
    qcolor_1 : QuantumColor
        The color of the second argument
    gamma : integer
        The value of the gamma angle parameter used in QAOA

    Returns
    -------
    None.
        Applies a phase if the colors of the two arguments are matching.

    """
    if qcolor_0.one_hot_enc !=  qcolor_1.one_hot_enc:
          raise Exception("....")
    #Raise exception if color list if different
    if qcolor_0.one_hot_enc:
        #qbl = (qcolor_0 != qcolor_1)
        #p(2 * gamma, qbl)
        #qbl.uncompute()
        for i in range(qcolor_0.size):
            cp(2*gamma, qcolor_0[i], qcolor_1[i])

    else:
        cx(qcolor_0, qcolor_1)
        mcp(2*gamma, qcolor_1, ctrl_state = 0)
        cx(qcolor_0, qcolor_1)

def apply_phase_if_eq_multi(qcolors, qcolor_1, gamma):
    """
    Applies a phase if the colors of the two arguments are matching.

    Parameters
    ----------
    qcolor_0 : QuantumColor
        The color of the first argument
    qcolor_1 : QuantumColor
        The color of the second argument
    gamma : integer
        The value of the gamma angle parameter used in QAOA

    Returns
    -------
    None.
        Applies a phase if the colors of the two arguments are matching.

    """
    qcolor_0 = qcolor_1
    if qcolor_0.one_hot_enc !=  qcolor_1.one_hot_enc:
          raise Exception("....")
    #Raise exception if color list if different
    if qcolor_0.one_hot_enc:
        #qbl = (qcolor_0 != qcolor_1)
        #p(2 * gamma, qbl)
        #qbl.uncompute()
        for i in range(qcolor_0.size):
            cp(2*gamma, qcolor_0[i], qcolor_1[i])

    else:
        cx(qcolor_0, qcolor_1)
        mcp(2*gamma, qcolor_1, ctrl_state = 0)
        cx(qcolor_0, qcolor_1)


def create_coloring_operator(G):
    """
    Creates coloring operator and appliesa phase if two neighboring nodes
    on the graph are not equal in color.

    Parameters
    ----------
    G : nx.Graph
        Graph to color.

    Returns
    -------
    coloring_operator : function
        The coloring operator with correctly applied or not applied phase 
        based on if the color of two arguments in question are matching

    """
    def coloring_operator(quantumcolor_array, gamma):
        for pair in list(G.edges()):
            apply_phase_if_eq(quantumcolor_array[pair[0]],
                              quantumcolor_array[pair[1]], 
                              gamma)
    return coloring_operator 


# below: two alternative implementations for the graph_coloring operator
# first one should bring better results, second one is equivalent (?) to original operator

def create_coloring_operator_adjacent_singular(G):
    """
    Creates coloring operator and appliesa phase if two neighboring nodes
    on the graph are not equal in color.

    Parameters
    ----------
    G : nx.Graph
        Graph to color.

    Returns
    -------
    coloring_operator : function
        The coloring operator with correctly applied or not applied phase 
        based on if the color of two arguments in question are matching

    """
    def coloring_operator(quantumcolor_array, gamma):
        for node in list(G.nodes()):
            for adj in G.adj[node].keys():
                apply_phase_if_eq(quantumcolor_array[node],
                                quantumcolor_array[adj], 
                                gamma)
            

    return coloring_operator 

""" def color_in_array(qc_array, qcolor):
    from qrisp import mcx
    

    temp_qv = QuantumVariable(qv_0.size)

    cx(qv_0, temp_qv)
    cx(qv_1, temp_qv)
    x(temp_qv)

    res = QuantumBool()

    mcx(temp_qv, res)

    return res """


def create_coloring_operator_adjacent_all(G):
    """
    Creates coloring operator and appliesa phase if two neighboring nodes
    on the graph are not equal in color.

    Parameters
    ----------
    G : nx.Graph
        Graph to color.

    Returns
    -------
    coloring_operator : function
        The coloring operator with correctly applied or not applied phase 
        based on if the color of two arguments in question are matching

    """
    def coloring_operator(quantumcolor_array, gamma):
        for node in list(G.nodes()):
            adj_indeces = G.adj[node].keys()
            control_array = [quantumcolor_array[i] for i in range(len(G.nodes())) if i in adj_indeces]
            
            for color in control_array:
                with color == quantumcolor_array[node]:
                    mcp(2*gamma, quantumcolor_array[node],method= "gray", ctrl_state = 0)
            #if quantumcolor_array[node] in control_array:
                #mcp(2*gamma, quantumcolor_array[node], ctrl_state = 0)
            
    return coloring_operator 


def mkcs_obj(quantumcolor_array, G):
        # Set value of color integer to 1
    color = 1

        # Iterate over all edges in graph G
    for pair in list(G.edges()):

            # If colors of nodes in current pair are not same, multiply color by reward factor 4
        if quantumcolor_array[pair[0]] != quantumcolor_array[pair[1]]:
            color *= 10

        # Return negative color as objective function value. The negative value is used since we want to minimize the objective function       
    return -color

def create_coloring_cl_cost_function(G):
    """
    Creates the coloring classical cost function for graph G we are coloring.

    Parameters
    ----------
    G : nx.Graph
        Graph to color.

    Returns
    -------
    cl_cost_function : function
        Classical cost function, which in the end returns the ratio between 
        the energy calculated using the mkcs_obj objective funcion and the 
        amount of counts used in the experiment.

    """
    def cl_cost_function(counts):

        def mkcs_obj(quantumcolor_array, G):
            color = 1
            for pair in list(G.edges()):
                if quantumcolor_array[pair[0]] != quantumcolor_array[pair[1]]:
                    color *= 4
            return -color
    
        energy = 0
        total_counts = 0
        
        min_res = min([mkcs_obj(res, G) for res in counts.keys()])
        
        for meas, meas_count in list(counts.items())[::-1]:
            
            obj_for_meas = mkcs_obj(meas, G)
            
#            if obj_for_meas == min_res:
#                print(meas, obj_for_meas, "<=========== Optimal result")
#            else:
#                print(meas, obj_for_meas)
            
            energy += obj_for_meas * meas_count
            total_counts += meas_count
#        print("Total cost: ", energy/total_counts)
        return energy / total_counts
    return cl_cost_function

def graph_coloring_problem(G):
    """
    Creates a QAOA problem instance taking the phase separator, appropriate mixer, and
    appropriate classical cost function into account.

    Parameters
    ----------
    G : nx.Graph
        Graph to color.

    Returns
    -------
    QAOAProblem : function
        QAOA problem instance for graph coloring with which the QAOA algorithm is ran for.

    """    
    from qrisp.qaoa import QAOAProblem, XY_mixer

    return QAOAProblem(create_coloring_operator(G), XY_mixer, create_coloring_cl_cost_function(G))



def mkcs_obj(quantumcolor_array, G):

    # Set value of color integer to 1
    color = 1

    # Iterate over all edges in graph G
    for pair in list(G.edges()):

        # If colors of nodes in current pair are not same, multiply color by reward factor 4
        if quantumcolor_array[pair[0]] != quantumcolor_array[pair[1]]:
            color *= 10

    # Return negative color as objective function value. The negative value is used since we want to minimize the objective function       
    return -color

def create_coloring_cl_cost_function(G):
    """
    Creates the coloring classical cost function for graph G we are coloring.

    Parameters
    ----------
    G : nx.Graph
        Graph to color.

    Returns
    -------
    cl_cost_function : function
        Classical cost function, which in the end returns the ratio between 
        the energy calculated using the mkcs_obj objective funcion and the 
        amount of counts used in the experiment.

    """
    def cl_cost_function(counts):

        def mkcs_obj(quantumcolor_array, G):
            color = 1
            for pair in list(G.edges()):
                if quantumcolor_array[pair[0]] != quantumcolor_array[pair[1]]:
                    color *= 4
            return -color
    
        energy = 0
        total_counts = 0
        
        min_res = min([mkcs_obj(res, G) for res in counts.keys()])
        
        for meas, meas_count in list(counts.items())[::-1]:
            
            obj_for_meas = mkcs_obj(meas, G)
            
#            if obj_for_meas == min_res:
#                print(meas, obj_for_meas, "<=========== Optimal result")
#            else:
#                print(meas, obj_for_meas)
            
            energy += obj_for_meas * meas_count
            total_counts += meas_count
#        print("Total cost: ", energy/total_counts)
        return energy / total_counts
    return cl_cost_function