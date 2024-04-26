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
Created on Tue Feb 20 19:45:44 2024

@author: sea
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from qrisp import *

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


def init_params(graph, root, N, uniform=False):
    """
    This method returns a dictionary containing a list of (random) parameters for each node.

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
        A dictionary containing a list of (random) parameters for each node.

    """
    params = {}
    if uniform:
        params[root] = [2*np.arcsin(1/np.sqrt(N-i)) for i in range(N-1)]
    else:
        params[root] = [2*np.pi*np.random.random() for i in range(N-1)]

    recursive_init_params(graph, root, N, params, uniform)

    return params


def recursive_init_params(graph, node, N, params, uniform):
    """
    A recursive method for initializing a dictionary of parameters for each node.

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
        if uniform:
            params[predecesors[i]] = [2*np.arcsin(1/np.sqrt(N-1-i-j)) for j in range(N-2-i)]
        else:
            params[predecesors[i]] = [2*np.pi*np.random.random() for j in range(N-2-i)]

    # Recursivley add list of angles for predecessors
    for pred in predecesors:
        recursive_init_params(graph, pred, N, params, uniform)


def W(qv, n, angles=None):
    """
    This algorithms prepares tensor product of a (parametrized) W-state of the first n>=1 qubits of the QuantumVariable qv and the |0> state for the remaining qubits.
    We call such a state a partial W-state of size n.

    Parameters
    ----------
    qv : QuantumVariable
        A QuantumVariable of N qubits.
    n : int
        The index n.
    angles : list
        A list of n-1 angles. The default is None.

    """
    x(qv[0])
    for i in range(1,n):
        if angles == None:
            phi = 2*np.arcsin(1/np.sqrt(n+1-i)) # Phi for uniform superposition
        else:
            phi = angles[i-1]
        pswap(phi, qv[i], qv[0])


def conjugator(a,b):
    h(a)
    cx(a, b)

def quasi_swap(a, b):
    with conjugate(conjugator)(a,b):
        h([a,b])

def pswap(phi, a, b):
    with conjugate(conjugator)(a,b):
        ry(-phi/2, [a,b])    
    

def make_distinct(qv1, qv2, n):
    """
    This algorithm prepares a state of two QuantumVariables qv1, qv2 that is a superposition of all tensor products of basis states of length N with exactly one "1" in the first n qubits, such that 
    the QuantemVariables qv1, qv2 are never in the same state. 
    For example: qv1: |10>+|01>, qv2: |10> --->>> |10>|01> + |01>|10>. Here, the states |01>|01> and |10>|10> are forbidden.

    Parameters
    ----------
    qv1 : QuantumVariable
        N-qubit state that is a tensor product of a (parametrized) W-state of qubits q[0]...q[n] and the |0> state of qubits q[n+1]...q[N-1]
    qv2 : QuantumVariable
        N-qubit state that is a tensor product of a (parametrized) W-state of qubits q[0]...q[n-1] and |0> state of qubits q[n]...q[N-1]
    n : int
        The index n.

    """
    for i in range(n):
        with control(qv1[i]):
            quasi_swap(qv2[n], qv2[i])

def prepare_pbs_state(graph, root, N, q_array, params=None):
    """
    This algorithm prepares a superposition state of all feasible solutions for a product breakdown structure (PBS) problem.
    We call this state the PBS superposition state. For this, it utilizes the PBS tree structure: A node (part) and all its (direct) predecessors can never be in the same state (site).

    Parameters
    ----------
    graph : networkx.DiGraph
        The directed graph representing the PBS.
    root : int
        The root of the graph.
    N : int
        The number sites.
    params : dict
        A dictionary containing a list of parameters (angles) for each node. The default is None.

    """
    M = graph.number_of_nodes()
    
    assert M == len(q_array)

    if params == None:
        W(q_array[root], N)
    else:
        W(q_array[root], N, params[root])

    add_predecessors(graph, root, N, q_array, params)
    
    return q_array


def add_predecessors(graph, node, N, q_array, params=None):
    """
    A recursive algorithm to add predecessors of a node to the PBS superposition state

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
        A dictionary containing a list of angles for each node. The default is None.

    """
    predecessors = list(graph.predecessors(node))
    
    m = len(predecessors)
    if(N<m+1):
        raise Exception(
                "Insufficient number of sites N"
        )

    # Prepare states for predecessors (partial W-states of decreasing size n)
    for i in range(m):
        if params == None:
            W(q_array[predecessors[i]], N-1-i)
        else:
            W(q_array[predecessors[i]], N-1-i, params[predecessors[i]])

    for i in range(m-2,-1,-1):
        for j in range(i+1,m):
            make_distinct(q_array[predecessors[i]], q_array[predecessors[j]], N-2-i)

    # Create entangled satate of predecessors and node that does satisfy constraints
    for j in range(m):
        make_distinct(q_array[node],q_array[predecessors[j]],N-1)

    # Recursivley add predecessors
    for pred in predecessors:
        add_predecessors(graph, pred, N ,q_array, params)


def pbs_state_init(graph, root, N, params=None):
    """
    This algorithm prepares a superposition state of all feasible solutions for a product breakdown structure (PBS) problem.
    We call this state the PBS superposition state. For this, it utilizes the PBS tree structure: A node (part) and all its (direct) predecessors can never be in the same state (site).

    Parameters
    ----------
    graph : networkx.DiGraph
        The directed graph representing the PBS.
    root : int
        The root of the graph.
    N : int
        The number sites.
    params : dict
        A dictionary containing a list of parameters (angles) for each node. The default is None.

    """
    M = graph.number_of_nodes()
    def init_state(q_array):
        assert M == len(q_array)

        if params == None:
            W(q_array[root], N)
        else:
            W(q_array[root], N, params[root])

        add_predecessors(graph, root, N, q_array, params)
        
    
    return init_state