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
import numpy as np
from sympy import Symbol
from sympy import *

def app_sb_phase_polynomial(input_qf_list, poly, symbol_list=None):
    
    # As the polynomial has only boolean variables,
    # powers can be ignored since x**k = x for x in GF(2)
    poly = filter_pow(poly.expand()).expand()

    if isinstance(input_qf_list, QuantumArray):
        input_qf_list = list(input_qf_list.flatten())

    # Acquire list of symbols corresponding to the variables in input_qf_list present in the polynomial

    if symbol_list is None:
        symbol_list = []
        for qf in input_qf_list:
            temp_var_list = list(sp.symbols(qf.name + "_" + "0:" + str(qf.size)))
            symbol_list += temp_var_list

    n = len(symbol_list)

    # Substitute x_i -> (1-x_i)/2
    repl_dic = {symbol_list[i]: (1-symbol_list[i])/2 for i in range(n)}
    poly = poly.subs(repl_dic).expand()

    #if n != sum([var.size for var in input_qf_list]):
    #    raise Exception(
    #        "Input variables do not the required amount of qubits to encode polynomial"
    #    )

    # The list of qubits contained in the variables of input_var_list
    input_qubits = sum([list(var.reg) for var in input_qf_list], [])
    
    # Acquire monomials in list form
    monomial_list = expr_to_list(poly)

    rz_qubit_list = []
    y_list = []
    
    # Iterate through the monomials
    for monom in monomial_list:
        # Prepare coeff (coefficient of the monomial) and variables (list of variables from symbol_list in the monomial)
        # Note: coeff may also contain symbolic variables
        coeff = float(1)
        variables = []
        for term in monom:
            if isinstance(term, sp.core.symbol.Symbol) and term in symbol_list:
                variables.append(term)
            elif isinstance(term, sp.core.symbol.Symbol):
                coeff = coeff*term
            else:
                coeff = coeff*float(term)
    
        # Append coefficient to y_list
        y_list.append(coeff)
    
        # Prepare the qubits on which the RZ (or RZZ) should be applied
        rz_qubit_numbers = [symbol_list.index(var) for var in variables]
        
        if(len(rz_qubit_numbers)>2):
            raise Exception(
                "Provided polynomial has degree greater than 2"
            )

        rz_qubits = [input_qubits[nr] for nr in rz_qubit_numbers]
    
        rz_qubits = list(set(rz_qubits))
    
        rz_qubits.sort(key=lambda x: x.identifier)
    
        rz_qubit_list.append(rz_qubits)
    
    # Apply RZ, RZZ or gphase gates
    # Iterate through the list of phase gates
    while rz_qubit_list:
        monomial_index = 0
        # Find control qubits and their coefficient
        rz_qubits = rz_qubit_list.pop(monomial_index)
        y = y_list.pop(monomial_index)
    
        # Apply RZ, RZZ or gphase gates
        if len(rz_qubits)==2:
            rzz(-2*y, rz_qubits[0], rz_qubits[1])
        elif len(rz_qubits)==1:
            rz(-2*y, rz_qubits[0])
        else:
            gphase(y, input_qf_list[0][0])