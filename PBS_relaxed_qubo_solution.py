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
from scipy.optimize import minimize
from qrisp import *
from qrisp.qaoa import *

def objective_function(x, cost_matrix, phi, psi, lambdas):

    x = x.reshape((N, M))  # Reshape x with N as parts and M as sites

    # Calculate the original objective value C
    C = 0
    for r, s in phi:
        for i in range(M):
            for j in range(M):
                if i != j:
                    C += cost_matrix[i, j, r] * x[r, i] * x[s, j]
    
    # Penalty term C1 enforces one and only one assignment of the site per part
    C1 = sum((np.sum(x, axis=1) - 1)**2)
    
    # Penalty term C2 ensures origin and destination for each transport are different
    C2 = sum(np.sum(x[np.array(phi)[:, 0], :] * x[np.array(phi)[:, 1], :], axis=1))
    
    # Penalty term C3 ensures the origins of 2 sub-parts of a common part are different
    C3 = sum(np.sum(x[np.array(psi)[:, 0], :] * x[np.array(psi)[:, 1], :], axis=1))
    
    # Combine the objective and penalties
    objective = C + lambdas[0] * C1 + lambdas[1] * C2 + lambdas[2] * C3
    
    return objective

# Define the sets phi and psi
phi = [(2, 1), (3,1), (4, 1), (5, 2), (6, 2), (7, 2), (8, 3), (9, 4), (10, 4)]  # Example set phi
psi = [(2, 3), (2, 4), (3, 4), (5, 6), (5, 7), (6, 7), (9, 10)]  # Example set psi
phi = [(i-1, j-1) for i, j in phi]
psi = [(i-1, j-1) for i, j in psi]

N = 10  # Number of parts
M = 7   # Number of sites

# Define the table of cost_matrix c_{ij}^a
cost_matrix_data = [
    [2, 1, 2, 1.64], [4, 1, 2, 8.06], [6, 1, 2, 7.31], [8, 1, 2, 2.73], [10, 1, 2, 4.9],
    [2, 1, 3, 1.05], [4, 1, 3, 5.14], [6, 1, 3, 4.66], [8, 1, 3, 1.74], [10, 1, 3, 3.12],
    [2, 1, 4, 1.09], [4, 1, 4, 5.35], [6, 1, 4, 4.85], [8, 1, 4, 1.81], [10, 1, 4, 3.25],
    [2, 1, 5, 1.43], [4, 1, 5, 7.03], [6, 1, 5, 6.38], [8, 1, 5, 2.39], [10, 1, 5, 4.27],
    [2, 1, 6, 0.91], [4, 1, 6, 4.47], [6, 1, 6, 4.06], [8, 1, 6, 1.52], [10, 1, 6, 2.72],
    [2, 1, 7, 1.7], [4, 1, 7, 8.32], [6, 1, 7, 7.55], [8, 1, 7, 2.82], [10, 1, 7, 5.05],
    [2, 2, 3, 0.59], [4, 2, 3, 2.88], [6, 2, 3, 2.61], [8, 2, 3, 0.98], [10, 2, 3, 1.75],
    [2, 2, 4, 0.37], [4, 2, 4, 1.8], [6, 2, 4, 1.63], [8, 2, 4, 0.61], [10, 2, 4, 1.09],
    [2, 2, 5, 1.12], [4, 2, 5, 5.49], [6, 2, 5, 4.98], [8, 2, 5, 1.86], [10, 2, 5, 3.34],
    [2, 2, 6, 0.24], [4, 2, 6, 1.18], [6, 2, 6, 1.07], [8, 2, 6, 0.4], [10, 2, 6, 0.72],
    [2, 2, 7, 1.79], [4, 2, 7, 8.76], [6, 2, 7, 7.95], [8, 2, 7, 2.97], [10, 2, 7, 5.32],
    [2, 3, 4, 0.93], [4, 3, 4, 4.58], [6, 3, 4, 4.15], [8, 3, 4, 1.55], [10, 3, 4, 2.78],
    [2, 3, 5, 1.02], [4, 3, 5, 4.98], [6, 3, 5, 4.52], [8, 3, 5, 1.69], [10, 3, 5, 3.02],
    [2, 3, 6, 1.35], [4, 3, 6, 6.62], [6, 3, 6, 6.01], [8, 3, 6, 2.25], [10, 3, 6, 4.02],
    [2, 3, 7, 1.65], [4, 3, 7, 8.07], [6, 3, 7, 7.32], [8, 3, 7, 2.74], [10, 3, 7, 4.9],
    [2, 4, 5, 1.65], [4, 4, 5, 8.12], [6, 4, 5, 7.36], [8, 4, 5, 2.75], [10, 4, 5, 4.93],
    [2, 4, 6, 0.19], [4, 4, 6, 0.94], [6, 4, 6, 0.85], [8, 4, 6, 0.32], [10, 4, 6, 0.57],
    [2, 4, 7, 1.52], [4, 4, 7, 7.45], [6, 4, 7, 6.76], [8, 4, 7, 2.53], [10, 4, 7, 4.52],
    [2, 5, 6, 1.04], [4, 5, 6, 5.1], [6, 5, 6, 4.63], [8, 5, 6, 1.73], [10, 5, 6, 3.1],
    [2, 5, 7, 0.7], [4, 5, 7, 3.41], [6, 5, 7, 3.1], [8, 5, 7, 1.16], [10, 5, 7, 2.07],
    [2, 6, 7, 1.57], [4, 6, 7, 7.68], [6, 6, 7, 6.97], [8, 6, 7, 2.61], [10, 6, 7, 4.67],
    [3, 1, 2, 5.56], [5, 1, 2, 7.66], [7, 1, 2, 1.0], [9, 1, 2, 1.49],
    [3, 1, 3, 3.54], [5, 1, 3, 4.88], [7, 1, 3, 0.64], [9, 1, 3, 0.95],
    [3, 1, 4, 3.68], [5, 1, 4, 5.08], [7, 1, 4, 0.66], [9, 1, 4, 0.99],
    [3, 1, 5, 4.85], [5, 1, 5, 6.68], [7, 1, 5, 0.87], [9, 1, 5, 1.3],
    [3, 1, 6, 3.08], [5, 1, 6, 4.25], [7, 1, 6, 0.56], [9, 1, 6, 0.83],
    [3, 1, 7, 5.73], [5, 1, 7, 7.9], [7, 1, 7, 1.03], [9, 1, 7, 1.54],
    [3, 2, 3, 1.98], [5, 2, 3, 2.73], [7, 2, 3, 0.36], [9, 2, 3, 0.53],
    [3, 2, 4, 1.24], [5, 2, 4, 1.71], [7, 2, 4, 0.22], [9, 2, 4, 0.33],
    [3, 2, 5, 3.79], [5, 2, 5, 5.22], [7, 2, 5, 0.68], [9, 2, 5, 1.02],
    [3, 2, 6, 0.82], [5, 2, 6, 1.12], [7, 2, 6, 0.15], [9, 2, 6, 0.22],
    [3, 2, 7, 6.04], [5, 2, 7, 8.32], [7, 2, 7, 1.09], [9, 2, 7, 1.62],
    [3, 3, 4, 3.15], [5, 3, 4, 4.35], [7, 3, 4, 0.57], [9, 3, 4, 0.85],
    [3, 3, 5, 3.43], [5, 3, 5, 4.73], [7, 3, 5, 0.62], [9, 3, 5, 0.92],
    [3, 3, 6, 4.56], [5, 3, 6, 6.29], [7, 3, 6, 0.82], [9, 3, 6, 1.23],
    [3, 3, 7, 5.56], [5, 3, 7, 7.66], [7, 3, 7, 1.0], [9, 3, 7, 1.5],
    [3, 4, 5, 5.59], [5, 4, 5, 7.71], [7, 4, 5, 1.01], [9, 4, 5, 1.5],
    [3, 4, 6, 0.64], [5, 4, 6, 0.89], [7, 4, 6, 0.12], [9, 4, 6, 0.17],
    [3, 4, 7, 5.13], [5, 4, 7, 7.07], [7, 4, 7, 0.93], [9, 4, 7, 1.38],
    [3, 5, 6, 3.51], [5, 5, 6, 4.84], [7, 5, 6, 0.63], [9, 5, 6, 0.95],
    [3, 5, 7, 2.35], [5, 5, 7, 3.24], [7, 5, 7, 0.42], [9, 5, 7, 0.63],
    [3, 6, 7, 5.3], [5, 6, 7, 7.3], [7, 6, 7, 0.96], [9, 6, 7, 1.42]
]
# Check shapes
# Convert the data into a numpy array
cost_matrix_data = np.array(cost_matrix_data)

# Extracting the maximum indices to initialize the size of the cost matrix
max_a, max_i, max_j, _ = cost_matrix_data[-1]

# Convert the maximum indices to integers and add 1 for zero-based indexing
max_a = int(max_a) + 1
max_i = int(max_i) + 1
max_j = int(max_j) + 1

cost_matrix = np.zeros((M, M, N))  # Initialize the cost matrix with zeros

# Iterate over each row in cost_matrix_data
for row in cost_matrix_data:
    a, i, j, value = row  # Extract values a, i, j, and the cost value
    # Ensure indices are within bounds
    if 1 <= i <= M and 1 <= j <= M and 1 <= a <= N:
        cost_matrix[int(i) - 1, int(j) - 1, int(a) - 1] = value  # Populate the corresponding position in cost_matrix

lambdas = [1, 1, 1]  # Penalty weights

# Define the initial guess for the continuous variables (relaxed binary variables)
x0 = np.random.rand(M, N)

# Define the bounds for the relaxed variables (0 <= x_{ri} <= 1)
bounds = [(0, 1) for _ in range(M * N)]

# Minimize the objective function using SLSQP with bounds
result = minimize(objective_function, x0.flatten(), args=(cost_matrix, phi, psi, lambdas), method='SLSQP', bounds=bounds, options={'maxiter': 1000, 'disp': True})

# Reshape the result to the original shape
optimal_x = result.x.reshape((M, N))

print("Optimal solution:\n", optimal_x)
print("Optimal objective value:", result.fun)

# Flatten the optimal solution
optimal_x_flat = optimal_x.flatten()

# Create a list of variables
variables = ['x_{}_{}'.format(i, j) for i in range(M) for j in range(N)]

# Create a dictionary mapping variables to their optimal values
solution = dict(zip(variables, optimal_x_flat))

# Now, solution is a dictionary where the keys are the variable names and the values are the optimal values
for var, value in solution.items():
    print(f"{var} = {value}")

def warm_start_is(qarg, optimal_x_flat, eps=0.25):
    for i in range(len(optimal_x_flat)):
        c = float(optimal_x_flat[i])
        if c < eps:
            c = eps
        if c > 1 - eps:
            c = 1 - eps
        theta = 2 * np.arcsin(np.sqrt(c))
        ry(theta, qarg[i])
    
def warm_start_mixer(qarg, optimal_x_flat, beta, eps=0.25):    
    for i in range(len(optimal_x_flat)):
        c = float(optimal_x_flat[i])
        if c < eps:
            c = eps
        if c > 1 - eps:
            c = 1 - eps
        theta = 2 * np.arcsin(np.sqrt(c))
        ry(-theta, qarg[i])
        rz(-2 * beta, qarg[i])
        ry(theta, qarg[i])
    return qarg

# Size of the quantum array 'qarg' should be M*N
qarg = QuantumArray(qtype=QuantumVariable(1), shape=(M * N))

# Flatten the optimal solution to match the quantum array size
optimal_solution_flat = optimal_x.flatten()  # Flatten the 2D array to 1D

# Apply the warm start initialization to 'qarg' using the flattened optimal solution
warm_start_is(qarg, optimal_solution_flat)

# Choose the desired QAOA depth
depth = 3

# Create the QAOA problem instance with the appropriate cost operator and warm start mixer
pbs_instance_with_ws = QAOAProblem(create_pbs_cost_operator(cost_matrix, phi, psi), warm_start_mixer, create_pbs_cl_cost_function(cost_matrix, phi, psi))

# Set the warm start initial state function
pbs_instance_with_ws.set_init_function(lambda qarg: warm_start_is(qarg, optimal_solution_flat))

# Run the QAOA algorithm
res_with_ws = pbs_instance_with_ws.run(qarg, depth, max_iter=100)