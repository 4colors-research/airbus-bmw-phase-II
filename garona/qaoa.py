'''
This file is part of Garona. 
(c) 4colors Research Ltd, 2024

This file contains the implementation of the QAOA routine.
'''


from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_algorithms import QAOA
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms.optimizers import COBYLA
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Sampler
import copy 
from qiskit_braket_provider import BraketProvider
from qiskit_braket_provider import BraketLocalBackend


qubo_constraint_penalty = 1000


def convert_to_openqaoa(qubo_problem):
    """Convert Qiskit's QUBO to openqaoa QUBO format."""
    # Qiskit's QUBO representation
    quadratic_terms = qubo_problem.objective.quadratic.to_dict()  # Quadratic terms
    linear_terms = qubo_problem.objective.linear.to_dict()        # Linear terms
    constant_term = qubo_problem.objective.constant               # Constant offset in QUBO

    # Prepare QUBO
    qubo_dict = {}

    # Add linear terms
    for i, coeff in linear_terms.items():
        qubo_dict[(int(i), int(i))] = coeff  # Linear terms appear as diagonal elements in the QUBO matrix

    # Add quadratic terms
    for (i, j), coeff in quadratic_terms.items():
        qubo_dict[(int(i), int(j))] = coeff

    # Convert to QUBO object in openqaoa
    openqaoa_qubo = QUBO(terms=qubo_dict)
    
    return openqaoa_qubo


def local_search(weights, b, obj_coeffs, depth=1):
    # Copy the input
    weights = copy.copy(weights)
    b = copy.copy(b)
    obj_coeffs = copy.copy(obj_coeffs)

    # Convert to integers
    M = 1000
    weights = (M*weights).astype(int)
    b = (M*b).astype(int)
    obj_coeffs = (M*obj_coeffs).astype(int)

    problem = QuadraticProgram()
    n_constraints = weights.shape[0]
    n_coeffs = weights.shape[1]

    obj_dict = {}    
    variables = []
    # Add the objective function
    for i in range(n_coeffs):
        variable = 'x'+str(i)
        problem.binary_var(variable)  
        obj_dict[variable] = obj_coeffs[i]
        variables.append(variable)

    problem.minimize(linear=obj_dict)
    
    # Add the constraints
    for j in range(n_constraints):
        constraint_dict = {}
        for i in range(n_coeffs):
            constraint_dict[variables[i]] = weights[j][i]
        problem.linear_constraint(linear=constraint_dict, sense='LE', rhs=-b[j])
        
    # Penalize constraint violations
    converter = QuadraticProgramToQubo(penalty=qubo_constraint_penalty)
    qubo_problem = converter.convert(problem)    

    # Set up the sampler and QAOA with the chosen optimizer
    # Define the sampler using the Qiskit Aer backend (simulator)
    sampler = Sampler()
           
    # Solve the problem using QAOA 
    results = []
    for i in range(10):
        print(i)
        # Define the QAOA algorithm with COBYLA optimizer and the sampler    
        qaoa = QAOA(optimizer=COBYLA(), reps=depth, sampler=sampler)

        # Step 5: Use MinimumEigenOptimizer to run QAOA    
        qaoa_optimizer = MinimumEigenOptimizer(qaoa)        
        result = qaoa_optimizer.solve(qubo_problem)
        results.append([result[v] for v in result.variables_dict if v[0]=='x'])
    
    return results
