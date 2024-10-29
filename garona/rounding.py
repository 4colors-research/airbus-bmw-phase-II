'''
This file is part of Garona. 
(c) 4colors Research Ltd, 2024

This file contains the implementation of the rounding routines.
'''

import numpy as np
import math, random, itertools
from statistics import mean 
import numpy as np


def identify_non_integer_variables(vars, tolerance=1e-6):
    """
    This function identifies variables that are not close to integers within a given tolerance.
    """

    # Dictionary to store non-integer variables and their values
    non_integer_vars = []
    
    # Iterate over all variables in the problem
    for var in vars:
        var_value = vars[var]
        
        # Check if the variable is close to an integer
        if abs(var_value - round(var_value)) > tolerance:
            non_integer_vars.append(var)
    
    return non_integer_vars


def select_non_integral_vars_to_round(model, n_vars_to_round, delta):
    '''
    Selects the non-integral variables to be rounded.
    '''    

    obj_function_coefficients, nonzero_obj_function_coefficients = model.obj_function_to_dict()

    solution_dict = model.solution_to_dict()                              
    non_integral_vars = identify_non_integer_variables(solution_dict)
    filtered_non_integral_vars = [var for var in non_integral_vars if 'INTRA' not in var]                                                           # remove the variables that correspond to intra transports         
    filtered_non_integral_vars = [v for v in filtered_non_integral_vars if (v[0]=='T' and v in nonzero_obj_function_coefficients) or v[0]=='t' or v[0]=='c']     # select only T (with non-zero coefficnet) or c variables
    
    n_vars_to_round = min(n_vars_to_round, len(filtered_non_integral_vars))
    if n_vars_to_round == 0:
        return [], []
    vars_to_round = random.sample(filtered_non_integral_vars, n_vars_to_round)

    print('There are', len(non_integral_vars), 'non-integral variables and ', len(filtered_non_integral_vars), 'filtered non-integral variables.')       
    print('Variables selected for rounding :', vars_to_round)
    
    vars_to_round_index_by_name = {}                                                    # prepare a dictionary to get the index on vars_to_round list by variable name
    for i, var in enumerate(vars_to_round):
        vars_to_round_index_by_name[var] = i

    c = [obj_function_coefficients.get(var, 0) for var in vars_to_round]                # the objective function coefficients of the variables to round  

    non_integral_assignment = []                                                        # the values of the variables to round      
    for var in vars_to_round:
        non_integral_assignment.append(solution_dict[var])
    non_integral_assignment_floor = [math.floor(x) for x in non_integral_assignment]    # and the values of the variables rounded down
        
    new_constraints, new_constants = model.sub_system_constraints(vars_to_round, vars_to_round_index_by_name, solution_dict)

    base_values = []    
    for i, constraint in enumerate(new_constraints):                    
        base_values.append(np.dot(constraint, non_integral_assignment_floor) + new_constants[i])

    ranges = []
    for j in range(len(non_integral_assignment_floor)):
        ranges.append(range(max(non_integral_assignment_floor[j]-delta, 0)-non_integral_assignment_floor[j], delta))
        
    cartesian_product = list(itertools.product(*ranges))

    assignemnts = [list(i) for i in list(cartesian_product)]    
    obj_values = [np.dot(c, x) for x in assignemnts]

    constraint_violations = []
    for i, constraint in enumerate(new_constraints):        
        constraint_violations.append([np.max(np.dot(constraint, x) + new_constants[i] - base_values[i], 0) for x in assignemnts])        
    constraint_violations = list(map(list, zip(*constraint_violations)))
        
    best_solution = None
    first = None
    for i, value in enumerate(obj_values):        
        pen = value + 1e7*max(constraint_violations[i]) + 1e10*mean(constraint_violations[i])
        if first == None:
            first = pen        
        if pen <= first:    
            
            first = pen 
            best_solution = assignemnts[i]        

    # if len(new_constants) < 8:
    #     quantum.quantum_local_search(np.array(new_constraints), new_constants, c, 1000)
    # quantum.classical_local_search(np.array(new_constraints), new_constants, c)
    best_assignment = [x+y for x,y in zip(best_solution, non_integral_assignment_floor)]
    
    return vars_to_round, best_assignment
