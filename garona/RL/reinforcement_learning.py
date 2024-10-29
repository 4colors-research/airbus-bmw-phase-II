'''
This file is part of Garona. 
(c) 4colors Research Ltd, 2024

This file contains the implementation of the reinforcement learning routine.
'''


from colorama import init, Back,  Fore, Style
import instance, model, repair, rounding, utils
import pickle, sys, itertools, math, random, copy, time
from statistics import mean
import numpy as np
from copy import copy, deepcopy
from collections import defaultdict, Counter
from rounding import identify_non_integer_variables
from prediction_model import PredictionModel
from perturbed_model import PerturbedModel
import pulp
import pandas as pd

path_to_data_files = '../data/'

def select_variables_to_round(model, feature_map, pred_model, epoch, iter, n_vars_to_round):

    # select only from a small subset of nonint variables V, e.g., the 600 observed earlier
    # store configurations of these variables as states.
    # select variables for fixing based on an ML model M (agent) that has as input:
    #   1. the non-int values of the vars in V (env)
    #   2. the random allocation of the FALs (env)
    #   3. the constraints for the vars in V (env)
    # the model M predicts the probability for vars rounded down to yield an improvement based on historical data
    # reward the quality of solution found by the LP solver, i.e., improved over past epoch, and feed it as an example to M
    # retrain M when enough new examples are present

    obj_function_coefficients, nonzero_obj_function_coefficients = model.obj_function_to_dict()

    solution_dict = model.solution_to_dict()                              
    non_integral_vars = identify_non_integer_variables(solution_dict)
     # remove the variables that correspond to intra transports
    filtered_non_integral_vars = [var for var in non_integral_vars if 'INTRA' not in var]      
    # select only T (with non-zero coefficnet) or c variables              
    filtered_non_integral_vars = \
        [v for v in filtered_non_integral_vars if (v[0]=='T' and v in nonzero_obj_function_coefficients) or v[0]=='t' or v[0]=='c']    
    
    print('# non-integer variables to choose from', len(filtered_non_integral_vars))
    n_vars_to_round = min(n_vars_to_round, len(filtered_non_integral_vars))
    if n_vars_to_round == 0:
        return [], [], []

    # choose_next_var based on reward and model
    if pred_model is not None:
        intersection_size = len(set(filtered_non_integral_vars).intersection(pred_model.all_data_vars))
        print('intersection size', intersection_size)

        # use the model only when enough of the candidates for rounding have been observed in previous iterations
        if intersection_size < 0.5*(len(filtered_non_integral_vars)): 
            vars_to_round = random.sample(filtered_non_integral_vars, n_vars_to_round)
        else:
            df, vars_tuples = pred_model.simulate_X(feature_map, solution_dict, filtered_non_integral_vars, \
                                                    n_vars_to_round, seed=iter, random_samples=100)
            predictions = pred_model.predict(df) 
            print('predictions: ', len(vars_tuples), len(predictions), min(predictions), np.mean(predictions), max(predictions))
            max_idx = predictions.argmax()
            vars_to_round = vars_tuples[max_idx]
            print('selected vars to round', vars_to_round)
    else:
        vars_to_round = random.sample(filtered_non_integral_vars, n_vars_to_round)

    best_assignment, base_int_assignment = local_search(vars_to_round, 
                                   delta=4, 
                                   obj_function_coefficients=obj_function_coefficients, 
                                   solution_dict=solution_dict,
                                   sub_system_constraints=model.sub_system_constraints) 
    
    return vars_to_round, best_assignment, base_int_assignment


# this is the part that is optimized using quantum computing
def local_search(vars_to_round, delta, obj_function_coefficients, solution_dict, sub_system_constraints):
    vars_to_round_index_by_name = {}  # prepare a dictionary to get the index on vars_to_round list by variable name
    for i, var in enumerate(vars_to_round):
        vars_to_round_index_by_name[var] = i

    # the objective function coefficients of the variables to round
    c = [obj_function_coefficients.get(var, 0) for var in vars_to_round]   

    non_integral_assignment = []            # the values of the variables to round      
    for var in vars_to_round:
        non_integral_assignment.append(solution_dict[var])
    # and the values of the variables rounded down
    non_integral_assignment_floor = [math.floor(x) for x in non_integral_assignment]  
        
    new_constraints, new_constants = sub_system_constraints(vars_to_round, vars_to_round_index_by_name, solution_dict)
            
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
    best_assignment = [x+y for x,y in zip(best_solution, non_integral_assignment_floor)]
    return best_assignment, non_integral_assignment_floor

    
if __name__ == "__main__":    
    print()
    print()
    print('+------------------------------------------------------------------+')
    print('|                      Reinforcement earning                       |')
    print('+------------------------------------------------------------------+')
    
    
    vars_categories = {
        'x' : 'Continous',
        'y' : 'Continous',        
        't' : 'Continous',
        'c' : 'Continous',
        'T' : 'Continous',
        'supplier_site_workshare': 'Continous',
        'supplier_workshare': 'Continous',
        'site_workshare': 'Continous',
    }


    instance_cache_filename = 'instance_cached.pkl'
    try:
        with open(instance_cache_filename, 'rb') as f:
            print('Cached instance found in the path. Reading from disk.')
            instance = pickle.load(f)                                
            instance._Instance__load_additional_instance_input()
            instance._Instance__init_production_sites()
            instance._Instance__load_params()
    except FileNotFoundError:        
        print('No cached instance. Building an instance.')
        with open(instance_cache_filename, 'wb') as f:
            instance = instance.Instance(path_to_data_files)
            pickle.dump(instance, f)

    for site in instance.production_sites:
        is_first = True
        for part in instance.parts_by_site[site]:
            if part == instance.assembled_aircraft:
                continue
            if instance.parent_by_part[part] in instance.parts_by_site[site]:
                continue
            destinations_for_part = [j for (i, j, s) in instance.cargo_by_route_and_part if i==site and s==part]
            if site in destinations_for_part:
                destinations_for_part.remove(site)
            if len(destinations_for_part) == 0:
                if is_first:             
                    print()       
                    print(f'{instance.site_name[site]}  ({instance.site_id[site]})')
                    is_first = False
                print(f'\t {instance.part_name[part]} ({instance.part_id[part]})')
        
    pred_model = None 

    print('Starting relax-and-fix...')

    features = []
    labels = []
    n_vars_to_round = 4
    def_zero = 1e-4

    for epoch in range(1000):
        nr_iters = 20
        iter = 0
        feasible = True
        fixed_variables = []
        fixed_variables_assignments = []
        run_feasibility_pump = False
        min_non_integrality = 100000

        while iter < nr_iters and feasible and min_non_integrality > 10:
            iter += 1
            print(f'\nepoch: {epoch}, iter: {iter}')

            # a low sampling threshold leads to a more sparse FAL distribution, i.e., using only a subset of the FALs
            model_lp = PerturbedModel(instance, vars_categories, seed=epoch+1, sample_fal_threshold=0.99)  
            print(model_lp.FAL_distribution)
            var_names = [var.name for var in model_lp.prob.variables()]
            model_lp.fix_variable_values(fixed_variables, fixed_variables_assignments)
            model_lp.solve()

            feasible = (model_lp.status != pulp.LpStatusInfeasible)

            integral_vars, integral_vars_values = model_lp.get_integral_vars(instance.numerical_tolerance)

            x = model_lp.solution_to_vector(var_names)                
            total_variable_count, integer_variable_count, non_integer_variable_count, none_variable_count, non_int_map, nnz_map = \
                model_lp.integrality_measure(instance.numerical_tolerance)
            feature = {'non_int_map': non_int_map, 'nnz_map': nnz_map, 'FAL_distribution': model_lp.FAL_distribution, 'iter': iter}
            
            if not feasible or non_integer_variable_count <= 10:
                print(f'Breaking. feasible: {feasible}, non-int-cnt: {non_integer_variable_count}')
                break
            vars_to_round, best_assignment, base_int_assignment = \
                select_variables_to_round(model_lp, feature_map=feature, pred_model=pred_model, \
                                          epoch=epoch, iter=iter, n_vars_to_round=n_vars_to_round)
            #assert len(vars_to_round) == n_vars_to_round
            print('Rounding finished.')  
                    
            fixed_variables.extend(vars_to_round)
            fixed_variables_assignments.extend(best_assignment)                

            model_lp.fix_variable_values(fixed_variables, fixed_variables_assignments)                
            x = model_lp.solution_to_vector(var_names)

            integral_vars, integral_vars_values = model_lp.get_integral_vars(instance.numerical_tolerance)
            x = model_lp.solution_to_vector(var_names)                
            total_variable_count, integer_variable_count, non_integer_variable_count, none_variable_count, non_int_map, nnz_map = \
                model_lp.integrality_measure(instance.numerical_tolerance)
            
            if min_non_integrality == None or min_non_integrality > non_integer_variable_count:
                min_non_integrality = non_integer_variable_count

            print('Appending selected: ', {str(var): floored for var, floored in zip(vars_to_round, base_int_assignment)})
            print('iter: ', iter)
            features.append({'non_int_map': non_int_map, 
                        'selected_vars': {str(var): floored for var, floored in zip(vars_to_round, base_int_assignment)},
                        'nnz_map': nnz_map,
                        'FAL_distribution': model_lp.FAL_distribution,
                        'iter': iter
                            })
            labels.append((max(def_zero, non_integer_variable_count), min_non_integrality))

            print(f'Non-integrality : {non_integer_variable_count}, minimum non-integrality : {min_non_integrality}') 
            print()

        # generate a pandas dataframe from the features collected so far and train a prediction model
        pred_model = PredictionModel(features, labels, model_lp.constraint_groups)  
        pred_model.train()
        df = deepcopy(pred_model.X)
        df["y"] = pred_model.y

        # writing current dataset to file 
        df.to_csv(f"./../data/training_data/dense_X_epoch_{epoch}.csv", index=False)