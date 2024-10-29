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
from collections import defaultdict
import pulp
import lightgbm as lgb
from rounding import identify_non_integer_variables
from prediction_model import PredictionModel
from perturbed_model import PerturbedModel

from variable_selector import *

def softmax(l):
    s = sum([np.exp(l_i) for l_i in l])
    return [np.exp(l_i)/s for l_i in l]

def select_variables_to_round(model, feature_map, pred_model, iter, n_vars_to_round, random_samples, select_max):

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

    # choose next_var based on reward and model
    if pred_model is not None:
        intersection_size = len(set(filtered_non_integral_vars).intersection(pred_model.all_data_vars))
        if intersection_size < 0.05*(len(filtered_non_integral_vars)):
            vars_to_round = random.sample(filtered_non_integral_vars, n_vars_to_round)
        else:
            df, vars_tuples = pred_model.simulate_X_inference(feature_map, solution_dict, filtered_non_integral_vars, \
                                                    n_vars_to_round, seed=iter, random_samples=random_samples)
            if len(vars_tuples) == 0:
                vars_to_round = random.sample(filtered_non_integral_vars, n_vars_to_round)
            else:
                predictions = pred_model.predict(df) 
                print('predictions: ', len(vars_tuples), len(predictions), min(predictions), np.mean(predictions), max(predictions))
                if select_max:
                    max_idx = predictions.argmax()
                else:
                    np.random.seed(42)
                    max_idx = int(np.random.choice(list(range(len(vars_tuples))), size=1, p=softmax(predictions))[0])
                vars_to_round = vars_tuples[max_idx]
                # print('selected vars to round', vars_to_round)
    else:
        vars_to_round = random.sample(filtered_non_integral_vars, n_vars_to_round)

    best_assignment, base_int_assignment = local_search(vars_to_round, 
                                   delta=4, 
                                   obj_function_coefficients=obj_function_coefficients, 
                                   solution_dict=solution_dict,
                                   sub_system_constraints=model.sub_system_constraints) 
    
    return vars_to_round, best_assignment, base_int_assignment


def local_search(vars_to_round, delta, obj_function_coefficients, solution_dict, sub_system_constraints):
    vars_to_round_index_by_name = {}  # prepare a dictionary to get the index on vars_to_round list by variable name
    for i, var in enumerate(vars_to_round):
        vars_to_round_index_by_name[var] = i

    # the objective function coefficients of the variables to round
    c = [obj_function_coefficients.get(var, 0) for var in vars_to_round]   

    non_integral_assignment = []         # the values of the variables to round      
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


def filter_nnz_integral_vars(var_freqs, non_zero_integral_vars, non_zero_integral_vars_values, threshold):
    filtered_vars, filtered_vals = [], []
    for var, val  in zip(non_zero_integral_vars, non_zero_integral_vars_values):
        if var_freqs[var] <= threshold:
            filtered_vars.append(var)
            filtered_vals.append(val)
    return filtered_vars, filtered_vals

__version_number__ = "0.5"

path_to_data_files = '../data/'
    
if __name__ == "__main__":    
    print()
    print()
    print('+------------------------------------------------------------------+')
    print(f'|       Garona. Version {__version_number__}                                        |')
    print('|       Airbus x BMW Quantum Optimization Competition Solver       |')
    print('|       (c) 2024, 4colors Research Ltd. All rights reserved.       |')
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
          
    # by setting the sampling threshold to 1, we use the original distribution
    model_lp = PerturbedModel(instance, vars_categories, seed=1, sample_fal_threshold=1)
    
    model_nr = -1 # this is the number of the dataset generated by the RL model after model_nr epochs. If set to -1, we use the pickled model
    pred_model = PredictionModel(features=[], labels=[], model_nr=model_nr, constraint_map=model_lp.constraint_groups) 

    for random_samples in [1, 10, 20, 30, 40]:
        iter = 0
        select_max = True
        scores = []
        fixed_variables = []
        fixed_variables_assignments = []
        run_feasibility_pump = False

        min_non_integrality = None

        print('Starting relax-and-fix...')
        while True and iter < 40:
            print()
            iter += 1
            print('-'*90)
            print(f'samples={random_samples}, iter={iter}, max={select_max}')
            t0=time.time()
            model_lp = PerturbedModel(instance, vars_categories, seed=1, sample_fal_threshold=1) 
            t1=time.time()
            print(f'Copying the model. Elapsed time : {t1-t0}')
            
            if run_feasibility_pump:            
                print('Running the feasibility pump to fix non-feasibilities.')
                t0=time.time()
                variables_by_name = model_lp.get_variables_by_name()
                print('-'*90)
                model_lp = repair.feasibility_pump(model_lp, [variables_by_name[var] for var in fixed_variables], fixed_variables_assignments)
                print('-'*90)
                fixed_variables = []
                fixed_variables_assignments = []
                run_feasibility_pump = False
                t1=time.time()
                print(f'Feasibility pump finished. Elapsed time : {t1-t0:.2f}')
            else:            
                print('Running the LP relaxation. # of integral fixed variables :', len(fixed_variables))
                model_lp.fix_variable_values(fixed_variables, fixed_variables_assignments)
                #print('-'*90)
                model_lp.solve()
                #print('-'*90)

            if model_lp.status == pulp.LpStatusInfeasible:
                run_feasibility_pump = True
            else:
                if model_lp.status == pulp.LpStatusOptimal:                                
                    integral_vars, integral_vars_values = model_lp.get_integral_vars(instance.numerical_tolerance)
                    total_variable_count, integer_variable_count, non_integer_variable_count, none_variable_count, non_int_map, nnz_map = \
                    model_lp.integrality_measure(instance.numerical_tolerance)
                    feature = {'non_int_map': non_int_map, 'nnz_map': nnz_map, 'FAL_distribution': model_lp.FAL_distribution, 'iter': iter}              
                    
                    if min_non_integrality == None or min_non_integrality > non_integer_variable_count:
                        min_non_integrality = non_integer_variable_count

                    print(f'Non-integrality : {non_integer_variable_count}, minimum non-integrality : {min_non_integrality}')
                    print()

                    scores.append((non_integer_variable_count, min_non_integrality))
                    
                    if non_integer_variable_count < 10:
                        print('----------------  MIP  ----------------')
                        vars_categories = {
                            'x' : 'Binary',
                            'y' : 'Integer',        
                            't' : 'Continous',        
                            'c' : 'Integer',        
                            'T' : 'Integer',        
                            'supplier_site_workshare': 'Continous',
                            'supplier_workshare': 'Continous',
                            'site_workshare': 'Continous',
                        }
                        model = model.Model(instance, vars_categories)
                        model.fix_variable_values(fixed_variables, fixed_variables_assignments)
                        model.solve(True)
                        model.integral_vars_stats()
                        model.print_production_stats()
                        model.print_transportation_stats()
                        total_variable_count, integer_variable_count, non_integer_variable_count, none_variable_count, non_int_map, _ = \
                            model.integrality_measure(instance.numerical_tolerance)
                        break
                    
                    print('Rounding variables to fix non-integralities.')
                    t0=time.time()
                    vars_to_round, best_assignment, base_int_assignment = \
                    select_variables_to_round(model_lp, feature_map=feature, pred_model=pred_model, 
                                            iter=iter, n_vars_to_round=4, random_samples=random_samples, select_max=select_max)
                    t1=time.time()
                    print('Rounding finished. Elapsed time :', t1-t0)                
                    fixed_variables.extend(vars_to_round)
                    fixed_variables_assignments.extend(best_assignment)                

                    model_lp.fix_variable_values(fixed_variables, fixed_variables_assignments)                
                    
                    total_variable_count, integer_variable_count, non_integer_variable_count, none_variable_count, non_int_map, _ = \
                        model_lp.integrality_measure(instance.numerical_tolerance)         
                else:
                    break
        f = open(f"./../data/training_data/scores_{random_samples}_{select_max}_{model_nr}.txt", 'w')
        # for score in scores:
        #     f.write(str(score[0]) + ':' + str(score[1]) + '\n') 
        # f.close()