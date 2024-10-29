'''
This file is part of Garona. 
(c) 4colors Research Ltd, 2024

This file contains the core code of Garona.
'''
import instance, model, repair, population, rounding, utils

import copy, pickle, pulp, time
import numpy as np
from colorama import Back,  Fore, Style

__version_number__ = "1.0"

path_to_data_files = '../data/2/'
    
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

    solution_pool = population.SolutionPool()

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
       

    main_model = model.Model(instance, vars_categories)    

    variables = list(main_model.prob.variables())
    A, b, c, var_names = main_model.to_matrix()
    A_normalized = copy.deepcopy(A)
    b_normalized = copy.deepcopy(b)
    utils.normalize_matrix_and_vector(A_normalized, b_normalized)
        
    fixed_variables = []
    fixed_variables_assignments = []
    run_feasibility_pump = False

    min_non_integrality = None

    print('Starting relax-and-fix...', flush=True)

    while True:
        t0=time.time()
        model_lp = model.Model(instance, vars_categories) 
        t1=time.time()
        # print(f'Copying the model. Elapsed time : {t1-t0}', flush=True)
        
        if run_feasibility_pump:            
            print('Running the feasibility pump to fix non-feasibilities.', flush=True)
            t0=time.time()
            variables_by_name = model_lp.get_variables_by_name()
            print('-'*90)
            model_lp = repair.feasibility_pump(model_lp, [variables_by_name[var] for var in fixed_variables], fixed_variables_assignments)
            print('-'*90)
            fixed_variables = []
            fixed_variables_assignments = []
            run_feasibility_pump = False
            t1=time.time()
            print(f'Feasibility pump finished. Elapsed time : {t1-t0:.2f}', flush=True)
        else:            
            print('Running the LP relaxation. # of integral fixed variables :', len(fixed_variables), flush=True)
            model_lp.fix_variable_values(fixed_variables, fixed_variables_assignments)
            print('-'*90)
            model_lp.solve(True)
            print('-'*90)

        solution_pool.add(model_lp.solution_to_dict())                

        if model_lp.status == pulp.LpStatusInfeasible:
            run_feasibility_pump = True
        else:
            if model_lp.status == pulp.LpStatusOptimal:         

                integral_vars, integral_vars_values = model_lp.get_integral_vars(instance.numerical_tolerance)
                x = model_lp.solution_to_vector(var_names)                
                total_variable_count, integer_variable_count, non_integer_variable_count, none_variable_count = model_lp.integrality_measure(instance.numerical_tolerance)
                
                if min_non_integrality == None or min_non_integrality > non_integer_variable_count:
                    min_non_integrality = non_integer_variable_count

                print(f'Non-integrality : {non_integer_variable_count}, minimum non-integrality : {min_non_integrality}', ', infeasibility', np.sum(np.max(A_normalized@x-b_normalized, 0)), ', obj value :', c@x, flush=True)
                print()
                
                if non_integer_variable_count < 20:
                    print('----------------  MIP  ----------------', flush=True)
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
                 
                    solution_dict = main_model.solution_to_dict()         
                    nonzero_variable_count = 0
                    for i in solution_dict:
                        if solution_dict[i] != 0:
                            nonzero_variable_count+=1

                    total_variable_count, integer_variable_count, non_integer_variable_count, none_variable_count = main_model.integrality_measure(instance.numerical_tolerance)    

                    print('total_variable_count',total_variable_count)
                    print('integer_variable_count',integer_variable_count)
                    print('non_integer_variable_count',non_integer_variable_count)
                    print('nonzero_variable_count', nonzero_variable_count)

                    main_model.integral_vars_stats()
                    main_model.print_production_stats()
                    main_model.print_transportation_stats()
                    main_model.print_worshare_table()

                    solution_pool.add(model.solution_to_dict())
                    break
                
                print('Rounding variables to fix non-integralities.', flush=True)
                t0=time.time()
                vars_to_round, best_assignment = rounding.select_non_integral_vars_to_round(model_lp, 4, 2)
                t1=time.time()
                print('Rounding finished. Elapsed time :', t1-t0, flush=True)
                fixed_variables.extend(vars_to_round)
                fixed_variables_assignments.extend(best_assignment)                

                model_lp.fix_variable_values(fixed_variables, fixed_variables_assignments)                
                x = model_lp.solution_to_vector(var_names)
                solution_pool.add(model_lp.solution_to_dict())
                
                total_variable_count, integer_variable_count, non_integer_variable_count, none_variable_count = model_lp.integrality_measure(instance.numerical_tolerance)
                print(f'Non-integrality : {non_integer_variable_count}, infeasibility', np.sum(np.max(A_normalized@x-b_normalized, 0)), ', obj value :', c@x, flush=True)
                print()
                                
            else:
                break


