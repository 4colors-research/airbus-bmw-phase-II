'''
This file is part of Garona. 
(c) 4colors Research Ltd, 2024

This file contains the code to run the MIP model used by Garona.
'''

from colorama import init, Back,  Fore, Style
import instance, model, repair, rounding, utils
import pickle, sys, itertools, math, random, copy, time
import numpy as np
from statistics import mean
from collections import defaultdict
import pulp

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
    
    
    # vars_categories = {
    #     'x' : 'Continous',
    #     'y' : 'Continous',        
    #     't' : 'Continous',
    #     'c' : 'Continous',
    #     'T' : 'Continous',
    #     'supplier_site_workshare': 'Continous',
    #     'supplier_workshare': 'Continous',
    #     'site_workshare': 'Continous',
    # }


    vars_categories = {
            'x' : 'Binary',
            'y' : 'Integer',
            't' : 'Integer',
            'c' : 'Continous',
            'T' : 'Integer',
            'supplier_site_workshare': 'Continous',
            'supplier_workshare': 'Continous',
            'site_workshare': 'Continous',
        }


    instance_cache_filename = "instance_cached" + path_to_data_files.replace("/", "_").replace(".", "_") + ".pkl"

    try:
        with open(instance_cache_filename, 'rb') as f:
            print('Cached instance found in the path. Reading from disk.')
            instance = pickle.load(f)                                
            instance._Instance__load_files()
            instance._Instance__load_additional_instance_input()
            instance._Instance__init_production_sites()
            instance._Instance__load_params()
            instance._Instance__init_production_sites()
    except FileNotFoundError:        
        print('No cached instance. Building an instance.')
        with open(instance_cache_filename, 'wb') as f:
            instance = instance.Instance(path_to_data_files)
            pickle.dump(instance, f)
    
    main_model = model.Model(instance, vars_categories)        
    main_model.solve(True)

    solution_dict = main_model.solution_to_dict()
    for var in solution_dict:
        if solution_dict[var] != 0:
            print(var, solution_dict[var])

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