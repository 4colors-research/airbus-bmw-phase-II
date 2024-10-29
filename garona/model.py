'''
This file is part of Garona. 
(c) 4colors Research Ltd, 2024

This file contains the MIP model code.
'''

import instance, utils

import copy, pulp, time, math
import numpy as np

from tqdm import tqdm
from statistics import mean, stdev
from colorama import Fore, Back
from collections import defaultdict

from scipy.sparse import csr_matrix


class Model:
    def __init__(self, instance : instance, var_categories):
        self.instance = instance   
        self.var_categories = var_categories
        self.__build_mip_model()    

    def __create_variables(self):
        # x[s][i] is a binary variable that decides whether part s is produced at site i.
        # x's are defined for all parts but only for the sites that produce each part.
        self.x = {s: {i: {} for i in self.instance.production_sites_by_part[s]} for s in self.instance.parts}            
        for s in self.instance.parts:
            for i in self.instance.production_sites_by_part[s]:        
                self.x[s][i] = pulp.LpVariable(f'x_{s}_{i}', lowBound=0, upBound=1, cat=self.var_categories['x'])

        # y[s][i] is an integer variable that decides the quantity of part s produced at site i.
        # y's are defined for all parts but only for the sites that produce each part (like x's).
        self.y = {s: {i: {} for i in self.instance.production_sites_by_part[s]} for s in self.instance.parts}
        for s in self.instance.parts:
            for i in self.instance.production_sites_by_part[s]:
                self.y[s][i] = pulp.LpVariable(f'y_{s}_{i}', lowBound=0, upBound=self.instance.demand, cat=self.var_categories['y'])

        # supplier_site_workshare[k][i] is an integer variable that equals the value of parts producted by supplier k at site i.
        # supplier_site_workshare's are defined for all suppliers and the sites that belong to each supplier.
        self.supplier_site_workshare = {k: {i: {} for i in self.instance.locations_by_supplier[k]} for k in self.instance.suppliers}
        for k in self.instance.suppliers:
            for i in self.instance.locations_by_supplier[k]:
                self.supplier_site_workshare[k][i] = pulp.LpVariable(f'supplier_site_share_{k}_{i}', lowBound=0, cat=self.var_categories['supplier_site_workshare'])

        # supplier_workshare is the sum of all supplier_site_workshare's for a given supplier.
        self.supplier_workshare = pulp.LpVariable.dicts('supplier_workshare', self.instance.suppliers, lowBound=0, cat=self.var_categories['supplier_workshare'])
        self.supplier_workshare_deviation_above = pulp.LpVariable.dicts('supplier_workshare_deviation_above', self.instance.suppliers, lowBound=0)
        self.supplier_workshare_deviation_below = pulp.LpVariable.dicts('supplier_workshare_deviation_below', self.instance.suppliers, lowBound=0)

        # site_workshare[i] is an integer variable that equals the value of parts producted by site i.
        self.site_workshare = pulp.LpVariable.dicts('site_workshare', self.instance.production_sites, lowBound=0, cat=self.var_categories['site_workshare'])
        self.site_workshare_deviation_above = pulp.LpVariable.dicts('site_worksharedeviation_above', self.instance.production_sites, lowBound=0)
        self.site_workshare_deviation_below = pulp.LpVariable.dicts('site_worksharedeviation_below', self.instance.production_sites, lowBound=0)

        # t[s][i][j] is a integer variable that decides the quantity of part s to be sent from site i to j
        self.t = {s: {i: {} for i in self.instance.sites} for s in self.instance.sub_parts}

        self.t_domain = set()
        self.T_domain = set()
        for (i, j, s) in self.instance.cargo_by_route_and_part:            
            if s in self.instance.sub_parts:                
                if ((i in self.instance.warehouse_sites) or (s in self.instance.parts_by_site[i])) and ((j in self.instance.warehouse_sites) or (self.instance.parent_by_part[s] in self.instance.parts_by_site[j])):
                    self.t_domain.add((s, i, j))
                    self.T_domain.add((i, j))

        for (s, i, j) in self.t_domain:            
            self.t[s][i][j] = pulp.LpVariable(f't_{s}_{i}_{j}', lowBound=0, cat=self.var_categories['t'])        

        self.c = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))        
        for (s, i, j) in self.t_domain:
            for m in self.instance.cargo_by_route_and_part[(i, j, s)]['cargoCapacity']:
                self.c[s][i][j][m] = pulp.LpVariable(f'c_{s}_{i}_{j}_{m}', lowBound=0, cat=self.var_categories['c'])

        # T[i][j] is a integer variable that decides the quantity of part s to be sent from site i to j
        self.T = defaultdict(lambda: defaultdict(dict))
        for (i, j) in self.T_domain:     
            for m in self.instance.cargo_by_route[(i, j)]:
                self.T[i][j][m] = pulp.LpVariable(f'T_{i}_{j}_{m}', lowBound=0, cat=self.var_categories['T'])        
        print('done.')    

    def __create_objective(self):
        production_cost = pulp.lpSum(self.site_workshare[i] * self.instance.cost_factor[i] for i in self.instance.production_sites)                                                                
        transport_cost = pulp.lpSum( self.instance.cargo_by_route_recurring_cost[(i, j, m)] * self.instance.cargo_by_route_distance[(i, j, m)] * self.T[i][j][m] for (i, j, m) in self.instance.cargo_by_route_recurring_cost if (i, j) in self.T_domain)        
        emission_cost = pulp.lpSum( self.instance.cargo_by_route_emission[(i, j, m)] * self.instance.cargo_by_route_distance[(i, j, m)] * self.T[i][j][m] for (i, j, m) in self.instance.cargo_by_route_recurring_cost if (i, j) in self.T_domain)        

        supplier_workshare_target_penalty = self.instance.supplier_workshare_target_lambda_penalty * pulp.lpSum((self.supplier_workshare_deviation_above[k] + self.supplier_workshare_deviation_below[k]) for k in self.instance.suppliers)
        site_workshare_target_penalty = self.instance.site_workshare_target_lambda_penalty * pulp.lpSum((self.site_workshare_deviation_above[i] + self.site_workshare_deviation_below[i]) for i in self.instance.production_sites)
        penalties = supplier_workshare_target_penalty + site_workshare_target_penalty
        
        self.prob +=    (
                        self.instance.production_cost_obj_function_weight * production_cost + 
                        self.instance.transport_cost_obj_function_weight * transport_cost + 
                        self.instance.emission_cost_obj_function_weight * emission_cost +
                        self.instance.target_workshare_obj_function_weight * penalties
                        ) 

    def __create_constraints(self):
        # Each part is manufactured at two sites. 
        for s in self.instance.sub_parts:
            self.prob += pulp.lpSum(self.x[s][i] for i in self.instance.production_sites_by_part[s]) == 2

        # The number of units of each part manufactured at all sites equals demand.
        for s in self.instance.parts:
            self.prob += pulp.lpSum(self.y[s][i] for i in self.instance.production_sites_by_part[s]) == self.instance.demand

        # The the number of aircrafts manufactured at each FAL should be as specified in the input. 
        for i in self.instance.FALs:
            if i in self.instance.demand_per_FAL:
                self.prob += self.y[self.instance.assembled_aircraft][i] == self.instance.demand_per_FAL[i]
                self.prob += self.x[self.instance.assembled_aircraft][i] == (1 if self.instance.demand_per_FAL[i] > 0 else 0)
            else:
                self.prob += self.x[self.instance.assembled_aircraft][i] <= self.y[self.instance.assembled_aircraft][i] 
                self.prob += self.y[self.instance.assembled_aircraft][i] <= self.x[self.instance.assembled_aircraft][i] * self.instance.demand

        # TODO: fix country constraints
        # for s in self.instance.sub_parts:
        #     for country in self.instance.production_sites_by_country:
        #         self.prob += pulp.lpSum(self.x[s][i] for i in self.instance.production_sites_by_part[s] if i in self.instance.production_sites_by_country[country]) <= 1                                       
                    
        # The number of units of each part should satisfy the production split requirements (for instance, 20%-80%).
        # Additionally, this enforces that x[s][i] should be positive iff x[s][i] is 1. (Only needed for sub-parts; the complete aircraft is handled above.)
        for s in self.instance.sub_parts:
            for i in self.instance.production_sites_by_part[s]:
                self.prob += self.instance.production_split[s] * self.instance.demand * self.x[s][i] <= self.y[s][i] 
                self.prob += self.y[s][i] <= (1-self.instance.production_split[s]) * self.instance.demand * self.x[s][i]
                        
        for i in self.instance.production_sites:
            # The workshare assigned to a site is the sum of the values of all units of parts manufactured at that site.
            self.prob += self.site_workshare[i] == pulp.lpSum(self.y[s][i] * self.instance.part_value[s] for s in self.instance.parts if i in self.instance.production_sites_by_part[s])
            # The workshare assigned to a site is also the sum of the values of all units of parts manufactured by all suppliers at that site.
            self.prob += self.site_workshare[i] == pulp.lpSum(self.supplier_site_workshare[k][i] for k in self.instance.suppliers_by_location[i])
            # The workshare assigned to a site has to satisfy lower and upper input bounds.
            self.prob += self.site_workshare[i] >= self.instance.min_site_workshare[i] * self.instance.total_value * self.instance.demand
            self.prob += self.site_workshare[i] <= self.instance.max_site_workshare[i] * self.instance.total_value * self.instance.demand
            # Deviation variable constraints.
            self.prob += self.site_workshare[i] - (self.instance.total_value * self.instance.demand) * self.instance.target_site_workshare[i] == self.site_workshare_deviation_above[i] - self.site_workshare_deviation_below[i]
            self.prob += self.site_workshare_deviation_above[i] >= self.site_workshare[i]  - self.instance.target_site_workshare[i] * self.instance.total_value * self.instance.demand
            self.prob += self.site_workshare_deviation_below[i] >= self.instance.target_site_workshare[i] * self.instance.total_value * self.instance.demand - self.site_workshare[i]

        for k in self.instance.suppliers:
            # The workshare assigned to each supplier equals the sum of the workshare assigned to each supplier across its all different sites. 
            self.prob += self.supplier_workshare[k] == pulp.lpSum(self.supplier_site_workshare[k][i] for i in self.instance.locations_by_supplier[k])
            # The workshare assigned to a supplier has to satisfy lower and upper input bounds.
            self.prob += self.supplier_workshare[k] >= self.instance.min_supplier_workshare[k] * self.instance.total_value * self.instance.demand
            self.prob += self.supplier_workshare[k] <= self.instance.max_supplier_workshare[k] * self.instance.total_value * self.instance.demand
            # Deviation variable constraints.
            self.prob += self.supplier_workshare[k] - (self.instance.total_value * self.instance.demand) * self.instance.target_supplier_workshare[k] == self.supplier_workshare_deviation_above[k] - self.supplier_workshare_deviation_below[k]
            self.prob += self.supplier_workshare_deviation_above[k] >= self.supplier_workshare[k]  - self.instance.target_supplier_workshare[k] * self.instance.total_value * self.instance.demand
            self.prob += self.supplier_workshare_deviation_below[k] >= self.instance.target_supplier_workshare[k] * self.instance.total_value * self.instance.demand - self.supplier_workshare[k]


        for i in self.instance.production_sites:
            if i not in self.instance.warehouse_sites:
                for s in self.instance.parts_by_site[i]:
                    if s in self.instance.sub_parts:
                        parent = self.instance.parent_by_part[s]
                        destinations = [j for j in self.instance.sites if ((s, i, j) in self.t_domain)]
                        destinations = set([j for j in destinations if j in self.instance.production_sites_by_part[parent] or j in self.instance.warehouse_sites])
                        if len(destinations)>0:
                            self.prob += pulp.lpSum(self.t[s][i][j] for j in destinations) == self.y[s][i]

        # All subparts have to be delivered in right quantities to each production site.
        for s in self.instance.assembled_parts:
            for r in self.instance.children_by_part[s]:
                for i in self.instance.production_sites_by_part[s]:                
                    if i not in self.instance.warehouse_sites:                        
                        sources = [j for j in self.instance.sites if ((r, j, i) in self.t_domain)]
                        sources = set([j for j in sources if j in self.instance.production_sites_by_part[r] or j in self.instance.warehouse_sites])
                        if len(sources)>0:
                            self.prob += self.y[s][i] == pulp.lpSum(self.t[r][j][i] for j in sources)
                                                

        # The number of units of each sub-part coming in and going out have to be equal at each warehouse.
        # for w in self.instance.warehouse_sites:
        #     if w not in self.instance.production_sites:
        #         self.prob += pulp.lpSum(self.t[s][i][j] for (s, i, j) in self.t_domain if (i==w)) == pulp.lpSum(self.t[s][i][j] for (s, i, j) in self.t_domain if (j==w))        

        for w in self.instance.warehouse_sites:                       #todo: this is the proper way of formulating the constraint that what comes in to a warehouse must go out, but it yields the problem (and relaxation) infeasible. Problem with data?
            # if w not in self.instance.production_sites:
            for s in self.instance.sub_parts:            
                destinations = ([j for j in self.instance.sites if (s, w, j) in self.t_domain])                   
                sources = set([i for i in self.instance.sites if (s, i, w) in self.t_domain])
                if len(sources)>0 and len(destinations)>0:
                    self.prob += pulp.lpSum(self.t[s][w][j] for j in destinations) == pulp.lpSum(self.t[s][i][w] for i in sources)


        for (i, j) in self.T_domain:     
            for m in self.instance.cargo_by_route[(i, j)]:
                parts = [s for s in self.instance.sub_parts if (s, i, j) in self.t_domain and m in self.instance.cargo_by_route_and_part[(i, j, s)]['cargoCapacity']]                                    
                self.prob += self.instance.cargo_capacity_volume[m] * self.T[i][j][m] >= pulp.lpSum(self.c[s][i][j][m]*self.instance.part_volume[s] for s in parts)

        for (s, i, j) in self.t_domain:
            self.prob += self.t[s][i][j] == pulp.lpSum(self.c[s][i][j][m] for m in self.instance.cargo_by_route_and_part[(i, j, s)]['cargoCapacity'])



    def fix_variable_values(self, vars, values):
        assert len(vars) == len(values)
        variables = self.prob.variables()
        variable_names = [var.name for var in self.prob.variables()]
        variable_index = {}
        for i, v in enumerate(variable_names):
            variable_index[v] = i

        for j, var in enumerate(vars):
            i = variable_index[var]
            var = variables[i]
            bound_check = var.setInitialValue(values[j], True)
            assert bound_check
            var.fixValue()

    def __build_mip_model(self, verbose=False):
        if verbose:
            print('Building the MIP model:')
        t0 = time.time()        
        self.prob = pulp.LpProblem("Airbus_Integrated_Production_Planning_and_Transportation", pulp.LpMinimize)    

        # ------------------------------------------   VARIABLES   ------------------------------------------
        if verbose:
            print('\tAdding variables', end='... ', flush=True)            
        self.__create_variables()

        # ------------------------------------------   OBJECTIVE FUNCTION   ------------------------------------------        
        if verbose:
            print('\tAdding objective function', end='... ', flush=True)            
        self.__create_objective()
        if verbose:
            print('done.')

        # ------------------------------------------   CONSTRAINTS   ------------------------------------------    
        if verbose:
            print('\tAdding constraints', end='... ', flush=True)            
        self.__create_constraints()

        t1 = time.time()
        if verbose:            
            print('done.')
            print()

            print(f"\tNumber of variables: {len(self.prob.variables())}")
            print(f"\tNumber of constraints: {len(self.prob.constraints)}")
            print()
            print(f"\tElapsed time: {round(t1-t0, 2)} sec.")
            print()

    def solve(self, verbose=False):
        if verbose:
            print()
            print('Solving the model with HiGHS:', flush=True)
            print()
            print('-' * 140)        
            print()
        solver = pulp.HiGHS(msg=verbose)
        t0 = time.time()
        self.status = self.prob.solve(solver)
        t1 = time.time()        
        if verbose:
            print()
            print('-' * 140)        
            print()
            print(f'Solver terminated. Elapsed time: {round(t1-t0, 2)} sec.')

            # Check the status of the optimization
            if self.status == pulp.LpStatusOptimal:
                print(f"{Back.GREEN}Optimal{Back.RESET} solution found.")
                print(f"Objective value: {self.prob.objective.value()}")
            elif self.status == pulp.LpStatusInfeasible:
                print(f"The problem is {Back.RED}infeasible{Back.RESET}. No solution exists.")
            elif self.status == pulp.LpStatusUnbounded:
                print(f"The problem is {Back.RED}unbounded{Back.RESET}. The objective can increase indefinitely.")
            elif self.status == pulp.LpStatusNotSolved:
                print(f"{Back.RED}The problem has not been solved{Back.RESET}. Please check the setup.")
            else:
                print(f"Solver status: {self.status}.")

    def __count_integer_variables(self, var_dict, numerical_tolerance, verbose=True):
        integer_variable_count = 0
        total_variable_count = 0
        non_integer_variable_count = 0
        none_variable_count = 0
        non_integralities = []

        def non_integrality(value):
            """Return the absolute difference from the nearest integer."""
            return abs(value - round(value))

        # Recursive function to traverse nested dictionaries
        def traverse_variables(d):
            nonlocal integer_variable_count, total_variable_count, non_integer_variable_count, none_variable_count

            if isinstance(d, dict):
                for key, value in d.items():
                    if isinstance(value, dict):
                        traverse_variables(value)  # If it's another dictionary, recurse
                    else:
                        total_variable_count += 1                        
                        # Get the variable value                                                
                        if value.varValue != None:                            
                            var_value = value.value()                        
                            if utils.is_nearly_integer(var_value, numerical_tolerance):
                                integer_variable_count += 1
                            else:                              
                                non_integer_variable_count += 1
                                integrality_gap = non_integrality(var_value)
                                non_integralities.append(integrality_gap)
                        else:
                            none_variable_count += 1

        # Start the traversal
        traverse_variables(var_dict)

        # Compute statistics for non-integer variables
        if non_integralities:
            avg_non_integrality = mean(non_integralities)
            max_non_integrality = max(non_integralities)
            if len(non_integralities) > 1:  # Standard deviation requires at least 2 values
                stddev_non_integrality = stdev(non_integralities)
            else:
                stddev_non_integrality = 0
            proportion_significant_non_integrality_0_1 = sum(1 for gap in non_integralities if gap > 0.1) / non_integer_variable_count
            proportion_significant_non_integrality_0_2 = sum(1 for gap in non_integralities if gap > 0.2) / non_integer_variable_count
            proportion_significant_non_integrality_0_4 = sum(1 for gap in non_integralities if gap > 0.3) / non_integer_variable_count
        else:
            avg_non_integrality = max_non_integrality = stddev_non_integrality = 0
            proportion_significant_non_integrality = 0

        if verbose:
            print(f"\tNumber of variables: {total_variable_count}")
            print(f"\tNumber of integral variables: {integer_variable_count} ({(integer_variable_count*100/total_variable_count):.2f}%)")
            print(f"\tNumber of non-integer variables: {total_variable_count-integer_variable_count} ({((total_variable_count-integer_variable_count)*100/total_variable_count):.2f}%)")
            print(f"\tNumber of None variables: {none_variable_count} ({((none_variable_count)*100/total_variable_count):.2f}%)")

            if non_integer_variable_count > 0:
                print(f"\tAverage non-integrality: {avg_non_integrality:.2f}")
                print(f"\tMaximum non-integrality: {max_non_integrality:.2f}")
                print(f"\tStandard deviation of non-integrality: {stddev_non_integrality:.2f}")
                print(f"\tProportion of variables with significant non-integrality (>0.1): {(proportion_significant_non_integrality_0_1*100):.2f}%")
                print(f"\tProportion of variables with significant non-integrality (>0.2): {(proportion_significant_non_integrality_0_2*100):.2f}%")
                print(f"\tProportion of variables with significant non-integrality (>0.4): {(proportion_significant_non_integrality_0_4*100):.2f}%")


    def integrality_measure(self, numerical_tolerance):
        total_variable_count = 0
        integer_variable_count  = 0
        non_integer_variable_count  = 0
        none_variable_count  = 0
        for var in self.prob.variables():            
            total_variable_count += 1
            if var.varValue != None:                                            
                if utils.is_nearly_integer(var.varValue, numerical_tolerance):
                    integer_variable_count += 1
                else:                              
                    non_integer_variable_count += 1                                        
            else:
                none_variable_count += 1

        return total_variable_count, integer_variable_count, non_integer_variable_count, none_variable_count


    def get_integral_vars(self, numerical_tolerance):
        integral_vars= []
        integral_vars_values = []
        for var in self.prob.variables():                        
            if var.name[0] == 'q':    #discard the feasibility pump vars
                continue
            if var.varValue != None:                                            
                if utils.is_nearly_integer(var.varValue, numerical_tolerance):
                    integral_vars.append(var.name)
                    integral_vars_values.append(round(var.varValue))
        
        return integral_vars, integral_vars_values

    def get_variables_by_name(self):
        variables_by_name = {}
        for var in self.prob.variables():
            variables_by_name[var.name] = var
        return variables_by_name

    def integral_vars_stats(self):
        var_names = ['x', 'y', 't', 'c', 'T', 'supplier_workshare', 'site_workshare', 'supplier_site_workshare', 'supplier_workshare_deviation_above', 'site_workshare_deviation_above']
        var_objects = [getattr(self, var_name) for var_name in var_names]
    
        print()
        for var_name, var in zip(var_names, var_objects):        
            print('Variables', var_name)
            self.__count_integer_variables(var, self.instance.numerical_tolerance)
            print()
  
    def __get_color_for_difference(self, actual, target):
        """Return a color depending on how close actual is to target."""
        difference = abs(actual - target)
        # If the actual is very close to target, use green; if far, use red
        if difference < 0.02:  # Very close
            return Fore.GREEN
        elif difference < 0.05:  # Moderately close
            return Fore.YELLOW
        else:  # Far
            return Fore.RED
          
    def __print_workshare_table(self, title, ids, names, min_workshare, actual_workshare_vals, target_workshare, max_workshare, total_value, demand, id_label="Id", name_label="Name", width=71):
        """
        Prints a formatted table for workshare data.

        Args:
            title (str): Title of the table.
            ids (list): List of IDs (e.g., supplier or site IDs).
            names (list): List of names (e.g., supplier or site names).
            min_workshare (dict): Dictionary of minimum workshare values.
            actual_workshare_vals (dict): Dictionary of actual workshare values.
            max_workshare (dict): Dictionary of maximum workshare values.
            total_value (float): Total value for workshare calculation.
            demand (float): Demand for workshare calculation.
            id_label (str): Label for the ID column.
            name_label (str): Label for the Name column.
            width (int): Total width of the table.
        """
        # Print header and table title
        print('-' * width)
        print(f"{title:^{width}}")  # Center the title
        print('-' * width)

        # Print column headers with alignment
        print(f"{id_label:<5} {name_label:<25} {'Min Workshare':<15} {'Actual Workshare':<20} {'Target Workshare':<20} {'Max Workshare':<15}")
        print('-' * width)

        deviations = []
        # Loop through the data and print each row with proper alignment
        for i in ids:
            actual_workshare = actual_workshare_vals[i].varValue / (total_value * demand)
            deviations.append(abs(actual_workshare-target_workshare[i]))
            color = self.__get_color_for_difference(actual_workshare, target_workshare[i])
            print(f"{i:<5} {names[i]:<25} {min_workshare[i]:<15.3f} {color}{actual_workshare:<20.3f}{Fore.RESET} {target_workshare[i]:<20.3f} {max_workshare[i]:<15.3f}")

        # Print footer
        print('-' * width)
        print(f'Deviations: mean = {mean(deviations):.2f}, std = {stdev(deviations):.2f}, max = {max(deviations):.2f}')
        print('-' * width)

    def print_worshare_table(self):            
        print()        
        self.__print_workshare_table(
            title="Workshare by Production Site",
            ids=self.instance.production_sites,
            names=self.instance.production_site_name,
            min_workshare=self.instance.min_site_workshare,
            actual_workshare_vals=self.site_workshare,
            target_workshare=self.instance.target_site_workshare,
            max_workshare=self.instance.max_site_workshare,
            total_value=self.instance.total_value,
            demand=self.instance.demand,
            id_label="Id",
            name_label="Site Name",
            width=103
        )
        print()        
        print()        
        # Example usage for suppliers
        self.__print_workshare_table(
            title="Workshare by Supplier",
            ids=self.instance.suppliers,
            names=self.instance.supplier_name,
            min_workshare=self.instance.min_supplier_workshare,
            actual_workshare_vals=self.supplier_workshare,
            target_workshare=self.instance.target_supplier_workshare,
            max_workshare=self.instance.max_supplier_workshare,
            total_value=self.instance.total_value,
            demand=self.instance.demand,
            id_label="Id",
            name_label="Supplier Name",
            width=103
        )

    def print_production_stats(self):
        print()
        print('-' * 60)
        print('ASSIGNMENT OF SUBPARTS TO SITES')

        for s in self.instance.sub_parts:
            print('\t',self.instance.part_name[s])
            for i in self.instance.production_sites_by_part[s]:                
                if not math.isclose(self.x[s][i].varValue, 0, abs_tol=self.instance.numerical_tolerance):
                    percentage = abs((self.y[s][i].varValue * 100) / self.instance.demand)
                    print(f'\t\t{self.instance.production_site_name[i]}: {self.y[s][i].varValue:.2f} ({percentage:.2f}%) {self.x[s][i].varValue}')
            print()


        print()
        print('ASSIGNMENTS TO FALs')
        for i in self.instance.FALs:
            # if not math.isclose(self.y[self.instance.assembled_aircraft][i].varValue, 0, abs_tol=self.instance.numerical_tolerance):            
            print(f'\t{self.instance.production_site_name[i]}: {self.y[self.instance.assembled_aircraft][i].varValue:.2f} ({self.y[self.instance.assembled_aircraft][i].varValue*100/self.instance.demand:.2f}%)')
            

        # for s in self.instance.production_sites_by_part:
        #     print(s, self.instance.part_name[s], ' -> ', len(self.instance.production_sites_by_part[s]))
        #     print('\t', self.instance.production_sites_by_part[s])
        #     print()



        # for s in self.instance.sub_parts:            
        #     print(s)
        #     for country in self.instance.production_sites_by_country:                
        #         countries = set()
        #         for i in self.instance.production_sites_by_part[s]:
        #             if i in self.instance.production_sites_by_country[country]:
        #                 if self.x[s][i].varValue > 0:
        #                         print('\t', i, country)
        #                         countries.add(country)                                
        #     print('\tManufactured in', len(countries))
        #     print()

    def print_transportation_stats(self):
        # print()
        # print('-' * 60)
        # print('T variables')
        # for (i, j) in self.instance.cargo_by_route:     
        #     for m in self.instance.cargo_by_route[(i, j)]:
        #         var = self.T[i][j][m].varValue
        #         if not math.isclose(var, 0, abs_tol=self.instance.numerical_tolerance):                    
        #             print(f"T : {self.instance.site_name[i]}, {self.instance.site_name[j]}, {self.instance.cargo_capacity_name[m]} --> {var}")
        # print()


        # print()
        # print('-' * 60)
        # print('c variables')
        # for (i, j, s) in self.instance.cargo_by_route_and_part:
        #     for m in self.instance.cargo_by_route_and_part[(i, j, s)]['cargoCapacity']:
        #         var = self.c[s][i][j][m].varValue
        #         if not math.isclose(var, 0, abs_tol=self.instance.numerical_tolerance):                    
        #             print(f"c : {self.instance.site_name[i]}, {self.instance.site_name[j]}, {self.instance.cargo_capacity_name[m]}, {self.instance.part_name[s]} --> {var}")
        # print()


        print()
        print()
        print('-' * 60)
        print('y variables')
        for s in self.instance.parts:
            for i in self.instance.production_sites_by_part[s]:
                var = self.y[s][i].varValue
                if not math.isclose(var, 0, abs_tol=self.instance.numerical_tolerance):
                    print(f"y : {self.instance.part_name[s]}, {self.instance.site_name[i]} --> {var} {'*' if s in self.instance.sub_parts else ''}")

                for (s_, i_, j) in self.t_domain:
                    if (i_==i) and (s_==s):
                        var = self.t[s][i][j].varValue
                        if not math.isclose(var, 0, abs_tol=self.instance.numerical_tolerance):
                            print(f"\tt : {self.instance.part_name[s]}, {self.instance.site_name[i]}, {self.instance.site_name[j]} --> {var}")
                    

        print()
        print()
        print('-' * 60)
        print('t variables')
        for (s, i, j) in self.t_domain:
            var = self.t[s][i][j].varValue
            if var != None:
                if not math.isclose(var, 0, abs_tol=self.instance.numerical_tolerance):
                    print(f"\tt : {self.instance.part_name[s]}, {self.instance.site_name[i]}, {self.instance.site_name[j]} --> {var}")
            

        print()
        print()
        print('-' * 60)
        print('c variables')
        for (s, i, j) in self.t_domain:
            for m in self.instance.cargo_by_route_and_part[(i, j, s)]['cargoCapacity']:
                var = self.c[s][i][j][m].varValue
                if var != None:
                    if not math.isclose(var, 0, abs_tol=self.instance.numerical_tolerance):
                        print(f"\tc : {self.instance.part_name[s]}, {self.instance.site_name[i]}, {self.instance.site_name[j]} --> {var}")

        print()
        print()
        print('-' * 60)
        print('T variables')
        for (i, j) in self.T_domain:     
            for m in self.instance.cargo_by_route[(i, j)]:        
                var = self.T[i][j][m].varValue                
                if var != None:
                    if not math.isclose(var, 0, abs_tol=self.instance.numerical_tolerance):
                        print(f"\tT : {self.instance.cargo_capacity_name[m]} : {self.instance.site_name[i]} --> {self.instance.site_name[j]} : {var}")


    def solution_to_dict(self):
        ''' Returns a solution that contains names of variables as keys and variable values as values.'''
        solution_dict = {}
        for var in self.prob.variables():
            solution_dict[var.name] = var.varValue            
        return solution_dict
    
    def solution_to_vector(self, var_names):
        ''' Returns a solution vector that contains values in the same order as variable names in var_names.'''
        x = np.zeros(len(var_names))
        variable_index_by_name = {}
        for i, var_name in enumerate(var_names):
            variable_index_by_name[var_name] = i

        for var in self.prob.variables():
            if var.name in variable_index_by_name:
                x[variable_index_by_name[var.name]] = var.varValue        
            
        return x


    def get_constraints_containing_variables(self, var_names):
        ''' Returns the names of all the constraints containing (with a non-zero coefficient) at least one variable from var_names.'''
        constraints_containing_var = []
        for c_name in self.prob.constraints:        
            c = self.prob.constraints[c_name]                    
            coeffs = c.toDict()['coefficients']
            for coeff in coeffs:                
                if (coeff['name'] in var_names) and (coeff['value'] != 0):
                    constraints_containing_var.append(c_name)
        return constraints_containing_var

    
    def obj_function_to_dict(self):
        ''' Returns a dictionary containing variable names as keys and they coefficients in the obj function as values; and a set of variables 
            with non-zero coefficients in the objective function.
        '''
        obj_function_coefficients = {}
        nonzero_obj_function_coefficients = set()
        for item in self.prob.objective.to_dict():
            assert item['name'] not in obj_function_coefficients
            obj_function_coefficients[item['name']] = item['value']
            if item['value']>0:
                nonzero_obj_function_coefficients.add(item['name'])
        return obj_function_coefficients, nonzero_obj_function_coefficients


    def sub_system_constraints(self, vars, vars_to_round_index_by_name, solution_dict):
        n_vars = len(vars)

        new_constraints = []
        new_constants = []
        constraints = self.get_constraints_containing_variables(vars)
        for c_name in constraints:
            new_constraint = [0] * n_vars        
            constraint = self.prob.constraints[c_name].toDict()       
            coeffs = constraint['coefficients']                    
            new_constant = -constraint['constant']
            for j, coeff in enumerate(coeffs):            
                if coeff['name'] in vars:
                    new_constraint[vars_to_round_index_by_name[coeff['name']]] = coeff['value']
                else:
                    new_constant += coeff['value'] * solution_dict[coeff['name']]

            mean_coeff = mean([abs(w) for w in new_constraint] + [abs(new_constant)])
            new_constraint = [w*1.0/mean_coeff for w in new_constraint]
            new_constant /= mean_coeff

            if constraint['sense'] == pulp.LpConstraintGE:
                new_constraint = [-x for x in new_constraint]
                new_constant *= -1
            
            new_constraints.append(new_constraint)
            new_constants.append(new_constant)
            
            if constraint['sense'] == pulp.LpConstraintEQ:
                new_constraints.append([-x for x in new_constraint])
                new_constants.append(-new_constant)

        return new_constraints, new_constants
    
    import numpy as np
    from scipy.sparse import csr_matrix
    from tqdm import tqdm

    def to_matrix(self):
        ''' Returns the matrix A of constraint coefficients, vector b of constants, vector c of objective function coefficients, and the list of variable names. '''
        
        # Step 1: Collect variable names and set up mappings
        variable_names = [var.name for var in self.prob.variables()]
        n_vars = len(variable_names)
        variable_index_by_name = {var_name: i for i, var_name in enumerate(variable_names)}

        # Step 2: Initialize lists to hold data for sparse matrix construction
        row_indices = []  # To store the row index of each non-zero entry
        col_indices = []  # To store the column index of each non-zero entry
        values = []       # To store the values of each non-zero entry
        b = []            # Vector of constants
        
        # Step 3: Collect constraints
        row_index = 0  # Current row index for the constraint
        for c_name in self.prob.constraints:
            c = self.prob.constraints[c_name].toDict()  # Get constraint as a dictionary
            
            coeffs = c['coefficients']
            constant = -c['constant']  # b value for this constraint            

            # Fill in coefficients for this constraint
            for coeff in coeffs:
                var_name = coeff['name']
                value = coeff['value']
                col_index = variable_index_by_name[var_name]
                
                # Store the indices and value for the sparse matrix
                row_indices.append(copy.copy(row_index))
                col_indices.append(copy.copy(col_index))
                values.append(copy.copy(value))                
            
            # Adjust for the type of constraint (GE or LE)
            if self.prob.constraints[c_name].sense == pulp.LpConstraintGE:
                # Flip inequality
                for i in range(len(values)-len(coeffs), len(values)):
                    values[i] = -values[i]  # Negate the values
                constant *= -1
            
            row_index += 1
            b.append(constant)  # Append the corresponding constant to b
            
            # For equality constraints, handle both sides
            if self.prob.constraints[c_name].sense == pulp.LpConstraintEQ:
                for coeff in coeffs:
                    var_name = coeff['name']
                    value = coeff['value']
                    col_index = variable_index_by_name[var_name]
                    
                    # Store the negative of the coefficients for the equality constraint
                    row_indices.append(row_index)  # Reuse the same row index
                    col_indices.append(col_index)
                    values.append(-value)  # Negate the coefficient
                row_index += 1
                b.append(-constant)

        # Step 4: Construct the sparse matrix A
        A = csr_matrix((values, (row_indices, col_indices)), shape=(len(b), n_vars))

        # Step 5: Collect objective function coefficients (vector c)
        objective = self.prob.objective  # Get the objective function
        c = np.zeros(n_vars)  # Initialize vector for c
        for var in self.prob.variables():
            c[variable_index_by_name[var.name]] = objective.get(var, 0.0)  # Assign coefficients

        # Convert b to a numpy array
        b = np.array(b)

        return A, b, c, variable_names


