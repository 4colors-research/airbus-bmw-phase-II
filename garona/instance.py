'''
This file is part of Garona. 
(c) 4colors Research Ltd, 2024

This file contains code for loading the input instance.
'''

import pandas as pd
from collections import defaultdict
import networkx as nx
import copy, json, math, random, time 


class Instance:
    files = [ 
            'cargo-capacities',
            'cargos',
            'manufacturing-resources',
            'products',
            'products-to-parts',
            'production-locations',            
            'routes',
            'suppliers',
            'transportation-resources',
            'warehouse-locations',
            ]


    def __load_files(self):
        self.df = {}
        for file in self.files:                        
            file_path = self.path_to_data_files + file + '.csv'            
            self.df[file] = pd.read_csv(file_path)


    def __load_params(self):
        with open('instance-params.json', 'r') as file:
            self.params = json.load(file) 
            random.seed(self.params['random_seed'] if self.params['random_seed']!=0 else time.time())
            self.numerical_tolerance = self.params['numerical_tolerance']


    def __load_additional_instance_input(self):
        with open('instance-input.json', 'r') as file:
            self.input = json.load(file)            
            self.default_production_split = self.input['default_production_split']            

            self.production_cost_obj_function_weight = self.input['production_cost_obj_function_weight']
            self.transport_cost_obj_function_weight = self.input['transport_cost_obj_function_weight']
            self.emission_cost_obj_function_weight = self.input['emission_cost_obj_function_weight']
            self.target_workshare_obj_function_weight = self.input['target_workshare_obj_function_weight']

            self.supplier_workshare_target_lambda_penalty = self.input['supplier_workshare_target_lambda_penalty']
            self.site_workshare_target_lambda_penalty = self.input['site_workshare_target_lambda_penalty']

            self.demand = self.input['demand']
            self.demand_per_FAL_by_name = self.input['demand_per_FAL']                    


    def __init_products(self):
        self.parts = [] 
        self.part_id = {}
        self.part_name = {}
        self.part_value = {}
        self.part_volume = {}
        self.part_by_id = {}
        for index, row in self.df['products'].iterrows():
            self.part_id[index] = row['id']
            self.part_by_id[row['id']] = index
            self.part_name[index] = row['name']
            self.part_value[index] = row['valueAdded']
            self.part_volume[index] = row['length']
            if math.isnan(row['diameter']):
                self.part_volume[index] *= row['width'] * row['height']
            else:        
                self.part_volume[index] *= row['diameter']
            self.part_volume[index] *= 1e-9
            self.parts.append(index)
        self.n_parts = len(self.parts)
        self.total_volume = int(sum(self.part_volume.values()) * self.demand+1)

        self.total_value = sum(self.part_value.values())
        self.sub_parts = [self.part_by_id[part] for part in self.df['products-to-parts']['part'].dropna().unique()]
        self.n_sub_parts = len(self.sub_parts)
        self.assembled_parts = [self.part_by_id[part] for part in self.df['products-to-parts'][self.df['products-to-parts']['part'].notna()]['id'].unique()]
        self.n_assembled_parts = len(self.assembled_parts)

        df = self.df['products-to-parts'][self.df['products-to-parts']['part'].notna()].groupby(['id']).agg(list)
        self.children_by_part = {}
        self.parent_by_part = {}
        for index, row in df.iterrows():                
            self.children_by_part[self.part_by_id[index]] = [self.part_by_id[subpart] for subpart in row['part']]
            for subpart in row['part']:
                self.parent_by_part[self.part_by_id[subpart]] = self.part_by_id[index]         


    def __init_production_sites(self):        
        # self.df['production-locations'] = pd.read_csv(self.path_to_data_files + 'production-locations.csv')
        df = self.df['production-locations'].drop(['transportTerminal', 'transportRegion'], axis=1).drop_duplicates().reset_index(drop=True)

        self.production_sites = []
        self.production_site_id = {}
        self.production_site_name = {}
        self.cost_factor = {}
        self.max_site_workshare = {}
        self.min_site_workshare = {}
        self.target_site_workshare = {}
        self.production_site_by_id = {}
        self.production_site_by_name = {}
        self.production_sites_by_country = defaultdict(list)

        for index, row in df.iterrows():
            self.production_site_id[index] = row['id']
            self.production_site_by_id[row['id']] = index
            self.production_site_name[index] = row['name']
            self.production_site_by_name[row['name']] = index

            # >>> ------------------------------------------------------------------------------------------            
            # This needs to be adjusted manually as the numbers in the original file give an infeasible problem.
            self.min_site_workshare[index] = row['minimumWorkshare'] / 100.0
            self.max_site_workshare[index] = row['maximumWorkshare'] / 100.0
            # self.min_site_workshare[index] = 0.0
            # self.max_site_workshare[index] = 1.0            
            # >>> ------------------------------------------------------------------------------------------                        
            self.target_site_workshare[index] = (self.min_site_workshare[index] + self.max_site_workshare[index]) / 2.0
            self.cost_factor[index] = row['costFactor']
            self.production_sites.append(index)    
            self.production_site_ids = list(self.production_site_id.values())
            self.production_sites_by_country[row['country']].append(index)

        self.production_sites_by_part = defaultdict(list)
        self.parts_by_site = {}
        df = self.df['manufacturing-resources'].groupby(['location']).agg(list)
        for index, row in df.iterrows():    
            site = self.production_site_by_id[index]
            self.parts_by_site[site] = list(set([self.part_by_id[part] for part in row['product']]))    
            for part in row['product']:
                self.production_sites_by_part[self.part_by_id[part]].append(site)

        for s in self.production_sites_by_part:
            self.production_sites_by_part[s] = list(set(self.production_sites_by_part[s]))

        self.assembled_aircraft = list(set(self.parts)-set(self.sub_parts))[0]
        self.FALs = self.production_sites_by_part[self.assembled_aircraft]        
        self.n_FALs = len(self.FALs)
        self.n_production_sites = len(self.production_sites)        

        self.production_split = {}
        for part in self.parts:
            self.production_split[part] = self.default_production_split
        # self.production_split[self.assembled_aircraft] = 0.0
        
        self.demand_per_FAL = {}
        for FAL_name in self.input['demand_per_FAL']:
            FAL_index = self.production_site_by_name[FAL_name]
            self.demand_per_FAL[FAL_index] = self.input['demand_per_FAL'][FAL_name]
        assert self.demand >= sum(self.demand_per_FAL.values())


    def __init_warehouse_sites(self):
        self.warehouse_sites = []
        self.warehouse_site_id = {}
        self.warehouse_site_name = {}
        self.warehouse_site_by_id = {}

        df = self.df['warehouse-locations'].drop(['transportTerminal', 'transportRegion', 'country'], axis=1).drop_duplicates().reset_index(drop=True)        

        for index, row in df.iterrows():
            warehouse_site_id = index+self.n_production_sites
            if row['id'] in self.production_site_by_id:
                warehouse_site_id = self.production_site_by_id[row['id']]
            self.warehouse_site_id[index+self.n_production_sites] = row['id']
            self.warehouse_site_by_id[row['id']] = warehouse_site_id
            self.warehouse_site_name[warehouse_site_id] = row['name']        
            self.warehouse_sites.append(warehouse_site_id)
        self.warehouse_site_ids = list(self.warehouse_site_id.values())
        self.n_warehouse_sites = len(self.warehouse_sites)        

        self.sites = self.production_sites + self.warehouse_sites
        self.site_id = self.production_site_id | self.warehouse_site_id
        self.site_name = self.production_site_name | self.warehouse_site_name
        self.site_by_id = self.production_site_by_id | self.warehouse_site_by_id
        self.site_ids = self.production_site_ids + self.warehouse_site_ids
        self.n_sites = len(self.sites)
        assert self.n_sites == self.n_production_sites + self.n_warehouse_sites
        

    def __init_suppliers(self):
        self.locations_by_supplier = defaultdict(list)
        self.suppliers_by_location = defaultdict(list)
        self.supplier_name = {}
        self.min_supplier_workshare = {}
        self.max_supplier_workshare = {}
        self.target_supplier_workshare = {}
        self.suppliers = []
        df = self.df['suppliers'].groupby(['id']).agg(list).reset_index(drop=True)
        for index, row in df.iterrows():        
            self.locations_by_supplier[index] = [self.site_by_id[id] for id in row['location']]
            for i in [self.site_by_id[id] for id in row['location']]:
                self.suppliers_by_location[i].append(index)

            self.supplier_name[index] = row['name'][0]
            # >>> ------------------------------------------------------------------------------------------            
            # This needs to be adjusted manually as the numbers in the original file give an infeasible problem.
            # self.min_supplier_workshare[index] = row['minimumWorkshare'][0] / 100.0
            # self.max_supplier_workshare[index] = row['maximumWorkshare'][0] / 100.0
            self.min_supplier_workshare[index] = 0.0
            self.max_supplier_workshare[index] = 1.0    
            # >>> ------------------------------------------------------------------------------------------            
            # self.target_supplier_workshare[index] = row['targetWorkshare'][0]
            # self.target_supplier_workshare[index] = (self.min_supplier_workshare[index] + self.max_supplier_workshare[index]) / 2
            self.target_supplier_workshare[index] = row['targetWorkshare'][0] / 100.0
            self.suppliers.append(index)


    def __init_transport(self):
        df = self.df['routes'].drop(['id', 'name', 'transportationResource'], axis=1).drop_duplicates(['sourceLocation', 'destinationLocation'])
        df = df.reset_index(drop=True)

        self.allowed_routes = []
        self.transport_cost = {}
        for index, row in df.iterrows():
            source_location = row['sourceLocation']
            destination_location = row['destinationLocation']
            self.transport_cost[(self.site_by_id[source_location], self.site_by_id[destination_location])] = row['distance']
            self.allowed_routes.append((self.site_by_id[source_location], self.site_by_id[destination_location]))    
            
        for site in self.production_sites:
            self.transport_cost[(site, site)] = 0
            self.allowed_routes.append((site, site))


    def __init_cargo(self):
        df_cargo = self.df['cargos']
        cargo_volume = {}
        for index, row in df_cargo.iterrows():
            id = row['id']            
            cargo_volume[id] = row['length'] * row['width'] * row['height'] * 1e-9

        df_cargo_capacity = self.df['cargo-capacities']
        self.cargo_capacity_name = {}
        self.cargo_capacity_volume = {}
        for index, row in df_cargo_capacity.iterrows():
            id = row['id']
            self.cargo_capacity_name[id] = row['name']
            self.cargo_capacity_volume[id] = cargo_volume[row['cargo']] * row['cargoCount']

        intra_factory = 'INTRA_FACTORY'
        self.cargo_capacity_name[intra_factory] = intra_factory
        self.cargo_capacity_volume[intra_factory] = self.total_volume

        self.cargos = list(self.cargo_capacity_volume)

        df_routes = self.df['routes']
        df_cargo = self.df['transportation-resources']            
        merged_df = pd.merge(
            df_routes,   
            df_cargo,  
            left_on='transportationResource',              
            right_on='id',                                 
            how='inner'                                    
        )
        df = merged_df[['sourceLocation', 'destinationLocation', 'name_x', 'product', 'cargoCapacity', 'distance', 'recurringCosts', 'co2Emissions']].groupby(['sourceLocation', 'destinationLocation', 'product']).agg(list)
        
        self.cargo_by_route_and_part = defaultdict(list)
        self.cargo_by_route = defaultdict(set)        
        self.cargo_by_route_recurring_cost = defaultdict()
        self.cargo_by_route_emission = defaultdict()
        self.cargo_by_route_distance = defaultdict()
        for (source_location, destination_location, product), row in df.iterrows():
            for c in row['cargoCapacity']:
                self.cargo_by_route[(self.site_by_id[source_location], self.site_by_id[destination_location])].add(c)
                self.cargo_by_route_recurring_cost[(self.site_by_id[source_location], self.site_by_id[destination_location], c)] = row['recurringCosts'][0]
                self.cargo_by_route_emission[(self.site_by_id[source_location], self.site_by_id[destination_location], c)] = row['co2Emissions'][0]
                self.cargo_by_route_distance[(self.site_by_id[source_location], self.site_by_id[destination_location], c)] = row['distance'][0]
                self.cargo_by_route_and_part[(self.site_by_id[source_location], self.site_by_id[destination_location], self.part_by_id[product])] =  {   
                                                                                                                                    'name' : row['name_x'], 
                                                                                                                                    'cargoCapacity' : set(row['cargoCapacity']),
                                                                                                                                    'distance' : row['distance'],
                                                                                                                                    'recurringCosts' : row['recurringCosts'],
                                                                                                                                    'co2Emissions' : row['co2Emissions']
                                                                                                                                    }                                
        for i in self.production_sites:
            self.cargo_by_route[(i, i)].add(intra_factory)
            self.cargo_by_route_recurring_cost[(i, i, intra_factory)] = 0
            self.cargo_by_route_emission[(i, i, intra_factory)] = 0
            self.cargo_by_route_distance[(i, i, intra_factory)] = 0
            for s in self.parts_by_site[i]:
                if s in self.sub_parts:
                    self.cargo_by_route_and_part[(i, i, s)] =  {   
                                                            'name' : intra_factory, 
                                                            'cargoCapacity' : {intra_factory},
                                                            'distance' : 0,
                                                            'recurringCosts' : 0,
                                                            'co2Emissions' : 0
                                                            }                
        dict_copy = copy.deepcopy(self.cargo_by_route_and_part)

        #This fixes problems with several parts whose transportation networks are disconnected.
        for (i, j, s) in dict_copy:
            if s == 2:
                self.cargo_by_route_and_part[(i, j, 5)] = self.cargo_by_route_and_part[(i, j, 2)]
                self.cargo_by_route_and_part[(i, j, 13)] = self.cargo_by_route_and_part[(i, j, 2)]
                self.cargo_by_route_and_part[(i, j, 45)] = self.cargo_by_route_and_part[(i, j, 2)]
                self.cargo_by_route_and_part[(i, j, 0)] = self.cargo_by_route_and_part[(i, j, 2)]
                self.cargo_by_route_and_part[(i, j, 10)] = self.cargo_by_route_and_part[(i, j, 2)]
                self.cargo_by_route_and_part[(i, j, 12)] = self.cargo_by_route_and_part[(i, j, 2)]
                self.cargo_by_route_and_part[(i, j, 36)] = self.cargo_by_route_and_part[(i, j, 2)]
                self.cargo_by_route_and_part[(i, j, 40)] = self.cargo_by_route_and_part[(i, j, 2)]

        
    def __transportation_network_by_parts(self):
        for s in self.sub_parts:
            print('-' * 60)
            print(self.part_name[s], s)
            print('-' * 60)
            edges = [(i, j) for (i, j, r) in self.cargo_by_route_and_part if r == s]            
            G = nx.DiGraph()
            G.add_edges_from(edges)            
            G_undirected = G.to_undirected()
            
            print(f'\tNumber of vertices: {G.number_of_nodes()}')
            print(f'\tNumber of directed arcs: {G.number_of_edges()}')
            print(f'\tNumber of undirected edges: {G_undirected.number_of_edges()}')
            print() 

            is_strongly_connected = nx.is_strongly_connected(G)
            print(f"\tStrongly connected: {is_strongly_connected}")
            is_weakly_connected = nx.is_connected(G_undirected)
            print(f"\tWeakly connected: {is_weakly_connected}")
            is_connected = nx.is_connected(G_undirected)            
            print(f"\tConnected (in undirected sense): {is_connected}")
            print()

            # if not is_strongly_connected:
            #     # To find strongly connected components (SCCs)
            #     sccs = list(nx.strongly_connected_components(G))
            #     print(f"\tStrongly connected components: ")
            #     print()
            #     for i, cc in enumerate(sccs):
            #         print(f"{i}. {[self.site_name[i] for i in cc]}")
            #     print()

            if not is_weakly_connected:
                # To find weakly connected components
                wccs = list(nx.weakly_connected_components(G))
                print(f"\tWeakly connected components: ")
                print()
                for i, cc in enumerate(wccs):
                    print(f"{i}. {[self.site_name[i] for i in cc]}")
                    print()
               
    def __transportation_network_stats(self):
        edges = self.allowed_routes
        # Create a directed graph from the given edges
        G = nx.DiGraph()
        G.add_edges_from(edges)
        # Convert to an undirected graph to check weak connectivity
        G_undirected = G.to_undirected()

        print('Transportation network statistics:')        
        print(f'\tNumber of vertices: {G.number_of_nodes()}')
        print(f'\tNumber of directed arcs: {G.number_of_edges()}')
        print(f'\tNumber of undirected edges: {G_undirected.number_of_edges()}')
        print() 

        # Determine strong connectivity
        is_strongly_connected = nx.is_strongly_connected(G)
        print(f"\tStrongly connected: {is_strongly_connected}")

        if not is_strongly_connected:
            # To find strongly connected components (SCCs)
            sccs = list(nx.strongly_connected_components(G))
            print(f"\tStrongly connected components: {sccs}")
            print()
            for cc in sccs:
                for site in cc:
                    print(site, self.site_name[site], self.site_id[site], end = ',')
                print()
                print('-----')
                print()

        # Determine weak connectivity    
        is_weakly_connected = nx.is_connected(G_undirected)
        print(f"\tWeakly connected: {is_weakly_connected}")

        if not is_weakly_connected:
            # To find weakly connected components
            wccs = list(nx.weakly_connected_components(G))
            print(f"\tWeakly connected components: {wccs}")

        # To check if the graph is connected in terms of undirected components
        is_connected = nx.is_connected(G_undirected)
        print(f"\tConnected (in undirected sense): {is_connected}")
        print()


    def __input_instance_stats(self):
        print('Instance statistics:')
        print(f'\tTotal number of parts: {self.n_parts}')
        print(f'\tNumber of assembled parts: {self.n_assembled_parts}')
        print(f'\tNumber of sub-parts: {self.n_sub_parts}')
        print()

        print(f'\tTotal number of sites: {self.n_sites}')
        print(f'\tNumber of production sites: {self.n_production_sites}')
        print(f'\tNumber of warehouse sites: {self.n_warehouse_sites}')
        print(f'\tNumber of FALs (Final Assembly Lines): {self.n_FALs}')
        print()
        

    def __init__(self, path_to_data_files):        
        self.path_to_data_files = path_to_data_files                

        print()
        print('Loading input files:')

        print('\tLoading industrial system files', end='... ', flush=True)
        self.__load_files()
        print('done.')

        print('\tLoading parameters file', end='... ', flush=True)                
        self.__load_params()
        print('done.')

        print()
        print('Building the input instance:')
                
        print('\tLoading additonal instance input', end='... ', flush=True)                
        self.__load_additional_instance_input()
        print('done.')

        print('\tBuilding product information', end='... ', flush=True)
        self.__init_products()
        print('done.')

        print('\tInitializing site information', end='... ', flush=True)
        self.__init_production_sites()
        self.__init_warehouse_sites()
        self.__init_suppliers()
        print('done.')

        print('\tBuilding transportation network', end='... ', flush=True)
        self.__init_transport()        
        self.__init_cargo()        
        print('done.')
        print()
        
        self.__input_instance_stats()        
        self.__transportation_network_stats()
