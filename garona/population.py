'''
This file is part of Garona. 
(c) 4colors Research Ltd, 2024

This file contains code for population-based methods.
'''

import pickle, random
import numpy as np


class SolutionPool:
    def __init__(self):
        self.pool = []        

    def add(self, solution):
        self.pool.append(solution)
        self.save()

    def save(self):        
        with open("solutions.pkl", 'wb') as f:                
            pickle.dump(self.pool, f)      



def filter_variables(variables, name):
    filtered_variables = {}
    name_len = len(name)
    for var in variables:
        if var[0:name_len] == name:
            if bool(re.fullmatch(r'[\d_]+', var[name_len:])) or var[0] == 'T':
                filtered_variables[var] = var[name_len+1:].split("_")
                if var[0] == 'T':
                    filtered_variables[var] = [filtered_variables[var][0], filtered_variables[var][1], "_".join(filtered_variables[var][2:])]
    return filtered_variables



def evaluate(instance, variables, solution):
    site_workshare_vars = filter_variables(variables, 'site_workshare')
    T_vars = filter_variables(variables, 'T')
    supplier_workshare_deviation_above = filter_variables(variables, 'supplier_workshare_deviation_above')
    supplier_workshare_deviation_below = filter_variables(variables, 'supplier_workshare_deviation_below')
    site_workshare_deviation_above = filter_variables(variables, 'site_workshare_deviation_above')
    site_workshare_deviation_below = filter_variables(variables, 'site_workshare_deviation_below')

    production_cost = 0
    for var in site_workshare_vars:
        i = int(site_workshare_vars[var][0])
        production_cost += solution[var] * instance.cost_factor[i]

    transport_cost = 0
    emission_cost = 0
    for var in T_vars:
        if 'INTRA' not in var:
            i = int(T_vars[var][0])
            j = int(T_vars[var][1])
            m = T_vars[var][2]
            transport_cost += instance.cargo_by_route_recurring_cost[(i, j, m)] * instance.cargo_by_route_distance[(i, j, m)] * solution[var]                                   
            emission_cost += instance.cargo_by_route_emission[(i, j, m)] * instance.cargo_by_route_distance[(i, j, m)] * solution[var]


    supplier_workshare_target_penalty = 0
    for var in supplier_workshare_deviation_above:
        var_above = var
        var_below = var.replace("above", "below")        
        supplier_workshare_target_penalty += solution[var_above] + solution[var_below]
    supplier_workshare_target_penalty *= instance.supplier_workshare_target_lambda_penalty

    site_workshare_target_penalty = 0
    for var in site_workshare_deviation_above:
        var_above = var
        var_below = var.replace("above", "below")        
        supplier_workshare_target_penalty += solution[var_above] + solution[var_below]
    site_workshare_target_penalty *= instance.site_workshare_target_lambda_penalty

    penalties = supplier_workshare_target_penalty + site_workshare_target_penalty
        
    obj_value = (
                instance.production_cost_obj_function_weight * production_cost + 
                instance.transport_cost_obj_function_weight * transport_cost + 
                instance.emission_cost_obj_function_weight * emission_cost +
                instance.target_workshare_obj_function_weight * penalties
                )
    
    return obj_value




class GeneticAlgorithm:
    def __init__(self, instance, variables, population : SolutionPool, mutation_rate=0.01, crossover_rate=0.8, generations=100):
        self.instance = instance 
        self.variables = variables
        self.population_size = population_size        
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generations = generations
        self.population = population        
    
    
    def _selection(self):
        """
        Select two individuals based on their fitness using tournament selection.
        :return: Two selected individuals (parents)
        """
        tournament_size = 3
        selected = random.sample(self.population, tournament_size)
        selected.sort(key=self._fitness, reverse=True)
        return selected[0], selected[1]


    def _crossover(self, parent1, parent2):
        """
        Perform crossover between two parents.
        :param parent1: First parent (chromosome)
        :param parent2: Second parent (chromosome)
        :return: Two new children (chromosomes)
        """
        if random.random() < self.crossover_rate:
            crossover_point = random.randint(1, self.chromosome_length - 1)
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]
        else:
            child1, child2 = parent1.copy(), parent2.copy()
        return child1, child2


    def _fitness(self, individual):        
        return evaluate(self.instance, self.variables, individual)


    def _mutate(self, individual):
        """
        Mutate an individual by flipping bits with a probability equal to the mutation rate.
        :param individual: An individual (chromosome) to mutate
        :return: Mutated individual
        """
        for i in range(self.chromosome_length,
                       ):
            if random.random() < self.mutation_rate:
                individual[i] = 1 - individual[i]  # Flip the bit (0->1 or 1->0)
        return individual

    def _evolve(self):
        """
        Evolve the population for one generation.
        :return: New population
        """
        new_population = []
        while len(new_population) < self.population_size:
            # Selection
            parent1, parent2 = self._selection()
            
            # Crossover
            child1, child2 = self._crossover(parent1, parent2)
            
            # Mutation
            child1 = self._mutate(child1)
            child2 = self._mutate(child2)
            
            # Add offspring to the new population
            new_population.append(child1)
            new_population.append(child2)
        
        # Ensure the new population has the correct size (may slightly exceed due to even crossover)
        self.population = new_population[:self.population_size]
    
    def run(self):
        """
        Run the genetic algorithm for a specified number of generations.
        :return: The best individual found and its fitness value.
        """
        for generation in range(self.generations):
            self._evolve()
            best_individual = max(self.population, key=self._fitness)
            best_fitness = self._fitness(best_individual)
            print(f"Generation {generation + 1} | Best Fitness: {best_fitness}")
        
        best_individual = max(self.population, key=self._fitness)
        return best_individual, self._fitness(best_individual)


# Example usage
if __name__ == "__main__":
    # Define a simple fitness function
    def fitness_function(chromosome):
        return sum(chromosome)  # Example: maximize the number of 1's in the chromosome
    
    # Parameters
    population_size = 20
    chromosome_length = 10
    generations = 50
    mutation_rate = 0.05
    crossover_rate = 0.8
    
    # Create GA instance
    ga = GeneticAlgorithm(
        population_size=population_size,
        chromosome_length=chromosome_length,
        fitness_function=fitness_function,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        generations=generations
    )
    
    # Run the GA
    best_individual, best_fitness = ga.run()
    print(f"Best Individual: {best_individual}")
    print(f"Best Fitness: {best_fitness}")
