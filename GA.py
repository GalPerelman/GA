import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import heapq

class GA:
    def __init__(self, single_sol_shape, sol_lower, sol_upper, sols_in_population, num_parents_mating,objective_func,chromosome_type=int):
        self.single_sol_shape = single_sol_shape
        self.lower = sol_lower
        self.upper = sol_upper
        self.sols_in_generation = sols_in_population
        self.num_parents_mating = num_parents_mating
        self.objective_func = objective_func
        self.chromosome_type = chromosome_type
        self.population_saphe = (sols_in_population,single_sol_shape)
        self.sol_values = []
        self.best_sol = np.zeros(single_sol_shape)
        self.best_sol_value = 0


    def init_pop(self):
        if self.chromosome_type == int:
            return np.random.randint(low=self.lower, high=self.upper, size=self.population_saphe)
        if self.chromosome_type == float:
            return np.random.uniform(low=self.lower, high=self.upper, size=self.population_saphe)

    def init_best_sol_value(self,population,*args):
        ''' insert random value for the best values variable'''
        self.best_sol_value = self.objective_func(population[0],*args)
        if self.best_sol_value != None:
            return self.best_sol_value
        else:
            self.init_best_sol_value(population,*args)

    def fitness(self, population,*args):
        fitness_dict = {i:self.objective_func(population[i],*args) for i in range(self.sols_in_generation)}
        min_idx, min_sol = min(fitness_dict.items(), key=lambda x: x[1])
        max_sol = max(fitness_dict.values())
        avg_pop_sol = sum(fitness_dict.values())/len(fitness_dict)
        self.sol_values.append(min_sol)
        if min_sol < self.best_sol_value:
            self.best_sol_value = min_sol
            self.best_x = population[min_idx]
        return fitness_dict

    def choose_parents(self, fitness_dict, population):
        parents_idx = heapq.nsmallest(self.num_parents_mating, fitness_dict, key=fitness_dict.get)
        parents_dict = {i: population[i] for i in parents_idx}
        return parents_dict

    def crossover(self, parents_dict, crossover_portion):
        crossover_idx = int(crossover_portion * self.single_sol_shape)
        parents = np.array(list(parents_dict.values()), dtype=self.chromosome_type)

        offsprings = np.zeros((self.sols_in_generation - self.num_parents_mating, self.single_sol_shape))
        for q in range((self.sols_in_generation - self.num_parents_mating)):
            # Index of the first parent to mate.
            parent1 = q % self.num_parents_mating
            # Index of the second parent to mate.
            parent2 = (q + 1) % self.num_parents_mating

            # The new offspring will have its first half of its genes taken from the first parent.
            offsprings[q,0:crossover_idx] = parents[parent1][0:crossover_idx]
            # The new offspring will have its second half of its genes taken from the second parent.
            offsprings[q,crossover_idx:] = parents[parent2][crossover_idx:]

        return offsprings

    def mutate_offspring(self,single_offspring):
        mutate_idx = np.random.randint(0, self.single_sol_shape)
        random_value = np.random.randint(self.lower, self.upper, 1)
        single_offspring[mutate_idx] = random_value
        return single_offspring

    def new_population(self,parents_dict, offsprings, mutation_rate=1):
        parents = np.array(list(parents_dict.values()), dtype=self.chromosome_type)
        new_population = parents
        for o in offsprings:
            for i in range(mutation_rate):
                o = self.mutate_offspring(o)
            new_population = np.vstack((new_population,o))
        return new_population

    def optimize(self,n_generations,mutation_rate,*args):
        new_population = self.init_pop()
        self.best_sol_value = self.init_best_sol_value(new_population,*args)
        for i in range(n_generations):
            fitness_dict = self.fitness(new_population,*args)
            parent_dict = self.choose_parents(fitness_dict, new_population)
            offsprings = self.crossover(parent_dict, 0.5)
            new_population = self.new_population(parent_dict, offsprings, mutation_rate)
        return self.sol_values
