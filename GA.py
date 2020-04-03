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

    def init_pop(self):
        if self.chromosome_type == int:
            return np.random.randint(low=self.lower, high=self.upper, size=self.population_saphe)
        if self.chromosome_type == float:
            return np.random.uniform(low=self.lower, high=self.upper, size=self.population_saphe)

    def fitness(self, population):
        fitness_dict = {i:self.objective_func(population[i]) for i in range(self.sols_in_generation)}
        min_sol = min(fitness_dict.values())
        max_sol = max(fitness_dict.values())
        avg_pop_sol = sum(fitness_dict.values())/len(fitness_dict)
        self.sol_values.append(min_sol)
        print(min_sol)
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

    def new_population(self,parents_dict, offsprings):
        parents = np.array(list(parents_dict.values()), dtype=self.chromosome_type)
        new_population = parents

        for o in offsprings:
            o = self.mutate_offspring(o)
            new_population = np.vstack((new_population,o))

        return new_population
