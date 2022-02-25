import numpy as np
import heapq


class GA:
    def __init__(self, x_dim, x_lb, x_ub, pop_size, num_mating, objective_func, x_type=int):
        self.x_dim = x_dim
        self.x_lb = x_lb
        self.x_ub = x_ub
        self.pop_size = pop_size
        self.num_mating = num_mating
        self.objective_func = objective_func
        self.x_type = x_type
        self.pop_shape = (pop_size, x_dim)
        self.sol_values = []
        self.best_sol = np.zeros(x_dim)
        self.best_sol_value = 0
        self.best_x = None

    def init_pop(self):
        if self.x_type == int:
            return np.random.randint(low=self.x_lb, high=self.x_ub, size=self.pop_shape)
        if self.x_type == float:
            return np.random.uniform(low=self.x_lb, high=self.x_ub, size=self.pop_shape)

    def init_best_sol_value(self, population, *args):
        """ insert random value for the best values variable """

        self.best_sol_value = self.objective_func(population[0], *args)
        if self.best_sol_value is not None:
            return self.best_sol_value
        else:
            self.init_best_sol_value(population, *args)

    def fitness(self, population, *args):
        fitness_dict = {i: self.objective_func(population[i], *args) for i in range(self.pop_size)}
        min_idx, min_sol = min(fitness_dict.items(), key=lambda x: x[1])
        max_sol = max(fitness_dict.values())
        avg_pop_sol = sum(fitness_dict.values()) / len(fitness_dict)
        self.sol_values.append(min_sol)
        if min_sol < self.best_sol_value:
            self.best_sol_value = min_sol
            self.best_x = population[min_idx]
        return fitness_dict

    def choose_parents(self, fitness_dict, population):
        parents_idx = heapq.nsmallest(self.num_mating, fitness_dict, key=fitness_dict.get)
        parents_dict = {i: population[i] for i in parents_idx}
        return parents_dict

    def crossover(self, parents_dict, crossover_portion):
        crossover_idx = int(crossover_portion * self.x_dim)
        parents = np.array(list(parents_dict.values()), dtype=self.x_type)

        offsprings = np.zeros((self.pop_size - self.num_mating, self.x_dim))
        for q in range((self.pop_size - self.num_mating)):
            # Index of the first parent to mate.
            parent1 = q % self.num_mating
            # Index of the second parent to mate.
            parent2 = (q + 1) % self.num_mating

            # The new offspring will have its first half of its genes taken from the first parent.
            offsprings[q, 0:crossover_idx] = parents[parent1][0:crossover_idx]
            # The new offspring will have its second half of its genes taken from the second parent.
            offsprings[q, crossover_idx:] = parents[parent2][crossover_idx:]

        return offsprings

    def mutate_offspring(self, single_offspring):
        mutate_idx = np.random.randint(0, self.x_dim)
        random_value = np.random.randint(self.x_lb, self.x_ub, 1)
        single_offspring[mutate_idx] = random_value
        return single_offspring

    def new_population(self, parents_dict, offsprings, mutation_rate=1):
        parents = np.array(list(parents_dict.values()), dtype=self.x_type)
        new_population = parents
        for o in offsprings:
            for i in range(mutation_rate):
                o = self.mutate_offspring(o)
            new_population = np.vstack((new_population, o))
        return new_population

    def optimize(self, n_generations, mutation_rate, *args):
        new_population = self.init_pop()
        self.best_sol_value = self.init_best_sol_value(new_population, *args)
        for i in range(n_generations):
            fitness_dict = self.fitness(new_population, *args)
            parent_dict = self.choose_parents(fitness_dict, new_population)
            offsprings = self.crossover(parent_dict, 0.5)
            new_population = self.new_population(parent_dict, offsprings, mutation_rate)
        return self.sol_values
