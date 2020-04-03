import numpy as np

def ackley(sol):
  x1,x2 = sol
  part_1 = -0.2*np.sqrt(0.5*(x1*x1 + x2*x2))
  part_2 = 0.5*(np.cos(2*np.pi*x1) + np.cos(2*np.pi*x2))
  value = np.exp(1) + 20 -20*np.exp(part_1) - np.exp(part_2)
  return value
  
GA = GA(single_sol_shape=2,sol_lower=-5,sol_upper=6,sols_in_population=8,num_parents_mating=2,objective_func=ackley)
new_population = GA.init_pop()
num_generations = 10

for i in range(num_generations):
    fitness_dict = GA.fitness(new_population)
    parent_dict = GA.choose_parents(fitness_dict,new_population)
    offsprings = GA.crossover(parent_dict,0.5)
    new_population = GA.new_population(parent_dict,offsprings)

plt.plot(GA.sol_values)
plt.show()
