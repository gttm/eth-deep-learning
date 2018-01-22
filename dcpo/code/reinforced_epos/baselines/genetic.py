import reinforced_epos.helpers.dataset as ds
from  reinforced_epos.helpers.oop.Environment import Environment
import random
import numpy as np
from deap import creator, base, tools, algorithms


dataset = ds.get_dataset()
shape = np.shape(dataset)
print(dataset)
print(shape)
env = Environment(dataset, 1, 1)

creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


toolbox = base.Toolbox()

toolbox.register("gene", random.randint, 0, shape[1]-1) #a gene is a plan choice
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.gene, n=shape[0]) # an individulal of genome size 100
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

var_best = float("inf")

def evalOneMax(individual):
    #print(np.shape(np.array(individual)))
    #print(individual)
    var = float(env.state_variance(np.array(individual))[0])
    global var_best
    if var < var_best:
        var_best = var
    return var,


toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

population = toolbox.population(n=3000)

NGEN=1000
for gen in range(NGEN):
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))
    print(var_best)
top10 = tools.selBest(population, k=10)
print(var_best)
print(top10)
