import os
import argparse
import numpy as np
import tensorflow as tf
import random
from pyswarm import pso
import reinforced_epos.helpers.dataset as ds
from reinforced_epos.helpers.oop.Environment import Environment

# We have 100 agents with 4 plans, each plan has values for 100 time steps
PATH = "experiement_gaussian_4/numpy_dataset.npy"

dataset = ds.get_dataset(normalize=False)
env = Environment(dataset, 1, 1)
shape = np.shape(dataset)
print(shape)

num_iter = 2000

# x = [random.randint(0, shape[1]-1) for _ in range(shape[0])]

def con(x):
    return x

def var(x):
    x = np.round(x)
    x = np.clip(x, 0, shape[1]-1)
    x = np.ndarray.astype(x, np.int32)
    '''
    x = np.squeeze(x)
    print(np.shape(np.array(x)))
    print(np.max(x))
    :param x:
    :return:
    '''

    return env.state_variance(x)[0]

lb = [0 for _ in range(shape[0])]   # lower bound of plans
ub = [shape[1] - 1 for _ in range(shape[0])]   # upper bound of plans



xopt, fopt = pso(var, lb, ub, f_ieqcons=con, phip = 0.7, phig= 0.3, swarmsize= 1000, maxiter = 1000) # returns the best plan combination and its variance


# Print the plan and validate its var
print(fopt) #optimal variance
xopt = xopt.astype(int)
print(xopt) #optimal plan
total_energy = np.sum((dataset[i, j, :] for i, j in zip(range(shape[2]), xopt.astype(int))), axis=0) #add the 100 time-step values for each agent's plan

