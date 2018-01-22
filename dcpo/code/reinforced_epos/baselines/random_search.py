import numpy as np
import time
import multiprocessing as mp
import os, gc
import argparse
from random import shuffle
from reinforced_epos.helpers.dataset import get_dataset
import sys
from threading import RLock
FLAGS = None
users_choices = None
dataset = None


def get_sum_variance(indeces, dataset):
    shape = np.shape(dataset)
    sum_a = np.zeros(shape[2])
    for i in range(shape[0]):
        sum_a = np.add(sum_a, dataset[i, indeces[i], :])
    var = np.var(sum_a, ddof=1)
    return var





def exec_exhaustive(ds, total):

    global dataset
    dataset = ds
    shape =np.shape(ds)

    # print("dataset shape: " + str(np.shape(dataset)))



    sample = distinct_random(sample_size=10000, total_plans=shape[1], users=shape[0]).values()

    best_var = None

    # print("Sample size: " + str(len(sample)))

    for index in sample:
        var = get_sum_variance(index, dataset)
        best_var = best_var or var
        best_var = var if var < best_var else best_var

    print(best_var)



def distinct_random(sample_size, users, total_plans):
    done = True
    collec = dict()
    while done:
        sample = np.random.random_integers(0, total_plans-1, users)

        collec[hash(sample.tostring())] = sample
        if(len(collec) == sample_size):
            break
    return collec

def main(dataset_file, sample_size):
    ds = None
    if(dataset_file is None):
        ds = np.load(dataset_file)
    else:
        ds = get_dataset()
    #dataset = np.random.rand(10,10,144)
    for i in range(5):
        exec_exhaustive(ds, sample_size)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset_file", type=str, default="",
                        help="file containing the dataset in an npy format")
    FLAGS = parser.parse_args()
    main(FLAGS.dataset_file, 20000)
