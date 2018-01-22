import numpy as np
import time
import multiprocessing as mp
import os, gc
import argparse
from threading import RLock
FLAGS = None
users_choices = None
dataset = None
folder_name = None

def long_range(start, stop):
   i = start
   while i < stop:
       yield i
       i += 1


def get_sum_variance(indeces, dataset):
    shape = np.shape(dataset)
    sum_a = np.zeros(shape[2])
    for i in range(shape[0]):
        sum_a = np.add(sum_a, dataset[i, indeces[i], :])

    var = np.var(sum_a, ddof=1)
    return var


def index_job(i):
    index = np.unravel_index(i, users_choices)
    var = get_sum_variance(index, dataset)
    index_s = str(index)
    var_s = str(var)
    result = "".join((index_s, "::",var_s, "\n" ))
    return result, var


def range_job(bounds):
    #result = ""
    with open(folder_name+str(bounds[0])+"-"+str(bounds[1])+".txt", 'a') as appendFile:
        bvar = None
        for i in range(bounds[0], bounds[1]):
            res, var = index_job(i)
            bvar = bvar or var
            bvar = var if var < bvar else bvar
            appendFile.write(res)
        print(bvar)


def ranges(start, end, number):
    N = end - start
    step = N // number
    return list((round(step*i)+start, round(step*(i+1)+start)) for i in range(number))


def calc_user_choices(ds):
    shape = np.shape(dataset)
    result = np.array(list(0 for i in range(0, shape[0], 1)))

    for user in range(shape[0]):
        for plan in range(shape[1]):
            c_plan = dataset[user, plan, :]
            #any_non_zeroes = np.any(c_plan)
            all_non_nans = not np.any(np.isnan(c_plan))
            all_finite = not np.any(np.isinf(c_plan))

            if all_non_nans and all_finite:
                result[user] +=1

    return result


def exec_exhaustive(ds, output_folder):

    global dataset
    dataset = ds

    print("dataset shape: " + str(np.shape(dataset)))

    global users_choices
    users_choices = calc_user_choices(dataset)

    print("plans per user: " + str(users_choices))

    global folder_name
    folder_name = output_folder

    start = time.time()

    pool = mp.Pool()

    range_start = 0
    #range_end = plans ** users
    range_end = np.prod(np.array(users_choices))

    print("total combinations: " + str(range_end))

    rs = ranges(range_start, range_end, mp.cpu_count())
    print("combinations for threads: " + str(rs))
    start = time.time()
    pool.map(range_job, ([i[0], i[1]] for i in rs))
    end = time.time()
    diff = end - start
    print("total execution time: " + str(diff))

def main(dataset_file):
    ds = np.load(dataset_file)
    #dataset = np.random.rand(10,10,144)
    name = os.path.basename(dataset_file)
    output_folder = "./"+name.replace(".npy","") +"/"
    if not os.path.exists("./"+folder_name):
        os.makedirs("./"+folder_name)

    exec_exhaustive(ds, output_folder)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset_file", type=str, default="",
                        help="file containing the dataset in an npy format")
    FLAGS = parser.parse_args()
    main(FLAGS.dataset_file)
