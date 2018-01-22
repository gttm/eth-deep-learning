import json
import os, errno, inspect
import time
from pathlib import Path



__author__ = 'Thomas Asikis'
__institute__ = 'ETH Zurich'
__department__ = "Computational Social Science"
config_path = "config"
curr_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
curr_path = os.path.join(os.path.dirname(curr_path), os.pardir)
curr_path = os.path.join(curr_path, os.pardir)
curr_path = os.path.abspath(curr_path)
config_path = os.path.join(curr_path, config_path)

current_milli_time = lambda: int(round(time.time() * 1000))

def __init():
    """
    Loads the paths to the folders that contain the data and also the results
    :return:
    """



    config = json.load(open(config_path, 'r'))
    df = __abs_path(os.path.relpath(config['dataFolder']))
    of = __abs_path(os.path.relpath(config['outputFolder']))
    sc = config['sortingCriterion']
    maskDataAfterSort = config['maskDataAfterSort']
    agentSorting = eval(config['agentSorting'])
    agentShuffleSeed = eval(config['agentShuffleSeed'])
    experiment = config['experiment']

    return df, of, sc, maskDataAfterSort, agentSorting, agentShuffleSeed, experiment


def __prepare_i_epos():
        config = json.load(open(config_path, 'r'))
        iepos = config['i-epos']
        iepos_iter= iepos['numIterations']
        iepos_children = iepos['childrenPerNode']
        iepos_seed = iepos['seed']
        iepos_lambda = iepos['lambda']
        iepos_global = iepos['globalCost']
        iepos_algorithm = iepos['algorithm']
        return iepos_iter, iepos_children, iepos_seed, iepos_lambda, iepos_global, iepos_algorithm

def __abs_path(path):
    """
    Creates an absolute path, based on the relative path from the configuration file
    :param path: A relative path
    :return: The absolute path, based on the configuration file
    """
    if not os.path.isabs(path):
        parent = os.path.abspath(os.path.join(config_path, os.pardir))
        return os.path.abspath(os.path.join(os.path.relpath(parent), path)) + os.path.sep
    else:
        return path


DATA_FOLDER, OUTPUT_FOLDER, SORTING_CRIT, AFTER_SORT_MASK, AGENT_SORTING, AGENT_SHUFFLE_SEED, EXPERIMENT = __init()
IEPOS_ITER, IEPOS_CHILDREN, IEPOS_SEED, IEPOS_LAMBDA, IEPOS_GLOBAL, IEPOS_ALGO = __prepare_i_epos()
print(OUTPUT_FOLDER)
IEPOS_DIR = OUTPUT_FOLDER+"iepos"+os.path.sep #jar location
EXPERIEMENT_DIR = OUTPUT_FOLDER+"experiments"+os.path.sep
utc_version = str(current_milli_time())

def get_experiement_folder():
    if EXPERIMENT is None :
        name = "experiment"+utc_version + os.path.sep
    else:
        name = EXPERIMENT + os.path.sep

    current_experienment_folder = EXPERIEMENT_DIR + name
    if not os.path.exists(current_experienment_folder) :
        try:
             os.makedirs(current_experienment_folder)
             print(current_experienment_folder)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise e
    return current_experienment_folder




# super dirty test passed
# print(DATA_FOLDER)
# /home/thomas/Repos/repos/data/energy
