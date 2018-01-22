import numpy as np
import reinforced_epos.helpers.config as cf
import reinforced_epos.helpers.dataset as ds
from tensorboard.main import main

#import json
from subprocess import call
import os

def execute_iepos(dataset=None):
    max_iterations = cf.IEPOS_ITER
    children_node = cf.IEPOS_CHILDREN
    lamda = cf.IEPOS_LAMBDA
    seed = cf.IEPOS_SEED
    if(dataset is None):
        dataset = ds.get_dataset(normalize=False)
    print(np.shape(dataset))
    experiment_epos_folder = cf.get_experiement_folder()
    path_dataset = experiment_epos_folder + "numpy_dataset.npy"
    jar = os.path.abspath("../../bin/iepos.jar")
    call(["java", "-Xmx10g", "-jar", jar,  experiment_epos_folder, str(cf.IEPOS_CHILDREN), str(cf.IEPOS_ITER), str(cf.IEPOS_LAMBDA), str(cf.IEPOS_SEED), str(cf.IEPOS_ALGO)], cwd="../../bin/")

execute_iepos()
