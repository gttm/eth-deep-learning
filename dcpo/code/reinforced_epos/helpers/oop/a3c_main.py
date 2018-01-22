import reinforced_epos.helpers.dataset as ds
import tensorflow as tf
import os
import threading
import numpy as np
import multiprocessing
from reinforced_epos.helpers.oop.Network import Network
from reinforced_epos.helpers.oop.Worker import Worker
from reinforced_epos.helpers.oop.Environment import Environment
from reinforced_epos.helpers.config import get_experiement_folder
import shutil
from time import sleep

dataset = ds.get_dataset(normalize=False)

print("ds shape " + str(np.shape(dataset)))
print("initializing stuff")

max_episode_length = 300
gamma = .8 # discount rate for advantage estimation and reward discounting
total_actions = 5 # total actions, agent can move plan index left, right and stay in position

load_model = False
folder = get_experiement_folder()
a3cfolder = os.path.join(folder, "a3c")
if os.path.exists(a3cfolder):
    shutil.rmtree(a3cfolder)
model_path = os.path.join(a3cfolder, "model")


tf.reset_default_graph()

if not os.path.exists(model_path):
    os.makedirs(model_path)

with tf.device("/cpu:0"):
    global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)

with tf.device("/gpu:0"):
    trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
    master_network = Network(dataset, total_actions,'global', None) # Generate global network

    print("generated master network")

    # num_workers = multiprocessing.cpu_count() # Set workers to number of available CPU threads
    num_workers = 4
    print("creating workers")

    workers = []
    # Create worker classes
    for i in range(num_workers):
        env = Environment(dataset, 0, 0, total_actions)
        worker = Worker("worker_" + str(i), dataset, env, total_actions, trainer, model_path, global_episodes)
        workers.append(worker)
        print("worker created")
with tf.device("/cpu:0"):
    saver = tf.train.Saver(max_to_keep=5)

    print("workers created")

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    print("coordinator initialized")

    if load_model == True:
        print ('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess,ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    # This is where the asynchronous magic happens.
    # Start the "work" process for each worker in a separate threat.
    worker_threads = []
    for worker in workers:
        print("workers training")
        worker_work = lambda: worker.work(max_episode_length,gamma,sess,coord,saver)
        t = threading.Thread(target=(worker_work))
        t.start()
        sleep(0.5)
        worker_threads.append(t)
    coord.join(worker_threads)
