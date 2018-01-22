import reinforced_epos.helpers.config as cf
import reinforced_epos.helpers.reader as rdr
import pandas as pd
import numpy as np
import tensorflow as tf
import scipy.stats as stats
import scipy.signal as signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shutil

import mpld3 #send plots to browser
import os


def get_raw_data():
    '''
    An initialization method.
    Loads the datas/home/thomaset fromt he config folder. It splits files based
    on the provided EPOS format which is for a line:
    pref_weight(float32):consumption_value_1(float32),...,consumption_value144(float32)
    Uses pandas and numpy for intermediate storage.
    Possible memory inefficiency and extra processing
    :return: A numpy array of the following shape: (plans, time, agents)
    '''
    print("Loading agent data from folder: " + cf.DATA_FOLDER)
    data_paths = [file for file in rdr.walk_plans_dataset(cf.DATA_FOLDER, sort_asc=cf.AGENT_SORTING, shuffle_seed=cf.AGENT_SHUFFLE_SEED)]
    agent_arrays = []
    i = 0

    max_plan = None
    max_timesteps = None

    for path in data_paths:
        #print("Now loading: " + path)
        frame = pd.read_csv(path, skiprows=0, header=None, #usecols = range(1, num_dimensions[2] + 1), names= col_names,
                         sep=',|:', engine='python')
        frame.drop(0, axis = 1, inplace = True)
        np_array = frame.as_matrix()
        shape = np.shape(np_array)
        max_plan = max_plan or shape[0]
        max_plan = shape[0] if shape[0] > max_plan else max_plan
        max_timesteps = max_timesteps or shape[1]
        max_timesteps = shape[1] if shape[1] > max_timesteps else max_timesteps
        #print(currentShape)
        agent_arrays.append(np_array)
        i = i + 1
    raw_data = np.empty((len(data_paths), max_plan, max_timesteps))

    np.set_printoptions(threshold=np.nan)
    for agent in range(len(agent_arrays)):
        plans =  agent_arrays[agent]
        shape_plans = np.shape(plans)
        raw_data[agent, :shape_plans[0], :shape[1]] = plans
        #print(plans)
        #for plan_index in range(shape_plans[0]):
            #raw_data[agent, plan_index, :] = plans[plan_index]

    #print(raw_data)
    #raw_data = np.array(agent_arrays, ndmin=3)

    return raw_data


def crop_data(data):
    '''
    Crops the data array, to the size of defined in the config file
    :param data: the raw data after all possible sortings
    :return: the array cropped from 0 to crop-size on all dimensions
    '''
    current_shape = np.shape(data)
    requested_shape = cf.AFTER_SORT_MASK
    print("shape before crop: " + str(current_shape))
    if(requested_shape is not None):
        user_crop = current_shape[0] if current_shape[0] < requested_shape[0] else requested_shape[0]
        plan_crop = current_shape[1] if current_shape[1] < requested_shape[1] else requested_shape[1]
        timestep_crop = current_shape[2] if current_shape[2]<requested_shape[2] else requested_shape[2]
        data = data[0:user_crop, 0:plan_crop, 0:timestep_crop]
    current_shape = np.shape(data)
    print("shape after crop: " + str(current_shape))

    return data


def normalization_rescaling(data):
    '''
    This is max_min rescaler, which brings all the data to range [0,1]
    :param data: input data
    :return: normalized data
    '''
    max = np.max(data)
    min = np.min(data)
    print(np.shape(max))
    return (data - min )/(max-min)


def sort_data_plans(raw_data):
    '''
    sorts a raw dataset
    :param raw_data: a dataset of shape (plans, timesteps, users)
    :return:
    '''
    print("is used")
    for user in range(np.shape(raw_data)[2]):
        raw_data[:, :, user] = sort_on_user(raw_data[:, :, user])


def sort_on_user(np_array):
    '''
    a user of shape (plans, timesteps)
    :param np_array:
    :return:
    '''
    criterion = eval_stats_row(np_array)
    indeces = np.argsort(criterion, axis = None).tolist()
    return np_array[indeces, :]


def eval_stats_row(np_array):
    '''
    calculates rowise stats over an np array. removes other dimensions
    :param np_array:
    :return:
    '''
    mean = np.mean(np_array, axis = 1)
    median = np.median(np_array, axis = 1)
    max = np.max(np_array, axis = 1)
    variance = np.var(np_array, axis = 1)
    iqr = stats.iqr(np_array, axis = 1)
    mode = stats.mode(np_array, axis = 1)
    kurtosis = stats.kurtosis(np_array, axis = 1)
    skewness = stats.skew(np_array, axis = 1)
    # all the above wont work for sorting. Turns out the dataset is synthetic
    # and has a very special property
    f, psd = signal.periodogram(np_array, axis = 1)
    psd_mean = psd.mean(axis = 1)
    psd_max = psd.max(axis = 1)
    psd_min = psd.min(axis = 1)
    psd_kurtosis = stats.skew(psd, axis = 1)
    psd_skewness = stats.skew(psd, axis = 1)
    psd_mode = stats.mode(psd, axis = 1)
    psd_iqr = stats.iqr(psd, axis = 1)
    psd_var = psd.var(axis = 1)
    return eval(cf.SORTING_CRIT)


def plot_user_sep(user_data, title=None, smoothing = 4):
    '''
    for a user vector of shape (plans, timesteps)
    generate a plot over time for each plan. If a title is provided
    the value of the sorting criterion will be included in it
    :param user_data:
    :param title:
    :param smoothing: size of movind average smoothing
    :return:
    '''
    fig, pltgrid = plt.subplots(5, 2, figsize=(15,15))
    fig.suptitle(title or "title", fontsize=20)
    criterion = eval_stats_row(user_data)
    for i in range(np.shape(user_data)[0]):
        pltgrid[i//2, i%2].plot(moving_average(user_data[i,:], smoothing))
        row = i//2
        col = i%2
        title2 = str(row) + "," + str(col) + " "
        if title is not None:
            title2 = title2 + title + " : " + str(criterion[i])
        pltgrid[i//2, i%2].set_title(title2)
        #pltgrid[i//2, i%2].set_title(cf.SORTING_CRIT + " " + str(criterion[indeces[i]]))
    fig.subplots_adjust(hspace=1.3)
    mpld3.show()


def plot_user_joint(user_data, smoothing = 4):
    '''
    All plans of a user matrix of shape (plans, timesteps)
    plotted over time in the same plot
    :param user_data:
    :param smoothing: size of movind average smoothing
    :return:
    '''
    print(np.shape(user_data))
    for i in range(np.shape(user_data)[0]):
        plt.plot(moving_average(user_data[i, :], smoothing))
        #plt.set_title(cf.SORTING_CRIT + " " + str(criterion[indeces[i]]))
    mpld3.show()


def moving_average(a, n=3) :
    '''
    To smooth the time series
    :param a:
    :param n:
    :return:
    '''
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def get_dataset(normalize=True):
    result = None
    exp_folder = cf.get_experiement_folder()
    exp_dataset = os.path.join(exp_folder, "numpy_dataset.npy")
    if os.path.exists(exp_folder) and os.path.exists(exp_dataset):
        print("loading existing dataset")
        result = np.load(exp_dataset)
    else:
        print("creating new dataset based on conf")
        shutil.copy2(cf.config_path, os.path.join(exp_folder,"config"))
        result = get_raw_data()
        result = crop_data(result)
        if normalize:
            result = normalization_rescaling(result)
        np.save(exp_dataset, result)
    return result


if __name__ == '__main__':
    raw_data = get_dataset()
    print(np.shape(raw_data))
    #plot_user_joint(raw_data[:,:,1])
    #plot_user_sep(raw_data[:,:,1])

    print(np.shape(sort_data_plans(raw_data)))
    #plot_user_joint(sort_on_data(raw_data[1,:,:]))
    plot_user_sep(raw_data[1,:,:], cf.SORTING_CRIT)
