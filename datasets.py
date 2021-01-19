'''
Functions to load training & evaluation datasets.
'''
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

data_dir = 'data/'


def get_training_data(data_type, resolution, train_data_file_name):
    # read normalised data in npy
    norm_training_data = np.load(data_dir + train_data_file_name + '_%s_%s.npy' % (data_type, resolution))
    return norm_training_data


def get_evaluation_data(data_type, resolution, evaluation_data_file_name):
    # read normalised data in npy
    norm_evaluation_data = np.load(data_dir + evaluation_data_file_name + '_%s_%s.npy' % (data_type, resolution))
    return norm_evaluation_data


