import numpy as np
import pickle
import os
import csv
from skimage.color import rgb2gray
from scipy.io import loadmat
import pandas as pd
import random
from random import randrange
from scipy.special import expit
from scipy import signal as sg
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from skimage.util import view_as_blocks, view_as_windows
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import json
from io import BytesIO
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import threading
import time
from pathlib import Path
from typing import Iterator, Dict
from tokenizers import Tokenizer
from datasets import load_dataset
from huggingface_hub import login


import os
import math
import random
import zipfile
import tarfile
import urllib.request
from random import randrange

import numpy as np
from PIL import Image


#gaussian2d
class EllipticGaussianGenerator:

    @staticmethod
    def default_params():
        return {'mean': [2.0, -1.0], 'std_major': 2.0, 'std_minor': 0.5,
                'rotation_deg': 30.0, 'noise_freq': 0.0}

    def __init__(self, hparams):
        self.name = 'gaussian2d'
        self.num_dims = 2
        self.num_channels = 1

        # non-zero mean
        self.mean = np.array(hparams['mean'], dtype=np.float64)
        assert self.mean.shape == (2,)

        # elliptic covariance: two different axis variances + rotation
        self.std_major = hparams['std_major']
        self.std_minor = hparams['std_minor']
        self.rotation_deg = hparams['rotation_deg']

        theta = np.deg2rad(self.rotation_deg)
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta),  np.cos(theta)]])
        diag = np.diag([self.std_major ** 2, self.std_minor ** 2])
        self.covariance = rot @ diag @ rot.T   # elliptic (anisotropic, correlated)

        self.noise_freq = hparams['noise_freq']

        random.seed(420)
        np.random.seed(420)

        self.current_sample = np.random.multivariate_normal(self.mean, self.covariance)

    def _sample(self):
        self.current_sample = np.random.multivariate_normal(self.mean, self.covariance)

    def set_random_position(self):
        self._sample()

    def set_test_position(self):
        self._sample()

    def set_validation_position(self):
        self._sample()

    def set_next_position(self):
        self._sample()

    def get_current_frame(self):
        frame = self.current_sample.copy()
        if self.noise_freq > 0:
            a = np.random.rand(self.num_dims)
            frame = frame + a * (a < self.noise_freq)
        return frame

    def get_blind_frame(self):
        return np.zeros(self.num_dims)

    def get_label_for(self, label_category):
        match label_category:
            case 'category_string':
                # quadrant relative to the mean, e.g. "+_-"
                dx = '+' if self.current_sample[0] >= self.mean[0] else '-'
                dy = '+' if self.current_sample[1] >= self.mean[1] else '-'
                return dx + '_' + dy
            case 'category_one_hot':
                # one-hot over the 4 quadrants relative to the mean
                label = np.zeros(4)
                idx = 2 * int(self.current_sample[0] >= self.mean[0]) \
                        + int(self.current_sample[1] >= self.mean[1])
                label[idx] = 1.0
                return label
            case 'position_x':
                return np.array([self.current_sample[0]])
            case 'position_y':
                return np.array([self.current_sample[1]])
            case 'deviation':
                # deviation from the mean
                return self.current_sample - self.mean
            case 'mahalanobis':
                diff = self.current_sample - self.mean
                d2 = diff @ np.linalg.inv(self.covariance) @ diff
                return np.array([np.sqrt(d2)])
            case _:
                return None

    def get_all_labels(self):
        label_obj = dict()
        dx = '+' if self.current_sample[0] >= self.mean[0] else '-'
        dy = '+' if self.current_sample[1] >= self.mean[1] else '-'
        label_obj['category_string'] = dx + '_' + dy
        label_obj['category_one_hot'] = np.zeros(4)
        idx = 2 * int(self.current_sample[0] >= self.mean[0]) \
                + int(self.current_sample[1] >= self.mean[1])
        label_obj['category_one_hot'][idx] = 1.0
        label_obj['position_x'] = np.array([self.current_sample[0]])
        label_obj['position_y'] = np.array([self.current_sample[1]])
        label_obj['deviation'] = self.current_sample - self.mean
        diff = self.current_sample - self.mean
        d2 = diff @ np.linalg.inv(self.covariance) @ diff
        label_obj['mahalanobis'] = np.array([np.sqrt(d2)])
        return label_obj

    def get_label_list(self):
        return ["category_string", "category_one_hot", "position_x",
                "position_y", "deviation", "mahalanobis"]

    def get_action_label(self):
        # direction of the deviation from the mean (normalized)
        diff = self.current_sample - self.mean
        norm = np.linalg.norm(diff)
        if norm == 0:
            return np.zeros(self.num_dims)
        return diff / norm

    def get_hyperparameters(self):
        hyperparameters = dict()
        hyperparameters['vg_name'] = self.name
        hyperparameters['mean'] = self.mean.tolist()
        hyperparameters['std_major'] = self.std_major
        hyperparameters['std_minor'] = self.std_minor
        hyperparameters['rotation_deg'] = self.rotation_deg
        hyperparameters['noise'] = self.noise_freq
        return hyperparameters

    def get_extensive_name(self):
        return self.name + '_' + str(self.mean[0]) + '_' + str(self.mean[1]) \
            + '_' + str(self.std_major) + '_' + str(self.std_minor) \
            + '_' + str(self.rotation_deg)

    def get_shape(self):
        return (self.num_dims, self.num_channels)

    def get_category_string_from_one_hot(self, one_hot):
        idx = int(np.argmax(one_hot))
        dx = '+' if idx // 2 == 1 else '-'
        dy = '+' if idx % 2 == 1 else '-'
        return dx + '_' + dy

    def get_name(self):
        return self.name

#crossbar
class MultipleSpeedCrossingBar:

    @staticmethod
    def default_params():
        return {'screen_size': 12, 'bar_size': 1, 'max_bar_speed': 1, 'noise_freq': 0.1}
    
    def __init__(self, hparams):
        self.name = 'crossbar'
        self.x_size = hparams['screen_size']
        self.y_size = hparams['screen_size']
        self.num_channels = 1
        self.bar_size = hparams['bar_size']

        self.speed_max = hparams['max_bar_speed']


        self.possible_directions = list()
        for dir_idx in range(-self.speed_max, self.speed_max+1):
            if dir_idx != 0:
                self.possible_directions.append(dir_idx)
        
        
        self.current_x_bar_position = randrange(self.x_size)
        self.current_y_bar_position = randrange(self.y_size)
        self.current_x_bar_direction = random.choice(self.possible_directions)
        self.current_y_bar_direction = random.choice(self.possible_directions)

        random.seed(420)
        
        self.noise_freq =  hparams['noise_freq']
    
    def set_random_position(self):
        self.current_x_bar_position = randrange(self.x_size)
        self.current_y_bar_position = randrange(self.y_size)
        self.current_x_bar_direction = random.choice(self.possible_directions)
        self.current_y_bar_direction = random.choice(self.possible_directions)

    def set_test_position(self):
        self.current_x_bar_position = randrange(self.x_size)
        self.current_y_bar_position = randrange(self.y_size)
        self.current_x_bar_direction = random.choice(self.possible_directions)
        self.current_y_bar_direction = random.choice(self.possible_directions)

    def set_validation_position(self):
        self.current_x_bar_position = randrange(self.x_size)
        self.current_y_bar_position = randrange(self.y_size)
        self.current_x_bar_direction = random.choice(self.possible_directions)
        self.current_y_bar_direction = random.choice(self.possible_directions)
    
    def set_next_position(self):
        self.current_x_bar_position = (self.current_x_bar_position + self.current_x_bar_direction) % self.y_size
        self.current_y_bar_position = (self.current_y_bar_position + self.current_y_bar_direction) % self.x_size
    
    def get_current_frame(self):
        a = np.random.rand(self.y_size, self.x_size)
        frame = a*(a < self.noise_freq)

        #draw x bars
        for i in range(0,self.y_size):
            for j in range(0, self.bar_size):
                if (self.current_x_bar_position + j) < self.x_size:
                    frame[i][self.current_x_bar_position + j] = 1
        
        #draw y bar
        for i in range(0,self.x_size):
            for j in range(0, self.bar_size):
                if (self.current_y_bar_position + j) < self.y_size:
                    frame[self.current_y_bar_position + j][i] = 1
        
        return frame
    
    def get_blind_frame(self):
        frame = np.zeros((self.y_size, self.x_size))
        return frame
    
    def get_label_for(self, label_category):
        match label_category:
            case 'category_string':
                return str(self.current_x_bar_position) + '_' + str(self.current_y_bar_position)
            case 'category_one_hot':
                label = np.zeros(self.x_size * self.y_size)
                label[self.current_x_bar_position * self.y_size + self.current_y_bar_position] = 1
                return label
            case 'position_x':
                label_pos_x = np.zeros(self.y_size)
                label_pos_x[self.current_x_bar_position] = 1.0
                return label_pos_x
            case 'position_y':
                label_pos_y = np.zeros(self.x_size)
                label_pos_y[self.current_y_bar_position] = 1.0
                return label_pos_y
            case 'direction_x':
                label_dir_x = np.zeros(len(self.possible_directions))
                label_dir_x[self.current_x_bar_direction] = 1.0
                return label_dir_x
            case 'direction_y':
                label_dir_y = np.zeros(len(self.possible_directions))
                label_dir_y[self.current_y_bar_direction] = 1.0
                return label_dir_y
            case _:
                return None
    
    def get_all_labels(self):
        label_obj = dict()
        label_obj['category_string'] = str(self.current_x_bar_position) + '_' + str(self.current_y_bar_position)
        label_obj['category_one_hot'] = np.zeros(self.x_size * self.y_size)
        label_obj['category_one_hot'][self.current_x_bar_position * self.y_size + self.current_y_bar_position] = 1
        label_obj['position_x'] = np.zeros(self.y_size)
        label_obj['position_x'][self.current_x_bar_position] = 1.0
        label_obj['position_y'] = np.zeros(self.x_size)
        label_obj['position_y'][self.current_y_bar_position] = 1.0
        label_obj['direction_x'] = np.zeros(len(self.possible_directions))
        label_obj['direction_x'][self.current_x_bar_direction] = 1.0
        label_obj['direction_y'] = np.zeros(len(self.possible_directions))
        label_obj['direction_y'][self.current_y_bar_direction] = 1.0
        return label_obj
        
    def get_label_list(self):
        return ["category_string", "category_one_hot", "position_x", "position_y", "direction_x", "direction_y"]
    
    def get_action_label(self):
        dir_x = np.zeros(len(self.possible_directions))
        dir_y = np.zeros(len(self.possible_directions))
        dir_x[self.current_x_bar_direction] = 1.0
        dir_y[self.current_y_bar_direction] = 1.0
        return np.concatenate((dir_x, dir_y))
    
    def get_hyperparameters(self):
        hyperparameters = dict()
        hyperparameters['vg_name'] = self.name
        hyperparameters['screen_size'] = self.x_size
        hyperparameters['bar_size'] = self.bar_size
        hyperparameters['speed'] = self.speed_max
        hyperparameters['noise'] = self.noise_freq
        return hyperparameters

    def get_extensive_name(self):
        return self.name + '_' + str(self.bar_size) + '_' + str(self.speed_max) + '_' + str(self.noise_freq)

    def get_shape(self):
        return (self.x_size, self.y_size, self.num_channels)
    
    def get_category_string_from_one_hot(self, one_hot):
        index_one_hot = np.argmax(one_hot)
        return str(int(index_one_hot//self.y_size)) + '_' + str(index_one_hot % self.y_size)

    def get_name(self):
        return self.name

#crossbar_zero_mean
class MultipleSpeedCrossingBarZeroMean:

    @staticmethod
    def default_params():
        return {'screen_size': 12, 'bar_size': 1, 'max_bar_speed': 1, 'noise_freq': 0.1}
    
    def __init__(self, hparams):
        self.name = 'crossbar_zero_mean'
        self.x_size = hparams['screen_size']
        self.y_size = hparams['screen_size']
        self.num_channels = 1
        self.bar_size = hparams['bar_size']
        self.noise_freq =  hparams['noise_freq']
        self.speed_max = hparams['max_bar_speed']


        self.possible_directions = list()
        for dir_idx in range(-self.speed_max, self.speed_max+1):
            if dir_idx != 0:
                self.possible_directions.append(dir_idx)
        
        self.current_x_bar_position = randrange(self.x_size)
        self.current_y_bar_position = randrange(self.y_size)
        self.current_x_bar_direction = random.choice(self.possible_directions)
        self.current_y_bar_direction = random.choice(self.possible_directions)

        random.seed(420)

        self.mean = np.zeros((self.y_size, self.x_size))
        for ex_idx in range(3000):
            self.set_random_position()
            self.mean += self.get_current_frame()
        self.mean /= 3000
        
    
    def set_random_position(self):
        self.current_x_bar_position = randrange(self.x_size)
        self.current_y_bar_position = randrange(self.y_size)
        self.current_x_bar_direction = random.choice(self.possible_directions)
        self.current_y_bar_direction = random.choice(self.possible_directions)

    def set_test_position(self):
        self.current_x_bar_position = randrange(self.x_size)
        self.current_y_bar_position = randrange(self.y_size)
        self.current_x_bar_direction = random.choice(self.possible_directions)
        self.current_y_bar_direction = random.choice(self.possible_directions)

    def set_validation_position(self):
        self.current_x_bar_position = randrange(self.x_size)
        self.current_y_bar_position = randrange(self.y_size)
        self.current_x_bar_direction = random.choice(self.possible_directions)
        self.current_y_bar_direction = random.choice(self.possible_directions)
    
    def set_next_position(self):
        self.current_x_bar_position = (self.current_x_bar_position + self.current_x_bar_direction) % self.y_size
        self.current_y_bar_position = (self.current_y_bar_position + self.current_y_bar_direction) % self.x_size
    
    def get_current_frame(self):
        a = np.random.randn(self.y_size, self.x_size)
        frame = a*(np.abs(a) < self.noise_freq)

        #draw x bars
        for i in range(0,self.y_size):
            for j in range(0, self.bar_size):
                if (self.current_x_bar_position + j) < self.x_size:
                    frame[i][self.current_x_bar_position + j] = 1
        
        #draw y bar
        for i in range(0,self.x_size):
            for j in range(0, self.bar_size):
                if (self.current_y_bar_position + j) < self.y_size:
                    frame[self.current_y_bar_position + j][i] = 1
        
        return frame - self.mean
    
    def get_blind_frame(self):
        frame = np.zeros((self.y_size, self.x_size))
        return frame
    
    def get_label_for(self, label_category):
        match label_category:
            case 'category_string':
                return str(self.current_x_bar_position) + '_' + str(self.current_y_bar_position)
            case 'category_one_hot':
                label = np.zeros(self.x_size * self.y_size)
                label[self.current_x_bar_position * self.y_size + self.current_y_bar_position] = 1
                return label
            case 'position_x':
                label_pos_x = np.zeros(self.y_size)
                label_pos_x[self.current_x_bar_position] = 1.0
                return label_pos_x
            case 'position_y':
                label_pos_y = np.zeros(self.x_size)
                label_pos_y[self.current_y_bar_position] = 1.0
                return label_pos_y
            case 'direction_x':
                label_dir_x = np.zeros(len(self.possible_directions))
                label_dir_x[self.current_x_bar_direction] = 1.0
                return label_dir_x
            case 'direction_y':
                label_dir_y = np.zeros(len(self.possible_directions))
                label_dir_y[self.current_y_bar_direction] = 1.0
                return label_dir_y
            case _:
                return None
    
    def get_all_labels(self):
        label_obj = dict()
        label_obj['category_string'] = str(self.current_x_bar_position) + '_' + str(self.current_y_bar_position)
        label_obj['category_one_hot'] = np.zeros(self.x_size * self.y_size)
        label_obj['category_one_hot'][self.current_x_bar_position * self.y_size + self.current_y_bar_position] = 1
        label_obj['position_x'] = np.zeros(self.y_size)
        label_obj['position_x'][self.current_x_bar_position] = 1.0
        label_obj['position_y'] = np.zeros(self.x_size)
        label_obj['position_y'][self.current_y_bar_position] = 1.0
        label_obj['direction_x'] = np.zeros(len(self.possible_directions))
        label_obj['direction_x'][self.current_x_bar_direction] = 1.0
        label_obj['direction_y'] = np.zeros(len(self.possible_directions))
        label_obj['direction_y'][self.current_y_bar_direction] = 1.0
        return label_obj
        
    def get_label_list(self):
        return ["category_string", "category_one_hot", "position_x", "position_y", "direction_x", "direction_y"]
    
    def get_action_label(self):
        dir_x = np.zeros(len(self.possible_directions))
        dir_y = np.zeros(len(self.possible_directions))
        dir_x[self.current_x_bar_direction] = 1.0
        dir_y[self.current_y_bar_direction] = 1.0
        return np.concatenate((dir_x, dir_y))
    
    def get_hyperparameters(self):
        hyperparameters = dict()
        hyperparameters['vg_name'] = self.name
        hyperparameters['screen_size'] = self.x_size
        hyperparameters['bar_size'] = self.bar_size
        hyperparameters['speed'] = self.speed_max
        hyperparameters['noise'] = self.noise_freq
        return hyperparameters

    def get_extensive_name(self):
        return self.name + '_' + str(self.bar_size) + '_' + str(self.speed_max) + '_' + str(self.noise_freq)

    def get_shape(self):
        return (self.x_size, self.y_size, self.num_channels)
    
    def get_category_string_from_one_hot(self, one_hot):
        index_one_hot = np.argmax(one_hot)
        return str(int(index_one_hot//self.y_size)) + '_' + str(index_one_hot % self.y_size)

    def get_name(self):
        return self.name

#mnist
class MNIST:

    @staticmethod
    def get_input_shape():
        return (32, 32, 1)

    @staticmethod
    def default_params():
        return {'file': '/disk/scratch/hraguiar/samba_disk/Datasets/mnist/mnist_train.csv', 'samples': 42000, 'validation_set': 1000}
    
    def __init__(self, file, samples, validation_set):
        self.name = 'mnist'
        a = pd.read_csv(file)
        if samples > len(a):
            raise ValueError(
                f"Requested {samples} MNIST samples, but the dataset contains "
                f"only {len(a)} rows."
            )
        if samples <= validation_set:
            raise ValueError(
                f"samples ({samples}) must be greater than "
                f"validation_set ({validation_set})."
            )
        self.label_size = 10
        self.labels = a['label'].to_numpy()[:samples]
        a = a.drop(['label'], axis=1)
        self.x_size = 32
        self.y_size = 32
        self.num_channels = 1
        self.images = a.to_numpy()[:samples].reshape(samples, 28, 28) / 256
        self.validation_set = validation_set
        self.num_samples = samples
        self.cur_index = 0
        self.it = 0

    def get_current_frame(self):
        frame = np.zeros((32,32,1))
        frame[2:30, 2:30, 0] = self.images[self.cur_index]
        return frame
      
    def set_next_position(self):
        pass
    
    def set_random_position(self):
        self.cur_index = np.random.randint(self.validation_set, self.num_samples)

    def set_test_position(self):
        self.cur_index = np.random.randint(self.validation_set, self.num_samples)

    def set_validation_position(self):
        self.cur_index = np.random.randint(0, self.validation_set)
    
    def get_blind_frame(self):
        arrayy = np.zeros((32,32,1))
        return arrayy
    
    def get_label_for(self, label_category):
        match label_category:
            case 'category_string':
                return str(self.labels[self.cur_index])
            case 'category_one_hot':
                label = np.zeros(max(self.labels)+1)
                label[self.labels[self.cur_index]] = 1
                return label
            case _:
                return None
    
    def get_all_labels(self):
        label_obj = dict()
        label_obj['category_string'] = self.labels[self.cur_index]
        label_obj['category_one_hot'] = np.zeros(max(self.labels)+1)
        label_obj['category_one_hot'][self.labels[self.cur_index]] = 1
        return label_obj
        
    def get_label_list(self):
        return ["category_string", "category_one_hot"]

    def get_shape(self):
        return (32, 32, 1)

    def get_name(self):
        return self.name

    def get_category_string_from_one_hot(self, one_hot):
        index_one_hot = np.argmax(one_hot)
        return str(index_one_hot)

#moving_mnist **needs update
class MovingMNIST32:

    @staticmethod
    def get_input_shape():
        return (32, 32, 1)

    @staticmethod
    def default_params():
        return {'file': '/disk/scratch/hraguiar/Datasets/mnist/mnist_train.csv'}
    
    def __init__(self, file):
        self.name = 'moving_mnist'
        samples = 42000
        validation_set = 1000
        a = pd.read_csv(file)
        if samples > len(a):
            raise ValueError(
                f"Requested {samples} MNIST samples, but the dataset contains "
                f"only {len(a)} rows."
            )
        if samples <= validation_set:
            raise ValueError(
                f"samples ({samples}) must be greater than "
                f"validation_set ({validation_set})."
            )
        self.label_size = 10
        self.labels = a['label'].to_numpy()[:samples]
        a = a.drop(['label'], axis=1)
        self.x_size = 32
        self.y_size = 32
        self.num_channels = 1
        self.images = a.to_numpy()[:42000].reshape(42000,28,28)/256
        self.cur_index = 0
        self.x_offset = 0
        self.y_offset = 0
        self.x_dir = 1
        self.y_dir = 1
        self.noise_freq = 0.1
            
    def set_next_position(self):
        self.x_offset += self.x_dir 
        self.y_offset += self.y_dir

    def set_random_position(self):
        if self.cur_index < (len(self.images) - 1):
            self.cur_index += 1
        else:
            self.cur_index = 0
        
        self.x_offset = np.random.randint(-5, 10)
        self.y_offset = np.random.randint(-5, 10)
        
        if np.random.rand(1) > 0.5:
            self.x_dir = -1
        else:
            self.x_dir = 1

        if np.random.rand(1) > 0.5:
            self.y_dir = -1
        else:
            self.y_dir = 1
    
    def set_test_position(self):
        if self.cur_index < (len(self.images) - 1):
            self.cur_index += 1
        else:
            self.cur_index = 0
        
        self.x_offset = np.random.randint(-5, 10)
        self.y_offset = np.random.randint(-5, 10)
        
        if np.random.rand(1) > 0.5:
            self.x_dir = -1
        else:
            self.x_dir = 1

        if np.random.rand(1) > 0.5:
            self.y_dir = -1
        else:
            self.y_dir = 1

    def set_validation_position(self):
        self.set_random_position()
    
    def get_current_frame(self):
        a = np.random.rand(32, 32)
        frame_noise = a*(a < self.noise_freq)
        arrayy = np.zeros((32,32))
        #return the frame which should embeed the 28x28 image with a certain offset
        no_intersection = False
        relative_ax_index_left = self.x_offset
        relative_ax_index_right = relative_ax_index_left + 28
        if relative_ax_index_left < -28:
            no_intersection=True
        elif relative_ax_index_left <= 0:
            absolute_embeeded_x_index_left = 0
            absolute_embeeded_x_index_right = 28 + relative_ax_index_left
            crop_x_index_left = -relative_ax_index_left
            crop_x_index_right = 28
        elif relative_ax_index_left >= 32:
            no_intersection=True
        else:
            absolute_embeeded_x_index_left = relative_ax_index_left
            if relative_ax_index_right >= 32:
                absolute_embeeded_x_index_right = 32
                crop_x_index_left = 0
                crop_x_index_right = 28 - (relative_ax_index_right - 32)
            else:
                absolute_embeeded_x_index_right = relative_ax_index_right
                crop_x_index_left = 0
                crop_x_index_right = 28
        
        relative_ay_index_top = self.y_offset
        relative_ay_index_bottom = relative_ay_index_top + 28
        if relative_ay_index_top < -28:
            no_intersection=True
        elif relative_ay_index_top <= 0:
            absolute_embeeded_index_top = 0
            absolute_embeeded_index_bottom = 28 + relative_ay_index_top
            crop_y_index_top = -relative_ay_index_top
            crop_y_index_bottom = 28
        elif relative_ay_index_top >= 32:
            no_intersection=True
        else:
            absolute_embeeded_index_top = relative_ay_index_top
            if relative_ay_index_bottom >= 32:
                absolute_embeeded_index_bottom = 32
                crop_y_index_top = 0
                crop_y_index_bottom = 28 - (relative_ay_index_bottom - 32)
            else:
                absolute_embeeded_index_bottom = relative_ay_index_bottom
                crop_y_index_top = 0
                crop_y_index_bottom = 28
        

        
        if not no_intersection:
            image_to_embeed = self.images[self.cur_index][crop_x_index_left:crop_x_index_right, crop_y_index_top:crop_y_index_bottom]
            arrayy[absolute_embeeded_x_index_left:absolute_embeeded_x_index_right, absolute_embeeded_index_top:absolute_embeeded_index_bottom] = image_to_embeed

        return arrayy + frame_noise
    
    def get_blind_frame(self):
        arrayy = np.zeros((32,32))
        return arrayy
    
    def get_label_for(self, label_category):
        match label_category:
            case 'category_string':
                return str(self.labels[self.cur_index])
            case 'category_one_hot':
                label = np.zeros(max(self.labels)+1)
                label[self.labels[self.cur_index]] = 1
                return label
            case 'direction_x':
                label_dir_x = np.zeros((2))
                if self.x_dir < 0:
                    label_dir_x[0] = 1.0
                else:
                    label_dir_x[1] = 1.0
                return label_dir_x
            case 'direction_y':
                label_dir_y = np.zeros((2))
                if self.y_dir < 0:
                    label_dir_y[0] = 1.0
                else:
                    label_dir_y[1] = 1.0
                return label_dir_y
            case _:
                return None
    
    def get_all_labels(self):
        label_obj = dict()
        label_obj['category_string'] = self.labels[self.cur_index]
        label_obj['category_one_hot'] = np.zeros(max(self.labels)+1)
        label_obj['category_one_hot'][self.labels[self.cur_index]] = 1
        label_obj['direction_x'] = np.zeros((2))
        if self.x_dir < 0:
            label_obj['direction_x'][0] = 1.0
        else:
            label_obj['direction_x'][1] = 1.0
        label_obj['direction_y'] = np.zeros((2))
        if self.y_dir < 0:
            label_obj['direction_y'][0] = 1.0
        else:
            label_obj['direction_y'][1] = 1.0

        return label_obj
        
    def get_label_list(self):
        return ["category_string", "category_one_hot", "direction_x", "direction_y"]

    def get_shape(self):
        return (32, 32, 1)
    
    def get_name(self):
        return self.name

    def get_category_string_from_one_hot(self, one_hot):
        index_one_hot = np.argmax(one_hot)
        return str(index_one_hot)

#on_off_gray_image_net_patches
class OnOffGrayImageNETPatches:

    @staticmethod
    def get_input_shape():
        return (16, 16, 2)

    @staticmethod
    def default_params():
        return {'folder' : '/disk/scratch/hraguiar/Datasets/image_net_folder', 'valid_samples_per_label': 20, 'preprocessing': 'whiten', 'patch_size': 16}

    def __init__(self, imgnet_kaggle_folder, patch_size=16, valid_samples_per_label=20, preprocessing='whiten'):
        self.name = 'on_off_gray_image_net_patches'
        self.x_size = patch_size
        self.y_size = patch_size
        self.patch_size = patch_size
        self.preprocessing = preprocessing
        self.image_net_kaggle_folder = imgnet_kaggle_folder
        self.num_channels = 2
        self.cur_index = 0
        self.valid_samples_per_label = valid_samples_per_label
        self.preprocessing = preprocessing
        self.running_input_mean = np.zeros((self.patch_size, self.patch_size, 1))
        self.running_input_variance = np.ones((self.patch_size, self.patch_size, 1))
        self.exponential_decay = 0.01

        labels_file = os.path.join(self.image_net_kaggle_folder, 'LOC_synset_mapping.txt')
        self.labels_info = list()
        with open(labels_file, 'r') as f:
            label_info_lines = f.readlines()
            for label_idx, line in enumerate(label_info_lines):
                line_list = line.split(' ')
                folder_label_name = line_list[0]
                label_string = line_list[1:]
                label_string = ' '.join(label_string)
                label_string = label_string.split(',')[0]
                self.labels_info.append({'folder_name': line_list[0], 'string': label_string, 'label_idx': label_idx})

        self.label_idx = randrange(len(self.labels_info)-1)
        label = self.labels_info[self.label_idx]
        self.images_folder = os.path.join(self.image_net_kaggle_folder, 'train', label['folder_name'])
        images_for_label = os.listdir(self.images_folder)
        image_idx = randrange(len(images_for_label)-1)
        self.image_filename = images_for_label[image_idx]
        self._whitening_fitted = False
        self.whitening = 'zca'  # or 'svd' for PCA whitening
        self.whiten_eps = 1e-5
        self.scale = 1.0

    def fit_whitening(self, n_samples=10000):
        """Fit the whitening transform on random patches. Called automatically
        on the first ``get_current_frame()`` if whitening is requested."""

        patches = []
        for _ in range(n_samples):
            self.set_random_position()
            patches.append(self._extract_patch().flatten())

        X = np.array(patches, dtype=np.float64)
        self._mean = X.mean(axis=0)
        X -= self._mean

        cov = (X.T @ X) / (len(X) - 1)
        U, S, _ = np.linalg.svd(cov)

        # Keep only the top-k eigenvalues/eigenvectors (highest variance directions).
        self.n_components = 50
        if self.n_components is not None:
            U = U[:, : self.n_components]
            S = S[: self.n_components]

        if self.whitening == "svd":
            # PCA whitening: rotate into eigenbasis and normalise variance
            self._W = (U / np.sqrt(S + self.whiten_eps)).T
        elif self.whitening == "zca":
            # ZCA whitening: stay in pixel space
            self._W = U @ np.diag(1.0 / np.sqrt(S + self.whiten_eps)) @ U.T
            # Z = U @ self._W.T
            # self.scale = 1.0 / (Z.std() + 1e-12)
            # self.dz = np.quantile(np.abs(Z * self.scale), 1.0 - self.p["deadzone_frac"])

        else:
            raise ValueError(
                f"Unknown whitening option '{self.whitening}'. Use None, 'svd', or 'zca'."
            )

        self._whitening_fitted = True

    def _whiten(self, patch_flat):
        return (self._W @ (patch_flat - self._mean)) * self.scale

    def set_next_position(self):
        pass

    def set_random_position(self):
        self.label_idx = randrange(len(self.labels_info)-1)
        label = self.labels_info[self.label_idx]
        self.images_folder = os.path.join(self.image_net_kaggle_folder, 'train', label['folder_name'])
        images_for_label = os.listdir(self.images_folder)
        image_idx = randrange(self.valid_samples_per_label, len(images_for_label)-1)
        self.image_filename = images_for_label[image_idx]
    
    def set_validation_position(self):
        self.label_idx = randrange(len(self.labels_info)-1)
        label = self.labels_info[self.label_idx]
        self.images_folder = os.path.join(self.image_net_kaggle_folder, 'train', label['folder_name'])
        images_for_label = os.listdir(self.images_folder)
        image_idx = randrange(0, self.valid_samples_per_label)
        self.image_filename = images_for_label[image_idx]
    
    def set_test_position(self):
        self.label_idx = randrange(len(self.labels_info)-1)
        label = self.labels_info[self.label_idx]
        self.images_folder = os.path.join(self.image_net_kaggle_folder, 'train', label['folder_name'])
        images_for_label = os.listdir(self.images_folder)
        image_idx = randrange(self.valid_samples_per_label, len(images_for_label)-1)
        self.image_filename = images_for_label[image_idx]

    def _extract_patch(self):
        if self.image_filename.endswith(".JPEG"):
            img = Image.open(os.path.join(self.images_folder, self.image_filename))
            
            
            width, height = img.size
            img = img.convert('L')
            # Ensure the image is large enough for the patch
            if (width < self.patch_size) or (height < self.patch_size):
                aspect_ratio = width / height if width >= height else height / width
                # Determine the new dimensions while maintaining the aspect ratio
                if width >= height:
                    new_width = int(self.x_size * aspect_ratio)
                    new_height = self.y_size
                else:
                    new_width = self.x_size
                    new_height = int(self.y_size * aspect_ratio)
                # Resize the image while maintaining the aspect ratio
                img = img.resize((new_width, new_height), Image.LANCZOS)
                # Calculate the cropping box
                left = int((new_width - self.x_size) / 2)
                top = int((new_height - self.y_size) / 2)
                right = left + self.x_size
                bottom = top + self.y_size
                # Crop the image to the central 256x256 region
                img = img.crop((left, top, right, bottom))
                width = new_width
                height = new_height
                
            
            # Generate random top-left corner coordinates
            max_x = width - self.patch_size
            max_y = height - self.patch_size
            
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
            
            # Crop the patch (left, top, right, bottom)
            patch = img.crop((x, y, x + self.patch_size, y + self.patch_size))

            return np.array(patch)
    
    def get_current_frame(self):
        if self.image_filename.endswith(".JPEG"):
            img = Image.open(os.path.join(self.images_folder, self.image_filename))
            
            
            width, height = img.size
            img = img.convert('L')
            # Ensure the image is large enough for the patch
            if (width < self.patch_size) or (height < self.patch_size):
                aspect_ratio = width / height if width >= height else height / width
                # Determine the new dimensions while maintaining the aspect ratio
                if width >= height:
                    new_width = int(self.x_size * aspect_ratio)
                    new_height = self.y_size
                else:
                    new_width = self.x_size
                    new_height = int(self.y_size * aspect_ratio)
                # Resize the image while maintaining the aspect ratio
                img = img.resize((new_width, new_height), Image.LANCZOS)
                # Calculate the cropping box
                left = int((new_width - self.x_size) / 2)
                top = int((new_height - self.y_size) / 2)
                right = left + self.x_size
                bottom = top + self.y_size
                # Crop the image to the central 256x256 region
                img = img.crop((left, top, right, bottom))
                width = new_width
                height = new_height
                
            
            # Generate random top-left corner coordinates
            max_x = width - self.patch_size
            max_y = height - self.patch_size
            
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
            
            # Crop the patch (left, top, right, bottom)
            patch = img.crop((x, y, x + self.patch_size, y + self.patch_size))

            array_img = np.copy(patch)
            array_img = np.expand_dims(array_img, axis=-1)
                
            if self.preprocessing == 'whiten':
                if self._whitening_fitted is False:
                    self.fit_whitening()
                array_img = self._whiten(array_img.flatten()).reshape(
                    self.x_size, self.y_size
                )
                onoff_frame = np.zeros((self.x_size, self.y_size, 2))
                onoff_frame[:, :, 0] = np.maximum(array_img, 0)
                onoff_frame[:, :, 1] = np.maximum(-array_img, 0)
            else:
                raise Exception('Preprocessing not recognized')
            return np.array(onoff_frame).reshape(self.x_size, self.y_size, self.num_channels)
        else:
            print("ERROR: image file is not a JPEG at " + self.images_folder + "," + self.image_filename)
            return np.zeros((self.x_size,self.y_size,self.num_channels))
    
    def get_blind_frame(self):
        arrayy = np.zeros((self.x_size,self.y_size,self.num_channels))
        return arrayy
    
    def get_label_for(self, label_category):
        match label_category:
            case 'category_string':
                return self.labels_info[self.label_idx]['string']
            case 'category_one_hot':
                label = np.zeros(len(self.labels_info))
                label[self.label_idx] = 1
                return label
            case _:
                return None
    
    def get_all_labels(self):
        label_obj = dict()
        label_obj['category_string'] = self.labels_info[self.label_idx]['string']
        label_obj['category_one_hot'] = np.zeros(len(self.labels_info))
        label_obj['category_one_hot'][self.label_idx] = 1

        return label_obj
        
    def get_label_list(self):
        return ["category_string", "category_one_hot",]
    
    def get_hyperparameters(self):
        hyperparameters = dict()
        hyperparameters['vg_name'] = self.name
        hyperparameters['screen_size'] = self.x_size
        hyperparameters['preprocessing'] = self.preprocessing
        return hyperparameters

    def get_extensive_name(self):
        return self.name + '_' + str(self.x_size) + '_' + str(self.preprocessing)
    
    def get_shape(self):
        return (self.x_size, self.y_size, self.num_channels)
    
    def get_category_string_from_one_hot(self, one_hot):
        index_one_hot = np.argmax(one_hot)
        return self.labels_info[index_one_hot]['string']

    def get_name(self):
        return self.name

#on_off_gray_image_net_patches_huggingface
class OnOffGrayImageNETPatchesHF:

    @staticmethod
    def get_input_shape():
        return (16, 16, 2)

    @staticmethod
    def default_params():
        return {'preprocessing': 'whiten', 'patch_size': 16}

    def __init__(self, patch_size=16, valid_samples_per_label=20, preprocessing='whiten'):
        self.name = 'on_off_gray_image_net_patches'
        self.x_size = patch_size
        self.y_size = patch_size
        self.patch_size = patch_size
        self.preprocessing = preprocessing
        self.num_channels = 2
        self.cur_index = 0
        self.valid_samples_per_label = valid_samples_per_label
        self.preprocessing = preprocessing
        self.exponential_decay = 0.01

        login()
        self.dataset = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)
        
        self._whitening_fitted = False
        self.whitening = 'zca'  # or 'svd' for PCA whitening
        self.whiten_eps = 1e-5
        self.scale = 1.0

    def fit_whitening(self, n_samples=10000):
        """Fit the whitening transform on random patches. Called automatically
        on the first ``get_current_frame()`` if whitening is requested."""

        patches = []
        for _ in range(n_samples):
            self.set_random_position()
            patches.append(self._extract_patch().flatten())

        X = np.array(patches, dtype=np.float64)
        self._mean = X.mean(axis=0)
        X -= self._mean

        cov = (X.T @ X) / (len(X) - 1)
        U, S, _ = np.linalg.svd(cov)

        # Keep only the top-k eigenvalues/eigenvectors (highest variance directions).
        self.n_components = 50
        if self.n_components is not None:
            U = U[:, : self.n_components]
            S = S[: self.n_components]

        if self.whitening == "svd":
            # PCA whitening: rotate into eigenbasis and normalise variance
            self._W = (U / np.sqrt(S + self.whiten_eps)).T
        elif self.whitening == "zca":
            # ZCA whitening: stay in pixel space
            self._W = U @ np.diag(1.0 / np.sqrt(S + self.whiten_eps)) @ U.T
            # Z = U @ self._W.T
            # self.scale = 1.0 / (Z.std() + 1e-12)
            # self.dz = np.quantile(np.abs(Z * self.scale), 1.0 - self.p["deadzone_frac"])

        else:
            raise ValueError(
                f"Unknown whitening option '{self.whitening}'. Use None, 'svd', or 'zca'."
            )

        self._whitening_fitted = True

    def _whiten(self, patch_flat):
        return (self._W @ (patch_flat - self._mean)) * self.scale

    def set_next_position(self):
        pass

    def set_random_position(self):
        sample = next(iter(self.dataset))
        self.current_image = sample["image"]
        self.current_label = sample["label"]
    
    def set_validation_position(self):
        sample = next(iter(self.dataset))
        self.current_image = sample["image"]
        self.current_label = sample["label"]
    
    def set_test_position(self):
        sample = next(iter(self.dataset))
        self.current_image = sample["image"]
        self.current_label = sample["label"]

    def _extract_patch(self):
        img = self.current_image
        
        
        width, height = img.size
        img = img.convert('L')
        # Ensure the image is large enough for the patch
        if (width < self.patch_size) or (height < self.patch_size):
            aspect_ratio = width / height if width >= height else height / width
            # Determine the new dimensions while maintaining the aspect ratio
            if width >= height:
                new_width = int(self.x_size * aspect_ratio)
                new_height = self.y_size
            else:
                new_width = self.x_size
                new_height = int(self.y_size * aspect_ratio)
            # Resize the image while maintaining the aspect ratio
            img = img.resize((new_width, new_height), Image.LANCZOS)
            # Calculate the cropping box
            left = int((new_width - self.x_size) / 2)
            top = int((new_height - self.y_size) / 2)
            right = left + self.x_size
            bottom = top + self.y_size
            # Crop the image to the central 256x256 region
            img = img.crop((left, top, right, bottom))
            width = new_width
            height = new_height
            
        
        # Generate random top-left corner coordinates
        max_x = width - self.patch_size
        max_y = height - self.patch_size
        
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        
        # Crop the patch (left, top, right, bottom)
        patch = img.crop((x, y, x + self.patch_size, y + self.patch_size))

        return np.array(patch)
    
    def get_current_frame(self):
        img = self.current_image
        
        
        width, height = img.size
        img = img.convert('L')
        # Ensure the image is large enough for the patch
        if (width < self.patch_size) or (height < self.patch_size):
            aspect_ratio = width / height if width >= height else height / width
            # Determine the new dimensions while maintaining the aspect ratio
            if width >= height:
                new_width = int(self.x_size * aspect_ratio)
                new_height = self.y_size
            else:
                new_width = self.x_size
                new_height = int(self.y_size * aspect_ratio)
            # Resize the image while maintaining the aspect ratio
            img = img.resize((new_width, new_height), Image.LANCZOS)
            # Calculate the cropping box
            left = int((new_width - self.x_size) / 2)
            top = int((new_height - self.y_size) / 2)
            right = left + self.x_size
            bottom = top + self.y_size
            # Crop the image to the central 256x256 region
            img = img.crop((left, top, right, bottom))
            width = new_width
            height = new_height
            
        
        # Generate random top-left corner coordinates
        max_x = width - self.patch_size
        max_y = height - self.patch_size
        
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        
        # Crop the patch (left, top, right, bottom)
        patch = img.crop((x, y, x + self.patch_size, y + self.patch_size))

        array_img = np.copy(patch)
        array_img = np.expand_dims(array_img, axis=-1)
            
        if self.preprocessing == 'whiten':
            if self._whitening_fitted is False:
                self.fit_whitening()
            array_img = self._whiten(array_img.flatten()).reshape(
                self.x_size, self.y_size
            )
            onoff_frame = np.zeros((self.x_size, self.y_size, 2))
            onoff_frame[:, :, 0] = np.maximum(array_img, 0)
            onoff_frame[:, :, 1] = np.maximum(-array_img, 0)
        else:
            raise Exception('Preprocessing not recognized')
        return np.array(onoff_frame).reshape(self.x_size, self.y_size, self.num_channels)
    
    def get_blind_frame(self):
        arrayy = np.zeros((self.x_size,self.y_size,self.num_channels))
        return arrayy
    
    def get_label_for(self, label_category):
        match label_category:
            case 'category_string':
                return self.current_label
            case 'category_one_hot':
                label = np.zeros(1000)
                label[self.label_idx] = 1
                return label
            case _:
                return None
    
    def get_all_labels(self):
        label_obj = dict()
        label_obj['category_string'] = self.labels_info[self.label_idx]['string']
        label_obj['category_one_hot'] = np.zeros(len(self.labels_info))
        label_obj['category_one_hot'][self.label_idx] = 1

        return label_obj
        
    def get_label_list(self):
        return ["category_string", "category_one_hot",]
    
    def get_hyperparameters(self):
        hyperparameters = dict()
        hyperparameters['vg_name'] = self.name
        hyperparameters['screen_size'] = self.x_size
        hyperparameters['preprocessing'] = self.preprocessing
        return hyperparameters

    def get_extensive_name(self):
        return self.name + '_' + str(self.x_size) + '_' + str(self.preprocessing)
    
    def get_shape(self):
        return (self.x_size, self.y_size, self.num_channels)
    
    def get_category_string_from_one_hot(self, one_hot):
        index_one_hot = np.argmax(one_hot)
        return self.labels_info[index_one_hot]['string']

    def get_name(self):
        return self.name



# cifar image patches
class CifarImagePatches:
    """
    Generator that yields random greyscale patches from CIFAR-10 images.

    Whitening options (via ``whitening`` param):
      - ``None``  : no whitening
      - ``'svd'`` : PCA whitening (projects into eigenbasis, unit variance per component)
      - ``'zca'`` : ZCA whitening (keeps patches in pixel space)

    Whitening is fitted lazily on the first call to ``get_current_frame()``
    (or explicitly via ``fit_whitening()``).
    """

    _RGB_WEIGHTS = np.array([0.2989, 0.5870, 0.1140])

    @staticmethod
    def default_params():
        return {
            "patch_size": 12,
            "whitening": 'zca',  # None | 'svd' | 'zca'
            "whiten_eps": 1e-5,  # regularisation added to singular values
            "cache_dir": "/tmp/datasets",  # where to store CIFAR-10 data
        }

    def __init__(self, hparams):
        self.name = "cifar_patches"
        self.patch_size = int(hparams.get("patch_size", 12))
        self.whitening = hparams.get("whitening", 'zca')
        self.whiten_eps = float(hparams.get("whiten_eps", 1e-5))

        self.cache_dir = hparams.get("cache_dir", "/tmp/datasets")

        self.x_size = self.patch_size
        self.y_size = self.patch_size
        self.num_channels = 1

        self._grey_images = None  # (N, 1024) float64 in [0, 1]
        self._W = None  # whitening matrix (d, d)
        self._mean = None  # mean patch (d,)
        self._whitening_fitted = False

        self._current_image_idx = 0
        self._current_row = 0
        self._current_col = 0



        self.scale = 1
        self.dz = 0

        self._load_data()
        self.set_random_position()

    # ------------------------------------------------------------------
    # data loading
    # ------------------------------------------------------------------

    def _load_data(self):
        # Try torchvision first (fast, reliable)
        try:
            import torchvision
            import torchvision.transforms as T
            import tempfile, os as _os

            # cache_dir = _os.path.join(tempfile.gettempdir(), "cifar10_tv")
            cache_dir = self.cache_dir
            ds = torchvision.datasets.CIFAR10(root=cache_dir, train=True, download=True)
            data = np.array(ds.data, dtype=np.float64)  # (50000, 32, 32, 3)
            rgb = data.reshape(-1, 32 * 32, 3) / 255.0  # (N, 1024, 3)
            self._grey_images = (rgb * self._RGB_WEIGHTS).sum(axis=2)  # (N, 1024)
            self._n_images = len(self._grey_images)
            return
        except Exception:
            pass

        # Try keras/tensorflow
        try:
            from tensorflow.keras.datasets import cifar10

            (x_train, _), (x_test, _) = cifar10.load_data()
            data = np.concatenate([x_train, x_test], axis=0).astype(
                np.float64
            )  # (60000,32,32,3)
            rgb = data.reshape(-1, 32 * 32, 3) / 255.0
            self._grey_images = (rgb * self._RGB_WEIGHTS).sum(axis=2)
            self._n_images = len(self._grey_images)
            return
        except Exception:
            pass

        # Fallback: sklearn fetch_openml with cache-bust on checksum error
        from sklearn.datasets import fetch_openml
        import sklearn.datasets._base as _skbase, tempfile

        def _fetch(data_home=None):
            return fetch_openml(
                name="cifar_10",
                as_frame=True,
                parser="auto",
                **({} if data_home is None else {"data_home": data_home}),
            )

        try:
            images = _fetch()
        except ValueError:
            # Corrupted cache — retry into a clean temp directory
            images = _fetch(data_home=tempfile.mkdtemp())

        n = len(images.data)
        grey = np.empty((n, 32 * 32), dtype=np.float64)
        for i in range(n):
            rgb = images.data.iloc[i].values.reshape(3, 32 * 32).T
            grey[i] = np.dot(rgb, self._RGB_WEIGHTS)
        grey /= 255.0
        self._grey_images = grey
        self._n_images = n

    # ------------------------------------------------------------------
    # whitening
    # ------------------------------------------------------------------

    def fit_whitening(self, n_samples=10000):
        """Fit the whitening transform on random patches. Called automatically
        on the first ``get_current_frame()`` if whitening is requested."""
        if self.whitening is None:
            self._whitening_fitted = True
            return

        patches = []
        for _ in range(n_samples):
            idx = np.random.randint(self._n_images)
            row = np.random.randint(0, 32 - self.patch_size + 1)
            col = np.random.randint(0, 32 - self.patch_size + 1)
            patches.append(self._extract_patch(idx, row, col).flatten())

        X = np.array(patches, dtype=np.float64)
        self._mean = X.mean(axis=0)
        X -= self._mean

        cov = (X.T @ X) / (len(X) - 1)
        U, S, _ = np.linalg.svd(cov)

        # Keep only the top-k eigenvalues/eigenvectors (highest variance directions).
        self.n_components = 50
        if self.n_components is not None:
            U = U[:, : self.n_components]
            S = S[: self.n_components]

        if self.whitening == "svd":
            # PCA whitening: rotate into eigenbasis and normalise variance
            self._W = (U / np.sqrt(S + self.whiten_eps)).T
        elif self.whitening == "zca":
            # ZCA whitening: stay in pixel space
            self._W = U @ np.diag(1.0 / np.sqrt(S + self.whiten_eps)) @ U.T
            # Z = U @ self._W.T
            # self.scale = 1.0 / (Z.std() + 1e-12)
            # self.dz = np.quantile(np.abs(Z * self.scale), 1.0 - self.p["deadzone_frac"])

        else:
            raise ValueError(
                f"Unknown whitening option '{self.whitening}'. Use None, 'svd', or 'zca'."
            )

        self._whitening_fitted = True

    def _whiten(self, patch_flat):
        if self.whitening is None:
            return patch_flat
        return (self._W @ (patch_flat - self._mean)) * self.scale

    # ------------------------------------------------------------------
    # patch extraction
    # ------------------------------------------------------------------

    def _extract_patch(self, image_idx, row, col):
        img = self._grey_images[image_idx].reshape(32, 32)
        return img[row : row + self.patch_size, col : col + self.patch_size]

    # ------------------------------------------------------------------
    # public interface (mirrors other generators)
    # ------------------------------------------------------------------

    def set_random_position(self):
        self._current_image_idx = np.random.randint(self._n_images)
        self._current_row = np.random.randint(0, 32 - self.patch_size + 1)
        self._current_col = np.random.randint(0, 32 - self.patch_size + 1)

    def get_current_frame(self):
        if not self._whitening_fitted:
            self.fit_whitening()

        patch = self._extract_patch(
            self._current_image_idx,
            self._current_row,
            self._current_col,
        ).astype(np.float64)

        if self.whitening is not None:
            patch = self._whiten(patch.flatten()).reshape(
                self.patch_size, self.patch_size
            )

        return patch.reshape(self.patch_size, self.patch_size, 1).copy()

    def get_blind_frame(self):
        return np.zeros((self.patch_size, self.patch_size, 1))

    def get_shape(self):
        return (self.y_size, self.x_size, self.num_channels)

    def get_name(self):
        return self.name


# cifar image patches
class CifarImagePatches:
    """
    Generator that yields random greyscale patches from CIFAR-10 images.

    Whitening options (via ``whitening`` param):
      - ``None``  : no whitening
      - ``'svd'`` : PCA whitening (projects into eigenbasis, unit variance per component)
      - ``'zca'`` : ZCA whitening (keeps patches in pixel space)

    Whitening is fitted lazily on the first call to ``get_current_frame()``
    (or explicitly via ``fit_whitening()``).
    """

    _RGB_WEIGHTS = np.array([0.2989, 0.5870, 0.1140])

    @staticmethod
    def default_params():
        return {
            "patch_size": 12,
            "whitening": 'zca',  # None | 'svd' | 'zca'
            "whiten_eps": 1e-5,  # regularisation added to singular values
            "cache_dir": "/tmp/datasets",  # where to store CIFAR-10 data
        }

    def __init__(self, hparams):
        self.name = "cifar_patches"
        self.patch_size = int(hparams.get("patch_size", 12))
        self.whitening = hparams.get("whitening", 'zca')
        self.whiten_eps = float(hparams.get("whiten_eps", 1e-5))

        self.cache_dir = hparams.get("cache_dir", "/tmp/datasets")

        self.x_size = self.patch_size
        self.y_size = self.patch_size
        self.num_channels = 1

        self._grey_images = None  # (N, 1024) float64 in [0, 1]
        self._W = None  # whitening matrix (d, d)
        self._mean = None  # mean patch (d,)
        self._whitening_fitted = False

        self._current_image_idx = 0
        self._current_row = 0
        self._current_col = 0



        self.scale = 1
        self.dz = 0

        self._load_data()
        self.set_random_position()

    # ------------------------------------------------------------------
    # data loading
    # ------------------------------------------------------------------

    def _load_data(self):
        # Try torchvision first (fast, reliable)
        try:
            import torchvision
            import torchvision.transforms as T
            import tempfile, os as _os

            # cache_dir = _os.path.join(tempfile.gettempdir(), "cifar10_tv")
            cache_dir = self.cache_dir
            ds = torchvision.datasets.CIFAR10(root=cache_dir, train=True, download=True)
            data = np.array(ds.data, dtype=np.float64)  # (50000, 32, 32, 3)
            rgb = data.reshape(-1, 32 * 32, 3) / 255.0  # (N, 1024, 3)
            self._grey_images = (rgb * self._RGB_WEIGHTS).sum(axis=2)  # (N, 1024)
            self._n_images = len(self._grey_images)
            return
        except Exception:
            pass

        # Try keras/tensorflow
        try:
            from tensorflow.keras.datasets import cifar10

            (x_train, _), (x_test, _) = cifar10.load_data()
            data = np.concatenate([x_train, x_test], axis=0).astype(
                np.float64
            )  # (60000,32,32,3)
            rgb = data.reshape(-1, 32 * 32, 3) / 255.0
            self._grey_images = (rgb * self._RGB_WEIGHTS).sum(axis=2)
            self._n_images = len(self._grey_images)
            return
        except Exception:
            pass

        # Fallback: sklearn fetch_openml with cache-bust on checksum error
        from sklearn.datasets import fetch_openml
        import sklearn.datasets._base as _skbase, tempfile

        def _fetch(data_home=None):
            return fetch_openml(
                name="cifar_10",
                as_frame=True,
                parser="auto",
                **({} if data_home is None else {"data_home": data_home}),
            )

        try:
            images = _fetch()
        except ValueError:
            # Corrupted cache — retry into a clean temp directory
            images = _fetch(data_home=tempfile.mkdtemp())

        n = len(images.data)
        grey = np.empty((n, 32 * 32), dtype=np.float64)
        for i in range(n):
            rgb = images.data.iloc[i].values.reshape(3, 32 * 32).T
            grey[i] = np.dot(rgb, self._RGB_WEIGHTS)
        grey /= 255.0
        self._grey_images = grey
        self._n_images = n

    # ------------------------------------------------------------------
    # whitening
    # ------------------------------------------------------------------

    def fit_whitening(self, n_samples=10000):
        """Fit the whitening transform on random patches. Called automatically
        on the first ``get_current_frame()`` if whitening is requested."""
        if self.whitening is None:
            self._whitening_fitted = True
            return

        patches = []
        for _ in range(n_samples):
            idx = np.random.randint(self._n_images)
            row = np.random.randint(0, 32 - self.patch_size + 1)
            col = np.random.randint(0, 32 - self.patch_size + 1)
            patches.append(self._extract_patch(idx, row, col).flatten())

        X = np.array(patches, dtype=np.float64)
        self._mean = X.mean(axis=0)
        X -= self._mean

        cov = (X.T @ X) / (len(X) - 1)
        U, S, _ = np.linalg.svd(cov)

        # Keep only the top-k eigenvalues/eigenvectors (highest variance directions).
        self.n_components = 50
        if self.n_components is not None:
            U = U[:, : self.n_components]
            S = S[: self.n_components]

        if self.whitening == "svd":
            # PCA whitening: rotate into eigenbasis and normalise variance
            self._W = (U / np.sqrt(S + self.whiten_eps)).T
        elif self.whitening == "zca":
            # ZCA whitening: stay in pixel space
            self._W = U @ np.diag(1.0 / np.sqrt(S + self.whiten_eps)) @ U.T
            # Z = U @ self._W.T
            # self.scale = 1.0 / (Z.std() + 1e-12)
            # self.dz = np.quantile(np.abs(Z * self.scale), 1.0 - self.p["deadzone_frac"])

        else:
            raise ValueError(
                f"Unknown whitening option '{self.whitening}'. Use None, 'svd', or 'zca'."
            )

        self._whitening_fitted = True

    def _whiten(self, patch_flat):
        if self.whitening is None:
            return patch_flat
        return (self._W @ (patch_flat - self._mean)) * self.scale

    # ------------------------------------------------------------------
    # patch extraction
    # ------------------------------------------------------------------

    def _extract_patch(self, image_idx, row, col):
        img = self._grey_images[image_idx].reshape(32, 32)
        return img[row : row + self.patch_size, col : col + self.patch_size]

    # ------------------------------------------------------------------
    # public interface (mirrors other generators)
    # ------------------------------------------------------------------

    def set_random_position(self):
        self._current_image_idx = np.random.randint(self._n_images)
        self._current_row = np.random.randint(0, 32 - self.patch_size + 1)
        self._current_col = np.random.randint(0, 32 - self.patch_size + 1)

    def get_current_frame(self):
        if not self._whitening_fitted:
            self.fit_whitening()

        patch = self._extract_patch(
            self._current_image_idx,
            self._current_row,
            self._current_col,
        ).astype(np.float64)

        if self.whitening is not None:
            patch = self._whiten(patch.flatten()).reshape(
                self.patch_size, self.patch_size
            )

        return patch.reshape(self.patch_size, self.patch_size, 1).copy()

    def get_blind_frame(self):
        return np.zeros((self.patch_size, self.patch_size, 1))

    def get_shape(self):
        return (self.y_size, self.x_size, self.num_channels)

    def get_name(self):
        return self.name


# catcam_patch16
class CatCamPatch16:
    IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".pgm", ".ppm", ".tif", ".tiff")
    _RGB_WEIGHTS = np.array([0.2989, 0.5870, 0.1140], dtype=np.float64)

    @staticmethod
    def get_input_shape(patch_size=16):
        return (patch_size, patch_size, 1)

    @staticmethod
    def default_params():
        return {
            "directory": "/content/drive/MyDrive/catcam_movies",
            "num_movies": 10,
            "patch_size": 16,
            "download": True,
            "whitening": "zca",  # None | 'svd' | 'zca'
            "whiten_eps": 1e-5,  # regularization added to singular values
            "n_components": None,  # None uses all d = patch_size**2 dims; int truncates SVD
            "whiten_samples": 1000,
        }

    def __init__(
        self,
        directory="/content/drive/MyDrive/catcam_movies",
        num_movies=10,
        patch_size=16,
        download=True,
        whitening="zca",
        whiten_eps=1e-5,
        n_components=None,
        whiten_samples=1000,
    ):
        self.name = "catcam_patch16"
        self.directory = directory
        self.num_movies = num_movies
        self.patch_size = int(patch_size)
        self.x_size = self.patch_size
        self.y_size = self.patch_size
        self.num_channels = 1
        self.download = download

        # Whitening configuration (mirrors CifarImagePatches)
        self.whitening = whitening
        self.whiten_eps = float(whiten_eps)
        self.n_components = n_components
        self.whiten_samples = int(whiten_samples)

        self._W = None
        self._mean = None
        self._whitening_fitted = False

        if self.download:
            self._download_and_extract()

        self.frames, self.movie_ids = self._index_frames()
        if len(self.frames) == 0:
            raise FileNotFoundError(
                f"No image frames found under '{self.directory}' for movies 01..{self.num_movies:02d}."
            )

        self.cur_index = 0
        self.x_offset = 0
        self.y_offset = 0
        self.x_dir = 1
        self.y_dir = 1
        self._cache = {}

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _download_and_extract(self):
        os.makedirs(self.directory, exist_ok=True)
        for i in range(1, self.num_movies + 1):
            movie_name = f"movie{i:02d}.tar"
            tar_path = os.path.join(self.directory, movie_name)
            extract_dir = os.path.join(self.directory, f"movie{i:02d}")

            if not os.path.exists(tar_path):
                url = f"https://zenodo.org/records/46481/files/{movie_name}?download=1"
                urllib.request.urlretrieve(url, tar_path)

            if not os.path.exists(extract_dir):
                with tarfile.open(tar_path) as tar:
                    tar.extractall(path=extract_dir)

    def _index_frames(self):
        frames, movie_ids = [], []
        for i in range(1, self.num_movies + 1):
            movie_tag = f"movie{i:02d}"
            movie_dir = os.path.join(self.directory, movie_tag)
            if not os.path.isdir(movie_dir):
                continue
            for root, _, files in os.walk(movie_dir):
                for fname in sorted(files):
                    if not fname.startswith("._") and fname.lower().endswith(self.IMAGE_EXTS):
                        frames.append(os.path.join(root, fname))
                        movie_ids.append(i)
        return frames, movie_ids

    def _load_frame(self, idx):
        if idx not in self._cache:
            with Image.open(self.frames[idx]) as img:
                arr = np.asarray(img, dtype=np.float32)
            if arr.ndim == 3:
                if arr.shape[2] == 1:
                    arr = arr[:, :, 0]
                else:
                    arr = np.tensordot(
                        arr[:, :, :3].astype(np.float64),
                        self._RGB_WEIGHTS,
                        axes=([-1], [0]),
                    ).astype(np.float32)
            self._cache[idx] = arr
        return self._cache[idx]

    # ------------------------------------------------------------------
    # Whitening (mirrors CifarImagePatches)
    # ------------------------------------------------------------------

    def fit_whitening(self, n_samples=None):
        """Fit PCA ('svd') or ZCA ('zca') whitening on sampled CatCam patches.
        Called automatically on the first get_current_frame() if whitening is enabled."""
        if self.whitening is None:
            self._whitening_fitted = True
            return
        print(f"Fitting {self.whitening} whitening on {self.whiten_samples} patches...")
        num_samples = self.whiten_samples if n_samples is None else int(n_samples)
        patches = []

        for _ in range(num_samples):
            frame_idx = int(np.random.randint(0, len(self.frames)))
            img = self._load_frame(frame_idx)
            h, w = img.shape[:2]

            max_x = max(0, w - self.patch_size)
            max_y = max(0, h - self.patch_size)
            center_x = max_x // 2
            center_y = max_y // 2

            x0 = max(0, min(max_x, center_x + int(np.random.randint(-4, 5))))
            y0 = max(0, min(max_y, center_y + int(np.random.randint(-4, 5))))

            patch = img[y0 : y0 + self.patch_size, x0 : x0 + self.patch_size].astype(np.float64)
            patches.append(patch.ravel())

        X = np.array(patches, dtype=np.float64)
        self._mean = X.mean(axis=0)
        X_centered = X - self._mean

        cov = (X_centered.T @ X_centered) / max(1, len(X_centered) - 1)
        U, S, _ = np.linalg.svd(cov)

        if self.n_components is not None:
            k = min(int(self.n_components), U.shape[1])
            U = U[:, :k]
            S = S[:k]

        if self.whitening == "svd":
            # PCA whitening: project into eigenbasis and normalize variance per component
            self._W = (U / np.sqrt(S + self.whiten_eps)).T
        elif self.whitening == "zca":
            # ZCA whitening: whiten and rotate back into pixel space
            self._W = (U * (1.0 / np.sqrt(S + self.whiten_eps))) @ U.T
        else:
            raise ValueError(
                f"Unknown whitening option '{self.whitening}'. Use None, 'svd', or 'zca'."
            )

        self._whitening_fitted = True

    def _whiten(self, patch_flat):
        if self.whitening is None:
            return patch_flat
        return self._W @ (patch_flat - self._mean)

    # ------------------------------------------------------------------
    # Position & patch extraction
    # ------------------------------------------------------------------

    def _sample_state(self, low_idx=0, high_idx=None):
        high = len(self.frames) if high_idx is None else min(high_idx, len(self.frames))
        self.cur_index = int(np.random.randint(low_idx, max(low_idx + 1, high)))
        self.x_offset = int(np.random.randint(-4, 5))
        self.y_offset = int(np.random.randint(-4, 5))
        self.x_dir = -1 if np.random.rand() < 0.5 else 1
        self.y_dir = -1 if np.random.rand() < 0.5 else 1

    def set_next_position(self):
        self.cur_index = (self.cur_index + 1) % len(self.frames)

    def set_random_position(self):
        self._sample_state(0, len(self.frames))

    def set_test_position(self):
        split = int(0.8 * len(self.frames))
        self._sample_state(split, len(self.frames))

    def set_validation_position(self):
        start = int(0.7 * len(self.frames))
        end = int(0.8 * len(self.frames))
        self._sample_state(start, max(start + 1, end))

    def get_current_frame(self):
        if not self._whitening_fitted:
            self.fit_whitening()

        img = self._load_frame(self.cur_index)
        h, w = img.shape[:2]
        cx = max(0, min(w - self.patch_size, (w - self.patch_size) // 2 + self.x_offset))
        cy = max(0, min(h - self.patch_size, (h - self.patch_size) // 2 + self.y_offset))
        patch = img[cy : cy + self.patch_size, cx : cx + self.patch_size].astype(np.float64)

        if self.whitening is not None:
            whitened = self._whiten(patch.ravel())
            if self.whitening == "svd" and self.n_components is not None:
                return whitened.reshape(len(whitened), 1).astype(np.float32)
            return whitened.reshape(self.patch_size, self.patch_size, 1).astype(np.float32)

        return patch.reshape(self.patch_size, self.patch_size, 1).astype(np.float32)

    def get_blind_frame(self):
        return np.zeros((self.y_size, self.x_size, self.num_channels), dtype=np.float32)

    def get_label_for(self, label_category):
        match label_category:
            case "movie_id":
                return self.movie_ids[self.cur_index]
            case "movie_one_hot":
                vec = np.zeros(self.num_movies, dtype=np.float32)
                vec[self.movie_ids[self.cur_index] - 1] = 1.0
                return vec
            case "direction_x":
                return np.array([1.0, 0.0], dtype=np.float32) if self.x_dir < 0 else np.array([0.0, 1.0], dtype=np.float32)
            case "direction_y":
                return np.array([1.0, 0.0], dtype=np.float32) if self.y_dir < 0 else np.array([0.0, 1.0], dtype=np.float32)
            case _:
                return None

    def get_all_labels(self):
        return {name: self.get_label_for(name) for name in self.get_label_list()}

    def get_label_list(self):
        return ["movie_id", "movie_one_hot", "direction_x", "direction_y"]

    def get_shape(self):
        return (self.y_size, self.x_size, self.num_channels)

    def get_name(self):
        return self.name