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

import os
import math
import random
import zipfile
import urllib.request
from random import randrange

import numpy as np
from PIL import Image



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
        self.label_size = 10
        self.labels = a['label'].to_numpy()
        a = a.drop(['label'], axis=1)
        self.x_size = 32
        self.y_size = 32
        self.num_channels = 1
        self.images = a.to_numpy().reshape(samples,28,28)/256
        self.validation_set = validation_set
        self.num_samples = samples
        assert samples > validation_set
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
        a = pd.read_csv(file)
        self.label_size = 10
        self.labels = a['label'].to_numpy()
        a = a.drop(['label'], axis=1)
        self.x_size = 32
        self.y_size = 32
        self.num_channels = 1
        self.images = a.to_numpy().reshape(42000,28,28)/256
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


#on_off_gray_video_patches
class OnOffGrayVideoPatches:

    @staticmethod
    def get_input_shape():
        return (16, 16, 2)             # (y, x, on/off) -- one frame per time step

    @staticmethod
    def default_params():
        return {'folder': '/disk/scratch/hraguiar/Datasets/davis',
                'source': 'davis2017-trainval',
                'patch_size': 16,
                'n_frames': 14,
                'output_mode': 'frame',
                'readout': 'last',
                'temporal_stride': 1,
                'downsample': 2,
                'n_valid_sequences': 8,
                'n_test_sequences': 8,
                'preprocessing': 'whiten',
                'motion_percentile': 50.0}

    def __init__(self,
                 folder,
                 source='davis2017-trainval',
                 patch_size=16,
                 n_frames=14,
                 output_mode='frame',
                 readout='last',
                 temporal_stride=1,
                 downsample=2,
                 n_valid_sequences=8,
                 n_test_sequences=8,
                 preprocessing='whiten',
                 whitening='zca',
                 n_components=None,
                 whiten_eps=1e-3,
                 motion_percentile=50.0,
                 n_direction_bins=8,
                 max_disp=None,
                 cache_size=32,
                 sequence_switch_prob=0.02,
                 auto_download=True,
                 split_seed=1234,
                 time_last=False):

        self.name = 'on_off_gray_video_patches'
        self.folder = folder
        self.source = source

        self.patch_size = patch_size
        self.x_size = patch_size
        self.y_size = patch_size
        self.n_frames = n_frames
        self.temporal_stride = temporal_stride

        # 'frame'  -> get_current_frame() returns (P, P, 2), one time step, and
        #             set_next_position() walks forward through the clip.
        # 'block'  -> returns the whole (n_frames, P, P, 2) space-time patch.
        if output_mode not in ('frame', 'block'):
            raise ValueError("output_mode must be 'frame' or 'block'")
        self.output_mode = output_mode
        self.readout = readout
        if readout == 'last':            # causal: current frame is the newest one
            self._readout_index = n_frames - 1
        elif readout == 'center':        # non-causal but symmetric temporal filter
            self._readout_index = n_frames // 2
        elif isinstance(readout, int):
            self._readout_index = readout
        else:
            raise ValueError("readout must be 'last', 'center' or an int")
        self.downsample = max(1, int(downsample))
        self.num_channels = 2
        self.time_last = time_last

        self.preprocessing = preprocessing
        self.whitening = whitening
        if output_mode == 'frame' and preprocessing == 'whiten' and whitening != 'zca':
            raise ValueError("output_mode='frame' needs whitening='zca' (PCA/'svd' "
                             "output lives in component space, not pixel space).")
        self.n_components = n_components
        self.whiten_eps = whiten_eps
        self.scale = 1.0
        self._whitening_fitted = False
        self._W = None
        self._mean = None

        # patches whose mean |dI/dt| falls below this are rejected -- without it
        # most random space-time patches are static and carry no motion signal.
        self.motion_percentile = motion_percentile
        self._motion_threshold = 0.0
        self.n_direction_bins = n_direction_bins
        # search radius of the velocity readout; >P/4 makes the overlap too small
        self.max_disp = max_disp if max_disp is not None else max(2, patch_size // 4)

        # ---- data -------------------------------------------------------- #
        if auto_download and source is not None and not self._folder_has_frames(folder):
            ensure_dataset(folder, source)

        self.sequences_info = self._scan_sequences(folder)
        if len(self.sequences_info) == 0:
            raise RuntimeError(
                f"No usable sequences found under '{folder}'. Expected sub-folders "
                f"holding at least {self._span()} consecutive frames.")

        # ---- deterministic train / valid / test split over *sequences* ---- #
        order = list(range(len(self.sequences_info)))
        random.Random(split_seed).shuffle(order)
        n_valid = min(n_valid_sequences, max(0, len(order) - 1))
        n_test = min(n_test_sequences, max(0, len(order) - n_valid - 1))
        self.valid_sequences = sorted(order[:n_valid])
        self.test_sequences = sorted(order[n_valid:n_valid + n_test])
        self.train_sequences = sorted(order[n_valid + n_test:])
        self.n_valid_sequences = n_valid
        self.n_test_sequences = n_test

        # kept for symmetry with the image class (labels = sequence identity)
        self.labels_info = [{'folder_name': s['name'], 'string': s['name'],
                             'label_idx': i} for i, s in enumerate(self.sequences_info)]

        # ---- frame cache + current position ------------------------------ #
        self.cache_size = cache_size
        self.sequence_switch_prob = sequence_switch_prob
        self._cache = {}
        self._cache_order = []
        self._last_raw_patch = None

        self.cur_index = 0
        self.current_split = 'train'
        self.sequence_idx = self.train_sequences[0]
        self.t0 = 0
        self.x = 0
        self.y = 0
        self.set_random_position()

    # ------------------------------------------------------------------ #
    #  Dataset discovery
    # ------------------------------------------------------------------ #

    def _span(self):
        """Number of source frames spanned by one patch."""
        return (self.n_frames - 1) * self.temporal_stride + 1

    @staticmethod
    def _folder_has_frames(folder):
        if not os.path.isdir(folder):
            return False
        for _, _, files in os.walk(folder):
            if any(f.endswith(IMAGE_EXTENSIONS) for f in files):
                return True
        return False

    def _scan_sequences(self, folder):
        """One directory listing for the whole dataset, done once at init."""
        sequences = []
        for root, dirs, files in os.walk(folder):
            frames = sorted(f for f in files if f.endswith(IMAGE_EXTENSIONS))
            if len(frames) < self._span():
                continue
            sequences.append({'name': os.path.basename(root.rstrip(os.sep)),
                              'path': root,
                              'frames': frames,
                              'n_frames': len(frames)})
        sequences.sort(key=lambda s: s['path'])
        return sequences

    # ------------------------------------------------------------------ #
    #  Frame loading (LRU-cached, grayscale, optionally downsampled)
    # ------------------------------------------------------------------ #

    def _load_sequence(self, sequence_idx):
        if sequence_idx in self._cache:
            self._cache_order.remove(sequence_idx)
            self._cache_order.append(sequence_idx)
            return self._cache[sequence_idx]

        info = self.sequences_info[sequence_idx]
        stack = []
        for fname in info['frames']:
            img = Image.open(os.path.join(info['path'], fname)).convert('L')
            if self.downsample > 1:
                img = img.resize((max(1, img.width // self.downsample),
                                  max(1, img.height // self.downsample)),
                                 Image.LANCZOS)
            stack.append(np.asarray(img, dtype=np.uint8))

        h = min(a.shape[0] for a in stack)      # a couple of DAVIS clips wobble by 1px
        w = min(a.shape[1] for a in stack)
        video = np.stack([a[:h, :w] for a in stack], axis=0)

        self._cache[sequence_idx] = video
        self._cache_order.append(sequence_idx)
        while len(self._cache_order) > self.cache_size:
            evict = self._cache_order.pop(0)
            del self._cache[evict]
        return video

    def preload(self, verbose=True):
        """Decode every sequence into RAM once, so sampling never touches disk.
        DAVIS 2017 480p at downsample=2 is roughly 0.6 GB."""
        self.cache_size = len(self.sequences_info)
        total = 0
        for i in range(len(self.sequences_info)):
            total += self._load_sequence(i).nbytes
            if verbose:
                print(f"\r  preloaded {i + 1}/{len(self.sequences_info)} sequences "
                      f"({total / 1e9:.2f} GB)", end='', flush=True)
        if verbose:
            print()
        return total

    # ------------------------------------------------------------------ #
    #  Positioning
    # ------------------------------------------------------------------ #

    def _sequences_for_split(self, split):
        if split == 'valid':
            return self.valid_sequences or self.train_sequences
        if split == 'test':
            return self.test_sequences or self.train_sequences
        return self.train_sequences

    def _pick_sequence(self, pool):
        """Prefer a sequence that is already decoded in RAM: reloading a clip
        costs ~100 JPEG decodes, which would otherwise dominate sampling time.
        With probability ``sequence_switch_prob`` we pull in a fresh one, so the
        sampler still covers the whole dataset over a long run."""
        warm = [s for s in pool if s in self._cache]
        if warm and len(self._cache) >= self.cache_size and \
                random.random() > self.sequence_switch_prob:
            return warm[randrange(len(warm))]
        return pool[randrange(len(pool))]

    def _sample_position(self, split, enforce_motion=True):
        pool = self._sequences_for_split(split)
        sequence_idx, t0, y, x = self.sequence_idx, self.t0, self.y, self.x
        for _ in range(200):
            sequence_idx = self._pick_sequence(pool)
            video = self._load_sequence(sequence_idx)
            T, H, W = video.shape
            if T < self._span() or H < self.patch_size or W < self.patch_size:
                continue
            t0 = randrange(0, T - self._span() + 1)
            y = randrange(0, H - self.patch_size + 1)
            x = randrange(0, W - self.patch_size + 1)

            if not enforce_motion or self.motion_percentile <= 0 or not self._whitening_fitted:
                self.sequence_idx, self.t0, self.y, self.x = sequence_idx, t0, y, x
                return
            patch = self._patch_at(sequence_idx, t0, y, x)
            if self._motion_energy(patch) >= self._motion_threshold:
                self.sequence_idx, self.t0, self.y, self.x = sequence_idx, t0, y, x
                return
        # give up on the motion criterion rather than loop forever
        self.sequence_idx, self.t0, self.y, self.x = sequence_idx, t0, y, x

    def set_random_position(self):
        self.current_split = 'train'
        self._sample_position('train')
        self._last_raw_patch = None

    def set_validation_position(self):
        self.current_split = 'valid'
        self._sample_position('valid')
        self._last_raw_patch = None

    def set_test_position(self):
        self.current_split = 'test'
        self._sample_position('test')
        self._last_raw_patch = None

    def set_next_position(self):
        """Slide the window one frame forward at the same retinal location.
        Gives a temporally continuous stream (useful for recurrent / STDP-ish
        learning); wraps to a fresh random position at the end of the clip."""
        video = self._load_sequence(self.sequence_idx)
        if self.t0 + self._span() < video.shape[0]:
            self.t0 += 1
            self.cur_index += 1
            self._last_raw_patch = None
        else:
            self._sample_position(self.current_split)
            self.cur_index = 0
            self._last_raw_patch = None

    # ------------------------------------------------------------------ #
    #  Patch extraction
    # ------------------------------------------------------------------ #

    def _patch_at(self, sequence_idx, t0, y, x):
        video = self._load_sequence(sequence_idx)
        ts = t0 + np.arange(self.n_frames) * self.temporal_stride
        patch = video[ts, y:y + self.patch_size, x:x + self.patch_size]
        return patch.astype(np.float64) / 255.0

    def _extract_patch(self):
        return self._patch_at(self.sequence_idx, self.t0, self.y, self.x)

    @staticmethod
    def _motion_energy(patch):
        """Mean |temporal derivative| -- cheap proxy for 'something moved here'."""
        if patch.shape[0] < 2:
            return 0.0
        return float(np.mean(np.abs(np.diff(patch, axis=0))))

    # ------------------------------------------------------------------ #
    #  Whitening (spatio-temporal)
    # ------------------------------------------------------------------ #

    def output_dim(self):
        if self.preprocessing == 'whiten' and self.whitening == 'svd':
            return self.n_components or (self.n_frames * self.patch_size ** 2)
        return self.n_frames * self.patch_size ** 2

    def _iter_random_patches(self, n, split='train', patches_per_visit=64):
        """Yield ``n`` random patches, drawing several per loaded sequence so we
        don't decode a whole clip for every single sample."""
        pool = self._sequences_for_split(split)
        produced = 0
        while produced < n:
            sequence_idx = pool[randrange(len(pool))]   # fitting wants full coverage
            video = self._load_sequence(sequence_idx)
            T, H, W = video.shape
            if T < self._span() or H < self.patch_size or W < self.patch_size:
                continue
            for _ in range(min(patches_per_visit, n - produced)):
                t0 = randrange(0, T - self._span() + 1)
                y = randrange(0, H - self.patch_size + 1)
                x = randrange(0, W - self.patch_size + 1)
                yield self._patch_at(sequence_idx, t0, y, x)
                produced += 1

    def fit_whitening(self, n_samples=20000, n_probe=4000, chunk=512, verbose=True):
        """Fit the space-time whitening transform (and the motion threshold).

        The covariance is accumulated in chunks, so memory is O(D^2) rather than
        O(n_samples * D) and ``n_samples`` can be as large as you like.
        Called automatically on the first ``get_current_frame()``.
        """
        D = self.n_frames * self.patch_size ** 2
        if verbose:
            print(f"[{self.name}] fitting whitening: D={D}, n_samples={n_samples}")
        if n_samples < 4 * D:
            print(f"  WARNING: n_samples={n_samples} < 4*D={4 * D}. The covariance will be "
                  f"rank-deficient and whitening will amplify noise directions. "
                  f"Use more samples or set n_components <~ {n_samples // 4}.")

        # ---- pass 1: motion threshold ------------------------------------ #
        if self.motion_percentile > 0:
            motions = [self._motion_energy(p)
                       for p in self._iter_random_patches(n_probe)]
            self._motion_threshold = float(np.percentile(motions, self.motion_percentile))
        else:
            self._motion_threshold = 0.0

        # ---- pass 2: streaming mean / covariance over moving patches ------ #
        s1 = np.zeros(D)
        s2 = np.zeros((D, D))
        n_kept = 0
        n_seen = 0
        buf = []
        keep_fraction = max(0.05, 1.0 - self.motion_percentile / 100.0)
        for p in self._iter_random_patches(int(1.3 * n_samples / keep_fraction) + 100):
            n_seen += 1
            if self._motion_energy(p) < self._motion_threshold:
                continue
            buf.append(p.ravel())
            n_kept += 1
            if len(buf) >= chunk:
                B = np.asarray(buf)
                s1 += B.sum(axis=0)
                s2 += B.T @ B
                buf = []
            if n_kept >= n_samples:
                break
        if buf:
            B = np.asarray(buf)
            s1 += B.sum(axis=0)
            s2 += B.T @ B
        if n_kept < 2:
            raise RuntimeError('Not enough patches passed the motion threshold.')

        self._mean = s1 / n_kept
        cov = (s2 - np.outer(s1, self._mean)) / (n_kept - 1)
        cov = 0.5 * (cov + cov.T)                        # kill numerical asymmetry

        U, S, _ = np.linalg.svd(cov)
        S = np.maximum(S, 0.0)

        if self.n_components is not None:
            U = U[:, :self.n_components]
            S = S[:self.n_components]

        eps = self.whiten_eps * float(S.mean())          # scale-free regulariser

        if self.whitening == 'svd':
            self._W = (U / np.sqrt(S + eps)).T
        elif self.whitening == 'zca':
            self._W = U @ np.diag(1.0 / np.sqrt(S + eps)) @ U.T
        else:
            raise ValueError(f"Unknown whitening option '{self.whitening}'. Use 'svd' or 'zca'.")

        # rows of W that produce the read-out frame: an (P*P, D) space-time
        # filter bank, ~n_frames times cheaper to apply than the full transform
        if self.whitening == 'zca':
            P2 = self.patch_size ** 2
            self._W_read = self._W[self._readout_index * P2:(self._readout_index + 1) * P2]
        else:
            self._W_read = None

        # ---- pass 3: output gain, measured on *fresh* patches ------------- #
        self.scale = 1.0
        self._whitening_fitted = True
        whiten_fn = self._whiten_readout if self.output_mode == 'frame' else self._whiten
        fresh = np.asarray([whiten_fn(p.ravel())
                            for p in self._iter_random_patches(3000)
                            if self._motion_energy(p) >= self._motion_threshold])
        self.scale = 1.0 / (fresh.std() + 1e-12)

        if verbose:
            spectrum = S / (S[0] + 1e-12)
            print(f"  kept {n_kept}/{n_seen} patches (motion thr {self._motion_threshold:.4f}), "
                  f"eigenvalue ratio S[-1]/S[0]={spectrum[-1]:.2e}, output gain {self.scale:.3f}")

    def _whiten(self, patch_flat):
        return (self._W @ (patch_flat - self._mean)) * self.scale

    def _whiten_readout(self, patch_flat):
        """Whiten the space-time window but return only the read-out frame."""
        return (self._W_read @ (patch_flat - self._mean)) * self.scale

    # ------------------------------------------------------------------ #
    #  Frames
    # ------------------------------------------------------------------ #

    def _to_on_off(self, signal):
        on = np.maximum(signal, 0.0)
        off = np.maximum(-signal, 0.0)
        frame = np.stack([on, off], axis=-1)
        if self.time_last and frame.ndim == 4:
            frame = np.transpose(frame, (1, 2, 0, 3))   # (y, x, t, 2)
        return frame

    def get_current_frame(self):
        """One frame of the stream.

        In the default ``output_mode='frame'`` this returns ``(16, 16, 2)`` --
        the on/off image at the current time step.  The space-time whitener is
        still applied over a window of ``n_frames``; we simply read out a single
        frame of it, so the returned image is temporally band-passed (the static
        DC component is gone, which is what makes motion the dominant signal)
        while the interface stays frame-by-frame.  Call ``set_next_position()``
        to advance one frame through the clip.
        """
        patch = self._extract_patch()                    # (T, ph, pw), [0, 1]
        self._last_raw_patch = patch
        single = (self.output_mode == 'frame')

        if self.preprocessing == 'whiten':
            if not self._whitening_fitted:
                self.fit_whitening()
            if single:
                z = self._whiten_readout(patch.ravel()).reshape(
                    self.patch_size, self.patch_size)
            else:
                z = self._whiten(patch.ravel())
                if self.whitening == 'zca':
                    z = z.reshape(self.n_frames, self.patch_size, self.patch_size)
            return self._to_on_off(z)

        elif self.preprocessing == 'temporal-diff':
            # cheap, whitening-free alternative: local contrast + frame differencing.
            # Kills the static DC component so only moving structure survives.
            z = patch - patch.mean()
            z = z / (z.std() + 1e-6)
            z = np.diff(z, axis=0, prepend=z[:1])
            return self._to_on_off(z[self._readout_index] if single else z)

        elif self.preprocessing == 'gray-scale':
            z = patch - patch.mean()
            z = z / (z.std() + 1e-6)
            return self._to_on_off(z[self._readout_index] if single else z)

        else:
            raise Exception('Preprocessing not recognized')

    def get_current_block(self):
        """The full space-time patch ``(n_frames, 16, 16, 2)`` at the current
        position, whatever ``output_mode`` is set to."""
        mode, self.output_mode = self.output_mode, 'block'
        try:
            return self.get_current_frame()
        finally:
            self.output_mode = mode

    def get_blind_frame(self):
        return np.zeros(self.get_shape())

    # ------------------------------------------------------------------ #
    #  Labels
    # ------------------------------------------------------------------ #

    @staticmethod
    def _match_pair(a, b, max_disp):
        """Integer + parabolic-subpixel displacement taking frame ``a`` onto ``b``.

        Block matching rather than Lucas-Kanade: gradient methods break down
        above ~1 px/frame, and naturalistic footage routinely moves faster.
        """
        P = a.shape[0]
        a = (a - a.mean()) / (a.std() + 1e-6)
        b = (b - b.mean()) / (b.std() + 1e-6)
        n = 2 * max_disp + 1
        cost = np.empty((n, n))
        for i, dy in enumerate(range(-max_disp, max_disp + 1)):
            ay0, ay1 = max(0, -dy), min(P, P - dy)
            by0, by1 = max(0, dy), min(P, P + dy)
            for j, dx in enumerate(range(-max_disp, max_disp + 1)):
                ax0, ax1 = max(0, -dx), min(P, P - dx)
                bx0, bx1 = max(0, dx), min(P, P + dx)
                d = a[ay0:ay1, ax0:ax1] - b[by0:by1, bx0:bx1]
                cost[i, j] = np.mean(d * d)

        i, j = np.unravel_index(np.argmin(cost), cost.shape)
        dy, dx = i - max_disp, j - max_disp

        def refine(c0, c1, c2):                 # parabola through 3 samples
            denom = c0 - 2 * c1 + c2
            return 0.0 if abs(denom) < 1e-12 else np.clip(0.5 * (c0 - c2) / denom, -0.5, 0.5)

        if 0 < j < n - 1:
            dx += refine(cost[i, j - 1], cost[i, j], cost[i, j + 1])
        if 0 < i < n - 1:
            dy += refine(cost[i - 1, j], cost[i, j], cost[i + 1, j])
        return np.array([dx, dy]), float(cost[i, j])

    def _local_velocity(self, patch=None):
        """Local image velocity of the current patch in **pixels per frame**
        (x = rightwards, y = downwards, in the *downsampled* pixel grid).

        Median over consecutive frame pairs, so a single bad match doesn't
        dominate.  Useful as a ground-truth-ish target when you later probe how
        direction selective the learned cells actually are.
        """
        if patch is None:
            patch = self._last_raw_patch
        if patch is None:
            patch = self._extract_patch()
        if patch.shape[0] < 2:
            return np.zeros(2)

        max_disp = self.max_disp
        estimates = [self._match_pair(patch[t], patch[t + 1], max_disp)[0]
                     for t in range(patch.shape[0] - 1)]
        v = np.median(np.asarray(estimates), axis=0) / self.temporal_stride
        return v                                                # (vx, vy)

    def _direction_one_hot(self, v):
        one_hot = np.zeros(self.n_direction_bins)
        speed = float(np.hypot(*v))
        if speed < 1e-3:
            return one_hot
        angle = math.atan2(v[1], v[0]) % (2 * math.pi)
        idx = int(round(angle / (2 * math.pi) * self.n_direction_bins)) % self.n_direction_bins
        one_hot[idx] = 1.0
        return one_hot

    def get_label_for(self, label_category):
        match label_category:
            case 'category_string':
                return self.sequences_info[self.sequence_idx]['name']
            case 'category_one_hot':
                label = np.zeros(len(self.sequences_info))
                label[self.sequence_idx] = 1
                return label
            case 'velocity':
                return self._local_velocity()
            case 'speed':
                return float(np.hypot(*self._local_velocity()))
            case 'direction_one_hot':
                return self._direction_one_hot(self._local_velocity())
            case _:
                return None

    def get_all_labels(self):
        v = self._local_velocity()
        label_obj = dict()
        label_obj['category_string'] = self.sequences_info[self.sequence_idx]['name']
        label_obj['category_one_hot'] = np.zeros(len(self.sequences_info))
        label_obj['category_one_hot'][self.sequence_idx] = 1
        label_obj['velocity'] = v
        label_obj['speed'] = float(np.hypot(*v))
        label_obj['direction_one_hot'] = self._direction_one_hot(v)
        return label_obj

    def get_label_list(self):
        return ["category_string", "category_one_hot", "velocity", "speed",
                "direction_one_hot"]

    def get_category_string_from_one_hot(self, one_hot):
        return self.sequences_info[int(np.argmax(one_hot))]['name']

    # ------------------------------------------------------------------ #
    #  Metadata
    # ------------------------------------------------------------------ #

    def get_shape(self):
        if self.output_mode == 'frame':
            return (self.patch_size, self.patch_size, self.num_channels)
        if self.preprocessing == 'whiten' and self.whitening == 'svd':
            return (self.output_dim(), self.num_channels)
        if self.time_last:
            return (self.patch_size, self.patch_size, self.n_frames, self.num_channels)
        return (self.n_frames, self.patch_size, self.patch_size, self.num_channels)

    def get_hyperparameters(self):
        return {'vg_name': self.name,
                'source': self.source,
                'screen_size': self.x_size,
                'n_frames': self.n_frames,
                'output_mode': self.output_mode,
                'readout': self.readout,
                'temporal_stride': self.temporal_stride,
                'downsample': self.downsample,
                'preprocessing': self.preprocessing,
                'whitening': self.whitening,
                'n_components': self.n_components,
                'motion_percentile': self.motion_percentile,
                'max_disp': self.max_disp,
                'n_sequences': len(self.sequences_info)}

    def get_extensive_name(self):
        return (f"{self.name}_{self.patch_size}_{self.n_frames}"
                f"_{self.temporal_stride}_{self.output_mode}_{self.preprocessing}")

    def get_name(self):
        return self.name

