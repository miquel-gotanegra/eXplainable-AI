## AUTHOR: GERARD CARAVACA, 2023

import os
import numpy as np
import fnmatch
import pandas as pd
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
from scipy.fftpack import fft
from sklearn.decomposition import PCA
from joblib import dump, load
from sklearn.preprocessing import MinMaxScaler


class FeatureExtractor:
    
    def __init__(self, name='ours', window_size=256, overlap=0.5, path=None):
        if not re.match(r'^0\.\d{1,2}$', str(overlap)):
            raise Exception (" Overlap must be a float following the pattern 0.xx.")
        if not isinstance(window_size, int):
            raise Exception (" Window size must be an integer.")
        if not name in ['ours', 'SHL', 'TMD', 'Zagreb']:
            raise Exception (" the dataset name must be 'ours', 'SHL', 'TMD' or 'Zagreb'.")
        
        if path is None:
            if name == 'ours':
                self.path = '../data/'
            elif name == 'SHL':
                self.path = '../otherDatasets/SHL/'
            elif name == 'TMD':
                self.path = '../otherDatasets/TMD/'
            else:
                self.path = '../otherDatasets/Zagreb/'
        else : 
            self.path = path
            
            
        self.name = name
        self.window_size = window_size
        self.overlap = overlap
        
        self.sensor_data = []
        self.users = []
        self.labels = []
        
        self.features = []
        
        self.idx_train = []
        self.idx_val = []
        self.idx_test = []
        
        self.processed = False
        self.splitted = False
        
        print(f"Loading processed dataset with Window_size={self.window_size}, Overlap={self.overlap} and name={self.name}...")
        self.__load_data()
    
    #------------- DATA MANAGEMENT -------------#
    
    def __load_data(self):
        data_name = self.path+'Window'+str(self.window_size)+'Overlap'+str(self.overlap)+'/'

        if not os.path.exists(data_name):
            raise Exception(data_name+" do not exists in the system.")
        
        print("Please wait, this loading could last some minutes...")
        
        new_labels = np.loadtxt(data_name+'labels.txt', dtype=str)
        print(data_name+'labels.txt - loaded succesfully')
        new_users = np.loadtxt(data_name+'users.txt', dtype=str)
        print(data_name+'users.txt - loaded succesfully')
        self.idx_train = np.loadtxt(data_name+'train.txt', dtype=int)
        print(data_name+'train.txt - loaded succesfully')
        self.idx_val = np.loadtxt(data_name+'val.txt', dtype=int)
        print(data_name+'val.txt - loaded succesfully')
        self.idx_test = np.loadtxt(data_name+'test.txt', dtype=int)
        print(data_name+'test.txt - loaded succesfully')
        
        new_sensors = np.array([])
        shape0 = 0
        shape1 = 0
        shape2 = 0
        matches = ''
        for filename in os.listdir(data_name):
            if 'sensors' in filename:
                pattern = r'\((.*?)\)'
                # Use re.findall to find all matches in the input string
                matches = '('+re.findall(pattern, filename)[0]+')'
                shape0 = re.findall(pattern, filename)[0].split(',')[0]
                shape1 = re.findall(pattern, filename)[0].split(',')[1]
                shape2 = re.findall(pattern, filename)[0].split(',')[2]
        if matches != '':
            new_sensors = np.loadtxt(data_name+'sensors'+matches+'.txt')
            new_sensors = new_sensors.reshape((int(shape0), int(shape1), int(shape2)))
            print(data_name+'sensors'+matches+'.txt - loaded succesfully')

        self.sensor_data = new_sensors
        self.labels = new_labels
        self.users = new_users
        self.processed = True
        self.splitted = True 
    
    #------------- FEATURE EXTRACTION -------------#
        
    def __extract_features_from_sample(self, window):
        features = []

        # Statistical features for all axes
        means = np.mean(window, axis=0)
        medians = np.median(window, axis=0)
        stds = np.std(window, axis=0)
        vars = np.var(window, axis=0)
        mins = np.min(window, axis=0)
        maxs = np.max(window, axis=0)
        mean_crossings = np.mean(np.diff(window, axis=0), axis=0)

        # Frequency domain features for all axes
        mags = np.abs(fft(window, axis=0))
        mean_mags = np.mean(mags, axis=0)
        energy = np.sum(mags**2, axis=0) / len(mags)
        epsilon = 1e-10
        entropy = -np.sum(mags * np.log(mags + epsilon), axis=0)

        # Time domain features for all axes
        zero_crossings = np.sum(np.diff(np.sign(window), axis=0) != 0, axis=0)

        # Append all features
        for i in range(9):
            features.extend([means[i], medians[i], stds[i], vars[i], mins[i], maxs[i], mean_crossings[i], mean_mags[i], energy[i], entropy[i], zero_crossings[i]])
        
        return np.array(features)
    
    def extract_features(self):
        print('Extracting features...')
        progress_bar = tqdm(total=self.sensor_data.shape[0])
        for sample_id in range(0, self.sensor_data.shape[0]):
            progress_bar.update(1)  # Update the progress bar
            self.features.append(self.__extract_features_from_sample(self.sensor_data[sample_id,:,:]))
        
        progress_bar.close() 
        self.features = np.array(self.features)
        
        print("Resultant features shape:", self.features.shape)
    
    #------------- NORMALIZATION -------------#
        
    def __min_max_scaling(self, data, feature_range=(0, 1)):
        """
        Apply min-max scaling to the data.
        
        :param data: numpy array of shape (n_samples, n_features)
        :param feature_range: tuple (min, max) desired for scaling. Default is (0, 1)
        
        :return: scaled data, (min, max) before scaling for each feature
        """
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
        
        X_std = (data - min_vals) / (max_vals - min_vals)
        
        X_scaled = X_std * (feature_range[1] - feature_range[0]) + feature_range[0]
        
        return X_scaled, (min_vals, max_vals)

    def __z_score_normalization(self, data):
        """
        Apply z-score normalization to the data.
        
        :param data: numpy array of shape (n_samples, n_features)
        
        :return: standardized data, (mean, std_dev) before standardization for each feature
        """
        mean_vals = np.mean(data, axis=0)
        std_vals = np.std(data, axis=0)
        
        standardized_data = (data - mean_vals) / std_vals
        
        return standardized_data, (mean_vals, std_vals)
    
    def standarization(self):
        print("Applying z-score normalization...")
        st_data, norm_values = self.__z_score_normalization(self.features[self.idx_train, :])
        self.features[self.idx_train, :] = st_data
        
        self.features[self.idx_val, :] = (self.features[self.idx_val, :] - norm_values[0]) / norm_values[1]
        self.features[self.idx_test, :] = (self.features[self.idx_test, :] - norm_values[0]) / norm_values[1]
        print("Done.")
        
    def scaling(self, range=(-1, 1)):
        print("Applying min-max scaling...")
        st_data, norm_values = self.__min_max_scaling(self.features[self.idx_train, :], range)
        self.features[self.idx_train, :] = st_data
        
        val_std = (self.features[self.idx_val, :] - norm_values[0]) / (norm_values[1] - norm_values[0])
        self.features[self.idx_val, :] = val_std * (range[1] - range[0]) + range[0]

        test_std = (self.features[self.idx_test, :] - norm_values[0]) / (norm_values[1] - norm_values[0])
        self.features[self.idx_test, :] = test_std * (range[1] - range[0]) + range[0]
        print("Done.")
        
    #------------- DIMENSIONALITY REDUCTION -------------#

    def PCA(self, variance=0.95, filename='pca'):
        
        print("Original number of features:", self.features.shape[1])
        print("Applying PCA reduction...")
        pca = PCA(n_components=variance)
        pca.fit(self.features[self.idx_train])
        
        self.features = pca.transform(self.features)
        
        print("Reduced number of features after PCA:", self.features.shape[1])
        
        dire = self.path+'Window'+str(self.window_size)+'Overlap'+str(self.overlap)+'/models/'

        if not os.path.exists(dire):
            os.makedirs(dire)

        
        dump(pca, dire+filename+'.pkl')
        print(f"PCA model has been saved to {dire+filename+'.pkl'}")
        
        
# Testing
    def LSTM(lstm_layer=[256, 512], n_labels=11, dropout_rate=0.2, input_shape=(512, 9), output_activation='softmax'):
        model = keras.models.Sequential()
        if len(lstm_layer) > 1:
            model.add(keras.layers.LSTM(lstm_layer[0], return_sequences=True, input_shape=input_shape))
            model.add(keras.layers.Dropout(dropout_rate))
            for i in range(len(lstm_layer)-1):
                model.add(keras.layers.LSTM(lstm_layer[i+1]))
                model.add(keras.layers.Dropout(dropout_rate))
        else:
            model.add(keras.layers.LSTM(lstm_layer[0], input_shape=input_shape))
            model.add(keras.layers.Dropout(dropout_rate))

        model.add(keras.layers.Dense(n_labels, activation=output_activation))

        return model
        
        
    
        
        
