## AUTHOR: GERARD CARAVACA, 2023
import os
import numpy as np
import fnmatch
import pandas as pd
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import random
from scipy.ndimage import gaussian_filter1d

class Dataset:
    
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
            
        self.sensor_data = []
        self.users = []
        self.labels = []
        
        self.idx_train = []
        self.idx_val = []
        self.idx_test = []
        
        self.name = name
        self.window_size = window_size
        self.overlap = overlap
        self.processed = False
        self.splitted = False
        self.smoothed = False

    #------------- DATA MANAGEMENT -------------#

    def read_raw_data(self):
        raw_path = self.path+'raw'
        files = os.listdir(raw_path)
        
        if self.name == 'ours':
            print('Reading raw data files...')
            progress_bar = tqdm(total=len(files))
            for i, file in enumerate(files):
                progress_bar.update(1)  # Update the progress bar
                #print('Reading files: ' + str(i+1) + '/' + str(len(files)), end='\r')
                if fnmatch.fnmatch(file, '*_ACC-MAG-GYR*.csv') and os.path.getsize(raw_path + '/' + file) > 0:
                    df = pd.read_csv(raw_path + '/' + file)
                    filename = file.split('_')
                    n_samples = df.shape[0]

                    start = 0
                    end = self.window_size
                    while end < n_samples:

                        if filename[2] == "ACC-MAG-GYR":
                            l = filename[1]
                        else : l = filename[2]

                        if l == 'E-Scooter':
                            l = 'e-Scooter'

                        if l == 'Bici':
                            l = 'Bike'
                        
                        if l == 'e-Bicing':
                            l = 'e-Bike'

                        self.labels.append(l)
                        self.users.append(filename[0])
                        self.sensor_data.append(df.iloc[start:end, 1:10].values)
                        start = int(start + self.window_size*(1-self.overlap))
                        end = int(end + self.window_size*(1-self.overlap))    
            progress_bar.close()    
        elif self.name == 'Zagreb':
            print('Reading raw data files...')
            progress_bar = tqdm(total=len(files))
            for i, file in enumerate(files):
                #print('Reading files: ' + str(i+1) + '/' + str(len(files)), end='\r')
                progress_bar.update(1)  # Update the progress bar
                user = file.split('_')[0]
                df = pd.read_csv(raw_path + '/' + file)
                cls = df.label.unique()

                for c in cls:
                    c_df = df[df['label'] == c]
                    s_df = c_df[['accX','accY','accZ', 'gyroX', 'gyroY', 'gyroZ', 'magnX', 'magnY', 'magnZ']]
                    if c == 'E scooter':
                        c = 'E-scooter'
                    start = 0
                    end = self.window_size
                    n_samples = c_df.shape[0]
                    while end < n_samples:

                            self.labels.append(c)
                            self.users.append(int(user))
                            self.sensor_data.append(s_df.iloc[start:end].values)
                            start = int(start + self.window_size*(1-self.overlap))
                            end = int(end + self.window_size*(1-self.overlap))
            progress_bar.close() 
        elif self.name == 'SHL':
            #raise Exception("Reading raw data function not already implemented for SHL dataset.")
            progress_bar = tqdm(total=len(files)*12)
            for user in files:
                if fnmatch.fnmatch(user, 'User*'):
                    files_in = os.listdir(raw_path+'/'+user+'/')
                    for day in files_in:
                        files_data = os.listdir(raw_path+'/'+user+'/'+day+'/')
                        labels = np.loadtxt(raw_path+'/'+user+'/'+day+'/Label.txt')
                        for data in files_data:
                            if fnmatch.fnmatch(data, '*_Motion*'):
                                progress_bar.update(1)  # Update the progress bar
                                print(user + '  ' + day + ' ' + data)
                                sensor = np.loadtxt(raw_path+'/'+user+'/'+day+'/'+data)
                                labels_ = labels[labels[:,1] != 0]
                                sensor_ = sensor[labels[:,1] != 0]

                                cls = np.unique(labels_[:,1])
                                for c in cls:
                                    c_df = sensor_[labels_[:,1] == c]
                                    s_df = c_df[:, [4,5,6,7,8,9,17,18,19]]
                                    start = 0
                                    end = self.window_size
                                    n_samples = c_df.shape[0]
                                    while end < n_samples:
                                        self.labels.append(c)
                                        self.users.append(user)
                                        self.sensor_data.append(s_df[start:end, :])
                                        start = int(start + self.window_size*(1-self.overlap))
                                        end = int(end + self.window_size*(1-self.overlap))
            progress_bar.close()
        else:
            raise Exception("Reading raw data function not already implemented for TMD dataset.")
        
        self.sensor_data = np.array(self.sensor_data, dtype=float)
        self.users = np.array(self.users)
        self.labels = np.array(self.labels)
        self.processed = True
        
        print("Succesfully data preprocessing.")

    def save_data(self, extra_name=None):
        if not self.processed:
            raise Exception("Not available preprocessed data.")
        if not self.splitted:
            raise Exception("Sensor data must be splitted before this operation.")
        
        if not extra_name is None:
            data_name = self.path+'Window'+str(self.window_size)+'Overlap'+str(self.overlap)+'/'+'_'+extra_name
        else:
            data_name = self.path+'Window'+str(self.window_size)+'Overlap'+str(self.overlap)+'/'

        if not os.path.exists(data_name):
            os.makedirs(data_name)

        np.savetxt(data_name+'labels.txt' ,self.labels, fmt='%s')
        print(data_name+'labels.txt - saved succesfully')
        np.savetxt(data_name+'users.txt' ,self.users, fmt='%s')   
        print(data_name+'users.txt - saved succesfully')
        np.savetxt(data_name+'train.txt' ,self.idx_train, fmt='%s')   
        print(data_name+'train.txt - saved succesfully')
        np.savetxt(data_name+'val.txt' ,self.idx_val, fmt='%s')   
        print(data_name+'val.txt - saved succesfully')
        np.savetxt(data_name+'test.txt' ,self.idx_test, fmt='%s')   
        print(data_name+'test.txt - saved succesfully')
        
        shape = self.sensor_data.shape
        # Flatten the 3D array to 2D
        flattened_sensors = self.sensor_data.reshape(-1, self.sensor_data.shape[-1])
        np.savetxt(data_name+'sensors'+str(shape)+'.txt' ,flattened_sensors, fmt='%s')
        print(data_name+'sensors'+str(shape)+'.txt - saved succesfully')
        
    def load_data(self):
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

    #------------- DATA AUGMENTATION -------------#
    
    def __generate_random_rotation_angles(self):
        angle_x = random.uniform(0, 180)
        angle_y = random.uniform(0, 180)
        angle_z = random.uniform(0, 180)
        return angle_x, angle_y, angle_z

    def __create_rotation_matrix(self, angle_x, angle_y, angle_z):
        rad_x = np.radians(angle_x)
        rad_y = np.radians(angle_y)
        rad_z = np.radians(angle_z)
        
        Rx = np.array([[1, 0, 0],
                    [0, np.cos(rad_x), -np.sin(rad_x)],
                    [0, np.sin(rad_x), np.cos(rad_x)]])
        
        Ry = np.array([[np.cos(rad_y), 0, np.sin(rad_y)],
                    [0, 1, 0],
                    [-np.sin(rad_y), 0, np.cos(rad_y)]])
        
        Rz = np.array([[np.cos(rad_z), -np.sin(rad_z), 0],
                    [np.sin(rad_z), np.cos(rad_z), 0],
                    [0, 0, 1]])
        
        rotation_matrix = Rz.dot(Ry).dot(Rx)
        return rotation_matrix

    def __augment_data_with_rotation(self, sample, num_augmentations):
        augmented_data = []
        #num_augmentations = 1  # You can adjust the number of augmentations as needed
        
        for _ in range(num_augmentations):
            angle_x, angle_y, angle_z = self.__generate_random_rotation_angles()
            rotation_matrix = self.__create_rotation_matrix(angle_x, angle_y, angle_z)
            
            augmented_sample = np.zeros_like(sample)
            for i in range(sample.shape[0]):
                sensor_reading = sample[i].reshape(3, 3)  # Reshape to (3, 3) for rotation
                augmented_reading = np.dot(sensor_reading, rotation_matrix.T)
                augmented_sample[i] = augmented_reading.flatten()
            
            augmented_data.append(augmented_sample)
        
        return augmented_data

    def __add_additive_noise(self, sensor_data, noise_level=0.01):
        noise = np.random.normal(0, noise_level, sensor_data.shape)
        return np.add(sensor_data, noise)

    def __add_multiplicative_noise(self, sensor_data, noise_level=0.01):
        noise = np.random.normal(1, noise_level, sensor_data.shape)
        return sensor_data * noise

    def __augment_data_with_obfuscation(self, sensor_data, num_augmentations, noise_level=0.01):
        augmented_data = []
        
        for _ in range(num_augmentations):
            # Apply additive noise
            augmented_sample_additive = self.__add_additive_noise(sensor_data, noise_level)
            
            # Apply multiplicative noise
            augmented_sample_multiplicative = self.__add_multiplicative_noise(sensor_data, noise_level)
            
            augmented_data.append(augmented_sample_additive)
            augmented_data.append(augmented_sample_multiplicative)
        
        return augmented_data
    
    def augment_data(self, rot_target, obf_perc=0.2, set='Train'):
        unique_labels = np.unique(self.labels)
        
        new_samples = []
        new_labels = []
        new_users = []
        new_train = []

        if set == 'Train':
            sensor_idx = self.idx_train.copy()
        else:
            sensor_idx = np.arange(self.sensor_data.shape[0])
            
        print("Applying rotation augmentation...")
        progress_bar = tqdm(total=len(unique_labels))
        last_idx = self.sensor_data.shape[0]-1
        # Perform data augmentation with rotation
        for label in unique_labels:
            label_samples = [sample for sample, sample_label in zip(sensor_idx, self.labels[sensor_idx]) if sample_label == label]
            # Calculate the desired number of samples for this label
            target_samples_per_label = rot_target  # Set this based on your desired label balance
            # Calculate the number of augmentation rounds required for this label
            needed = target_samples_per_label - len(label_samples)
            if (needed > 0):
                data_to_augment = random.choices(label_samples, k=needed)
                for idx in data_to_augment:
                        augmented_sample = self.__augment_data_with_rotation(self.sensor_data[idx].copy(), 1)
                        new_samples.append(augmented_sample[0])
                        new_labels.append(self.labels[idx])
                        new_users.append(self.users[idx])
                        if set == 'Train':
                            new_train.append(last_idx+1)
                        last_idx += 1
                        
                            
            progress_bar.update(1)  # Update the progress bar       
        progress_bar.close()        
        # mix data
        augmented_data = list(self.sensor_data)+new_samples
        self.sensor_data = np.array(augmented_data, dtype=float)
        
        augmented_labels = list(self.labels)+new_labels
        self.labels = np.array(augmented_labels)
        
        augmented_users = list(self.users)+new_users
        self.users = np.array(augmented_users)
        
        augmented_train = list(self.idx_train)+new_train
        self.idx_train = np.array(augmented_train)
              
        new_samples = []
        new_labels = []
        new_users = []
        new_train = []      
        
        print("Applying obfuscation augmentation...")
        progress_bar = tqdm(total=len(unique_labels))
        # Perform data augmentation with obfuscation
        for label in unique_labels:
            label_samples = [sample for sample, sample_label in zip(sensor_idx, self.labels) if sample_label == label]
            # Calculate the desired number of samples for this label
            target_samples_per_label = int(len(label_samples) + len(label_samples)*obf_perc)  # Set this based on your desired label balance
            # Calculate the number of augmentation rounds required for this label
            needed = target_samples_per_label - len(label_samples)
        
            if (needed > 0):
                data_to_augment = random.sample(label_samples, k=needed)
                for idx in data_to_augment:
                    augmented_sample = self.__augment_data_with_obfuscation(self.sensor_data[idx].copy(), 1, round(random.uniform(0.01,0.70), 2))
                    new_samples.append(augmented_sample[0])
                    new_labels.append(self.labels[idx])
                    new_users.append(self.users[idx]) 
                    if set == 'Train':
                        new_train.append(last_idx+1)
                    last_idx += 1
            progress_bar.update(1)  # Update the progress bar
        progress_bar.close()
        
        # mix data
        augmented_data = list(self.sensor_data)+new_samples
        self.sensor_data = np.array(augmented_data, dtype=float)
        
        augmented_labels = list(self.labels)+new_labels
        self.labels = np.array(augmented_labels)
        
        augmented_users = list(self.users)+new_users
        self.users = np.array(augmented_users)
        
        augmented_train = list(self.idx_train)+new_train
        self.idx_train = np.array(augmented_train)
        
        print("Data augmentation performed succesfully - new data shape is", self.sensor_data.shape)
        print("Train:", self.idx_train.shape[0])
        print("Validation:", self.idx_val.shape[0]) 
        print("Test:", self.idx_test.shape[0])        
    
    #------------- UNDERSAMPLING -------------#
    
    def random_undersampling(self, label, desired_samples):
        majority_indices = np.where(self.labels == label)[0]

        # Randomly select a subset of indices from the majority class
        desired_subset_size = majority_indices.shape[0] - desired_samples  # Adjust as needed
        random_subset_indices = np.random.choice(majority_indices, size=desired_subset_size, replace=False)

        self.sensor_data = np.delete(self.sensor_data, random_subset_indices, axis=0)
        self.labels = np.delete(self.labels, random_subset_indices, axis=0)
        self.users = np.delete(self.users, random_subset_indices, axis=0)
        
        print("Label: " + str(label) + " - has been undersampled by " + str(desired_subset_size) + " elements.")
    
    #------------- SMOOTHING -------------#
    
    def __gaussian_soothing(self, sample, sigma=1):
        smoothed_data = np.apply_along_axis(gaussian_filter1d, 0, sample, sigma)
        return smoothed_data
    
    def smooth_data(self, sigma=1):
        if not self.processed:
            raise Exception("Sensor data must be processed before this operation.")
        if self.smoothed:
            raise Exception("Sensor data is already smoothed.")
        
        print('Smoothing samples...')
        progress_bar = tqdm(total=self.sensor_data.shape[0])
        for sample_id in range(self.sensor_data.shape[0]):
            self.sensor_data[sample_id,:,:] = self.__gaussian_soothing(self.sensor_data[sample_id,:,:], sigma)
            progress_bar.update(1)  # Update the progress bar
        progress_bar.close()
        
        self.smoothed = True
    
    def plot_overlap_data(self, sample, smoothed_sample, title):
        num_sensors = 3
        num_axes = 3
        sensor_names = ['ACC', 'GYR', 'MAG']
        axes_labels = ['X', 'Y', 'Z']

        fig, axes = plt.subplots(num_sensors, num_axes, figsize=(12, 8), sharex=True, sharey=False)
        fig.suptitle(title)

        for i in range(num_sensors):
            for j in range(num_axes):
                ax = axes[i, j]
                z = i*3+j
                ax.plot(sample[:, z], label='Original', alpha=0.7)
                ax.plot(smoothed_sample[:, z], label='Smoothed', linestyle='--')
                ax.set_title(f'{sensor_names[i]} - {axes_labels[j]}')
                ax.legend()

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust subplot layout
        plt.show()
        
                
    #------------- TRAIN/VALIDATION/TEST SPLIT -------------#

    def split_sets(self, train_ratio=0.7, test_ratio=0.3):
        
        if train_ratio + test_ratio != 1:
            raise Exception("train_ratio, validation_ratio, test_ratio must sum to 1.")
            
        if not self.processed:
            raise Exception("Sensor data must be processed before this operation.")

        sensor_idx = np.arange(self.sensor_data.shape[0])
        
        X_train, X_rest, _, _, _, _ = train_test_split(
            list(sensor_idx), self.labels, self.users,
            test_size=test_ratio, random_state=42
        )
        
        X_val, X_test, _, _, _, _ = train_test_split(
            list(X_rest), self.labels[X_rest].copy(), self.users[X_rest].copy(),
            test_size=0.5, random_state=42
        )
        
        self.idx_test = np.array(X_test)
        self.idx_val = np.array(X_val)
        self.idx_train = np.array(X_train)
        
        print("Train data shape:", self.idx_train.shape)
        print("Validation data shape:", self.idx_val.shape)
        print("Test data shape:", self.idx_test.shape)
        
        self.splitted = True
    
    def smart_split_sets(self, train_ratio=0.7, test_ratio=0.3):
        label_users = {}
        for label, user in zip(self.labels, self.users):
            # Check if the label is already in the dictionary
            if label in label_users:
                # If yes, add the user to the set of users for that label
                label_users[label].add(user)
            else:
                # If no, create a new set with the user and associate it with the label
                label_users[label] = {user}
                    
        # Create lists to store labels and unique user counts
        unique_labels = []
        unique_user_counts = []

        # Iterate through the dictionary to count unique users for each label
        for label, user_set in label_users.items():
            unique_labels.append(label)
            unique_user_counts.append(len(user_set))
            
        label_user_samples = {}

        # Iterate through the labels and users
        for label, user in zip(self.labels, self.users):
            # Check if the label is already in the dictionary
            if label in label_user_samples:
                if user in label_user_samples[label]:
                    # If yes, increment the sample count for that user within the label
                    label_user_samples[label][user] += 1
                else:
                    # If no, create a new entry for the user and set the sample count to 1
                    label_user_samples[label][user] = 1
            else:
                # If the label is not in the dictionary, create a new entry for the label and user
                label_user_samples[label] = {user: 1}
        
        np.random.seed(42)
        sensor_idx = np.arange(self.sensor_data.shape[0])
        train_idx = set()
        test_idx = set()

        #first split train/test taking into account user distribution.
        for label in np.unique(self.labels):
            
            print('######### '+label+' #########')
            av_users = label_user_samples[label]
            total_samples = sum(av_users.values())
            train_samples = int(total_samples * train_ratio)
            test_samples = int(total_samples * test_ratio)
            user_keys = list(av_users.keys())
            np.random.shuffle(user_keys)
            
            print('     - Total samples:', total_samples)
            print('     - Train samples:', train_samples)
            print('     - Test samples:', test_samples)

            # Initialize sets for training and testing users and counters for samples
            train_users = set()
            test_users = set()
            train_samples_count = 0
            test_samples_count = 0
            
            # Iterate through the shuffled user keys and assign users to sets while respecting the sample counts
            for user in user_keys:
                if train_samples_count < train_samples:
                    train_users.add(user)
                    train_samples_count += av_users[user]
                elif test_samples_count < test_samples and user not in train_users:
                    test_users.add(user)
                    test_samples_count += av_users[user]

            # Ensure all users are included in either the training or testing set
            remaining_users = set(av_users.keys()) - (train_users | test_users)

            # Distribute remaining users alternately between training and testing sets
            for user in remaining_users:
                if len(train_users) < len(test_users):
                    train_users.add(user)
                else:
                    test_users.add(user)

            # Create training and testing datasets based on the selected users
            train_data = {user: av_users[user] for user in train_users}
            test_data = {user: av_users[user] for user in test_users}
            
            print('     - Train dist:', train_data)
            print('     - Test dist:', test_data)

            train_idx_ = []
            for usr in train_data:
                train_idx_.append(list(sensor_idx[(self.users == usr) & (self.labels == label)]))
            train_idx_ = set([item for row in train_idx_ for item in row])
            
            test_idx_ = []
            for usr in test_data:
                test_idx_.append(list(sensor_idx[(self.users == usr) & (self.labels == label)]))
            test_idx_ = set([item for row in test_idx_ for item in row])
            
            
            print('     - Mid train samp:', len(train_idx_))
            print('     - Mid test samp:', len(test_idx_))
            
            if len(test_idx_) < test_samples :
                print('     - Still needed test samples:', (test_samples-len(test_idx_)))
                news = random.sample(list(train_idx_), k=(test_samples-len(test_idx_)))
                test_idx_ = test_idx_.union(set(news))
                train_idx_ -= set(news)

            test_idx = test_idx.union(test_idx_)
            train_idx = train_idx.union(train_idx_)
            
        print('############# FINAL #############')
        print('TRAIN:', len(train_idx))
        print('TEST:', len(test_idx))
        self.idx_train = np.array(list(train_idx))

        test_user = [self.users[i] for i in test_idx]
        test_label = [self.labels[i] for i in test_idx]

        X_val, X_test, _, _, _, _ = train_test_split(
            list(test_idx), test_label, test_user,
            test_size=0.5, random_state=42
        )

        self.idx_test = np.array(X_test)
        self.idx_val = np.array(X_val)

        print("Train data shape:", self.idx_train.shape)
        print("Validation data shape:", self.idx_val.shape)
        print("Test data shape:", self.idx_test.shape)
        
        self.splitted = True
    
    def get_sets(self):
        if not self.splitted:
            raise Exception("You must split data first using split_sets() function.")
        
        return self.sensor_data[self.idx_train, :, :], self.labels[self.idx_train], self.sensor_data[self.idx_val, :, :], self.labels[self.idx_val], self.sensor_data[self.idx_test, :, :], self.labels[self.idx_test] 
         
    def oversample_minority_classes(self, X, y, min_samples=2):
        unique_labels, label_counts = np.unique(y, return_counts=True)
        oversampled_X = []
        oversampled_y = []
        
        for label in unique_labels:
            label_indices = np.where(y == label)[0]
            if label_indices.shape[0] < min_samples:
                # Oversample the minority class by duplicating samples
                oversampled_indices = np.random.choice(label_indices, min_samples, replace=True)
            else:
                oversampled_indices = label_indices
            
            oversampled_X.extend(X[oversampled_indices])
            oversampled_y.extend(y[oversampled_indices])
        
        return np.array(oversampled_X), np.array(oversampled_y)
            
    #------------- PLOTTING -------------#

    def plot_frequency_chart(self, sample, title="Frequency Chart"):
        if not self.processed:
            raise Exception("Not available preprocessed data.")
        
        num_sensors = 3
        num_axes = 3
        sensor_names = ['ACC', 'GYR', 'MAG']
        axes_labels = ['X', 'Y', 'Z']

        fig, axes = plt.subplots(num_sensors, num_axes, figsize=(12, 8), sharex=True, sharey=False)
        fig.suptitle(title)

        for i in range(num_sensors):
            for j in range(num_axes):
                ax = axes[i, j]
                z = i*3+j
                ax.plot(sample[:, z])
                ax.set_title(f'{sensor_names[i]} - {axes_labels[j]}')
                ax.grid(False)

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust subplot layout
        plt.show()
    
    def plot_class_distribution(self, set_type=None, path='../img/class_dist.png', title='Dataset Distribution'):
        if not self.processed:
            raise Exception("Not available preprocessed data.")
        
        if not set_type is None and not self.splitted:
            raise Exception("Not available splitted sets.")
        
        labels_ = self.labels
        
        if not set_type is None:
            if set_type == 'train':
                labels_ = self.labels[self.idx_train]
            elif set_type == 'val':
                labels_ = self.labels[self.idx_val]
            elif set_type == 'test':
                labels_ = self.labels[self.idx_test]
            else:
                raise Exception("set_type variable must be: 'train', 'val', 'test' or None")
        
        unique_labels, label_counts = zip(*zip(*np.unique(labels_, return_counts=True)))

        plt.figure(figsize=(8, 5))
        bars = plt.bar(unique_labels, label_counts, color = 'orange')
        plt.xlabel('Transport')
        plt.ylabel('Number of samples')
        plt.title(title)
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

        # Annotate each bar with its count
        for bar, count in zip(bars, label_counts):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(count), ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(path)
    
    def plot_user_distribution(self, set_type=None, path='../img/user_dist.png'):
        if not self.processed:
            raise Exception("Not available preprocessed data.")
        
        if not set_type is None and not self.splitted:
            raise Exception("Not available splitted sets.")
        
        total_users = self.users
        users_ = self.users
        
        if not set_type is None:
            if set_type == 'train':
                users_ = self.users[self.idx_train]
            elif set_type == 'val':
                users_ = self.users[self.idx_val]
            elif set_type == 'test':
                users_ = self.users[self.idx_test]
            else:
                raise Exception("set_type variable must be: 'train', 'val', 'test' or None")
        
        # Create a list of all possible user names (assuming unique names for each user).
        all_user_names = np.unique(total_users.astype(str))

        # Initialize counts for all user names to 0.
        counts = {}
        for usr in all_user_names:
            counts[usr] = 0
            

        # Count the frequency of each user name in the user data.
        user_counts = {}
        for user in users_:
            if user in user_counts:
                user_counts[user] += 1
            else:
                user_counts[user] = 1

        # Update the counts list with the counts of actual users.
        for user, count in user_counts.items():
            counts[user] = count

        # Create a bar plot.
        plt.figure(figsize=(12, 6))
        plt.bar(all_user_names, counts.values(), color='blue', alpha=0.7)

        # Add labels and title
        plt.xlabel('User Names')
        plt.ylabel('Count')
        plt.title('User Distribution')

        # Optionally, you can rotate the x-axis labels for better visibility.
        plt.xticks(rotation=45)

        # Show the plot
        plt.savefig(path)  
    
    def users_distribution(self, set_type=None, path='../img/users_dist.png'):
        if not self.processed:
            raise Exception("Not available preprocessed data.")
        
        labels_ = self.labels
        users_ = self.users
        
        if not set_type is None:
            if not self.splitted:
                raise Exception("You must split data first using split_sets() function.")
            
            if set_type == 'train':
                labels_ = self.labels[self.idx_train]
                users_ = self.users[self.idx_train]
            elif set_type == 'val':
                labels_ = self.labels[self.idx_val]
                users_ = self.users[self.idx_val]
            elif set_type == 'test':
                labels_ = self.labels[self.idx_test]
                users_ = self.users[self.idx_test]
            else:
                raise Exception("set_type variable must be: 'train', 'val', 'test' or None")
            
        
        # Initialize a dictionary to store unique users for each label
        label_users = {}

        # Iterate through the labels and users
        for label, user in zip(labels_, users_):
            if user != 'null':
                # Check if the label is already in the dictionary
                if label in label_users:
                    # If yes, add the user to the set of users for that label
                    label_users[label].add(user)
                else:
                    # If no, create a new set with the user and associate it with the label
                    label_users[label] = {user}


        # Create lists to store labels and unique user counts
        unique_labels = []
        unique_user_counts = []

        # Iterate through the dictionary to count unique users for each label
        for label, user_set in label_users.items():
            unique_labels.append(label)
            unique_user_counts.append(len(user_set))

        # Create a bar chart
        plt.figure(figsize=(8, 5))
        bars = plt.bar(unique_labels, unique_user_counts, color = 'orange')
        plt.xlabel('Labels')
        plt.ylabel('Unique User Count')
        plt.title('Unique Users for Each Label')
        plt.xticks(rotation=45)
        plt.tight_layout()

            # Annotate each bar with its count
        for bar, count in zip(bars, unique_user_counts):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(count), ha='center', va='bottom')

        # Display the plot
        plt.savefig(path)
        
    def users_class_distribution(self, desired_label, path='../img/class_dist.png'):
        if not self.processed:
            raise Exception("Not available preprocessed data.")
        
        # Initialize a dictionary to store sample counts for each user within each label
        label_user_samples = {}

        # Iterate through the labels and users
        for label, user in zip(self.labels, self.users):
            # Check if the label is already in the dictionary
            if label in label_user_samples:
                if user in label_user_samples[label]:
                    # If yes, increment the sample count for that user within the label
                    label_user_samples[label][user] += 1
                else:
                    # If no, create a new entry for the user and set the sample count to 1
                    label_user_samples[label][user] = 1
            else:
                # If the label is not in the dictionary, create a new entry for the label and user
                label_user_samples[label] = {user: 1}

        # Create a bar chart of the sample counts per user for a specific label (e.g., 'Label1')
        specific_label = desired_label
        users_specific_label = list(label_user_samples[specific_label].keys())
        sample_counts_specific_label = list(label_user_samples[specific_label].values())

        plt.figure(figsize=(8, 5))
        bars = plt.bar(users_specific_label, sample_counts_specific_label, color = 'orange')
        plt.xlabel('Users')
        plt.ylabel('Sample Count')
        plt.title(f'Sample Counts for Users within {specific_label}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        # Annotate each bar with its count
        for bar, count in zip(bars, sample_counts_specific_label):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(count), ha='center', va='bottom')

        # Display the plot
        plt.savefig(path)
 
 