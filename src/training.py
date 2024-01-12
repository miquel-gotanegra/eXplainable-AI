import re
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from tensorflow import keras
from keras.utils import to_categorical





class Trainer:
    
    def __init__(self, name='ours', window_size=256, overlap=0.5, path=None, custom_path=None):
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
        
        self.custom_path = custom_path
            
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
    
    #------------- DATA PREPROCESSING -------------#    
    
    def __load_data(self):
        
        if not self.custom_path is None:
            data_name = self.path+self.custom_path + '/'
        else:
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
    
    def label_preprocessing(self):  
        print("Applying one hot  encoding to labels...")        
        label_encoder = LabelEncoder()
        self.labels = label_encoder.fit_transform(self.labels)

        # Check the order of classes
        print("Classes:",label_encoder.classes_)
        self.labels = to_categorical(self.labels)
        
        return label_encoder
        
    def flatten_minmax_preprocessing(self):
        X_train = self.sensor_data[self.idx_train].reshape(self.sensor_data[self.idx_train].shape[0], -1)
        X_test = self.sensor_data[self.idx_val].reshape(self.sensor_data[self.idx_val].shape[0], -1)

        # Initialize and fit the scaler on the data
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        return X_train, X_test
    
    def raw_minmax_preprocessing(self):
        X_train = self.sensor_data[self.idx_train].copy()
        X_val = self.sensor_data[self.idx_val].copy()

        # flatten dataset,
        train_flat = X_train.reshape((X_train.shape[0] * X_train.shape[1], X_train.shape[2]))
        val_flat = X_val.reshape((X_val.shape[0] * X_val.shape[1], X_val.shape[2]))
        print(train_flat.shape)

        s = MinMaxScaler()
        # fit on the dataset
        s.fit(train_flat)
        train_flat = s.transform(train_flat)
        val_flat = s.transform(val_flat)

        # Reshape data back to 3D: (num_samples, num_steps, num_features)
        X_train = train_flat.reshape(self.idx_train.shape[0], self.sensor_data.shape[1], self.sensor_data.shape[2])
        X_val = val_flat.reshape(self.idx_val.shape[0], self.sensor_data.shape[1], self.sensor_data.shape[2])
        
        return X_train, X_val
    
    def raw_standardization_preprocessing(self):

        X_train = self.sensor_data[self.idx_train].copy()
        X_val = self.sensor_data[self.idx_val].copy()

        # flatten dataset,
        train_flat = X_train.reshape((X_train.shape[0] * X_train.shape[1], X_train.shape[2]))
        val_flat = X_val.reshape((X_val.shape[0] * X_val.shape[1], X_val.shape[2]))

        s = StandardScaler()
        # fit on the dataset
        s.fit(train_flat)
        train_flat = s.transform(train_flat)
        val_flat = s.transform(val_flat)
        print("mean values:")
        print(s.mean_)
        print("var values:")
        print(s.scale_)

        # Reshape data back to 3D: (num_samples, num_steps, num_features)
        X_train = train_flat.reshape(self.idx_train.shape[0], self.sensor_data.shape[1], self.sensor_data.shape[2])
        X_val = val_flat.reshape(self.idx_val.shape[0], self.sensor_data.shape[1], self.sensor_data.shape[2])
        
        return X_train, X_val

    
    #------------- DEEP LEARNING ARCHITECTURES -------------#
    
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

    def MLP(self, num_neurons=[512, 256, 128], hidden_layers=3, num_classes = 11, act='relu', act_out = 'softmax'):
        layers = []
        layers.append(keras.layers.Dense(num_neurons[0], activation=act, input_shape=(self.window_size*9,)))
        for i in range(hidden_layers-1):
            layers.append(keras.layers.Dense(num_neurons[i+1], activation=act))
        layers.append(keras.layers.Dense(num_classes, activation=act_out))
            
        return keras.models.Sequential(layers)
    
    def __ConvBlock(self, previous_layer, n_filters=64, pad='causal', acti='relu'):
        conv_x = keras.layers.Conv1D(filters=n_filters, kernel_size=7, padding=pad, activation=acti)(previous_layer)
        conv_x = keras.layers.BatchNormalization()(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_filters, kernel_size=5, padding=pad,activation=acti)(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_filters, kernel_size=3, padding=pad, activation=acti)(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)
        
        shortcut_y = keras.layers.Conv1D(filters=n_filters, kernel_size=1, padding=pad, activation=acti)(previous_layer)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_1 = keras.layers.add([shortcut_y, conv_z])
        output_block_1 = keras.layers.MaxPooling1D(pool_size=2)(output_block_1)
        
        return output_block_1

    def ResNet(self, num_blocks=3, filer_size=32, input_shape=(256, 9), num_classes=11):
        input_layer = keras.layers.Input(input_shape)
        # Input block
        block1 = self.__ConvBlock(input_layer, filer_size)
        last_block = block1   
        for i in range(num_blocks-1):
            block_x = self.__ConvBlock(last_block, filer_size*(2**(i+1)))
            last_block = block_x
        # Final
        gap_layer = keras.layers.GlobalAveragePooling1D()(last_block)
        output_layer = keras.layers.Dense(num_classes, activation='softmax')(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        
        return model
    
    def LSTM(self, lstm_layer, n_labels, dropout_rate, input_shape=(256, 9), output_activation='softmax'):
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
    

    #------------- EVALUATION -------------#
    
    def training_plot(self, history):
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Categorical Crossentropy')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def model_plot(self, model):
        keras.utils.plot_model(model)
