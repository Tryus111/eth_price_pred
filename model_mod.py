import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, LSTM
from tensorflow.keras.models import Sequential

data = pd.read_csv('training_set.csv')
x_train = data.iloc[:, :-1].values
y_train = data.iloc[:, -1].values

# Reshape
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# models
lstm_layers = 50
dropout_rate = 0.2
output_layers = 1

rnn = Sequential([
    # First LSTM layers
    LSTM(lstm_layers, return_sequences = True, input_shape = (x_train.shape[1], 1)),
    Dropout(dropout_rate),
    
    # Second LSTM layers
    LSTM(lstm_layers, return_sequences = True),
    Dropout(dropout_rate),
    
    # Third LSTM layers
    LSTM(lstm_layers, return_sequences = True),
    Dropout(dropout_rate),
    
    # Fourth LSTM layers
    LSTM(lstm_layers),
    Dropout(dropout_rate),    
    
    # Output layers
    Dense(output_layers)
    
                ])
batch = 40
epoc = 125
rnn.compile(optimizer = 'adam', loss = 'mean_squared_error')
rnn.fit(x_train, y_train, batch_size = batch, epochs = epoc)

# Predict ETH price
# Stock price for train
dataset_train = pd.read_csv('data_train.csv')
unscaled_train_data = dataset_train.copy()

# Stock price for test
test_data = pd.read_csv('test_set.csv')
real_ETH_price = test_data.iloc[:, 1:2].values

# Concatenate
dataset_total = pd.concat((dataset_train['Open'], test_data['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(test_data) - 60:].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

