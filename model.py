import pandas as pd
import numpy as np

raw_data = pd.read_csv('ETH-USD_data.csv')
data = raw_data.copy()

# Split data to training_set and validation_set
def split_train_validation_data(train_data, validation_ratio):
    n_samples = len(train_data)
    n_validation = int(n_samples * validation_ratio)

    validation_data = train_data[-n_validation:]
    train_data = train_data[:-n_validation]

    return train_data, validation_data

# Ratio
validation_ratio = 0.1  

# Splitting data
train_data, test_data = split_train_validation_data(data, validation_ratio)

# Normalization
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_data = train_data.iloc[:, 1:2]
train_scaled = sc.fit_transform(training_data)

# Dara structure
x_train = []
y_train = []

for i in range(60, 1412):
    x_train.append(train_scaled[i-60: i, 0])
    y_train.append(train_scaled[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build the model
from tensorflow.keras.layers import Dropout, Dense, LSTM
from tensorflow.keras.models import Sequential

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
dataset_train = train_data.copy()

# Stock price for test
dataset_test = test_data.copy()
real_ETH_price = dataset_test.iloc[:, 1:2].values

# Concatenate
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0).values
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].reshape(-1, 1)
inputs = sc.transform(inputs)

x_test = []
for i in range(60, 216):
    x_test.append(inputs[i-60: i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

pred_ETH_price = rnn.predict(x_test)
pred_ETH_price = sc.inverse_transform(pred_ETH_price)

# Visualization
import matplotlib.pyplot as plt
plt.plot(real_ETH_price, label = 'Real Price', color = 'blue')
plt.plot(pred_ETH_price, label = 'Predicted Price', color = 'red')
plt.xlabel('Time')
plt.ylabel('ETH Price')
plt.legend()
plt.show()

