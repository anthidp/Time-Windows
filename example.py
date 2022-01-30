""" Example of WindowGenerator application in Time Series Data. 
The aim is to predict the number of bicycle trips across Seattle's Fremont Bridge based on weather, season, and other factors.
The joined Dataset idea is from https://jakevdp.github.io/PythonDataScienceHandbook/05.06-linear-regression.html#Example:-Predicting-Bicycle-Traffic

"""

import pandas as pd
import tensorflow as tf
from window import  WindowGenerator
from matplotlib import pyplot as plt


window_size = 90
label_windth = 1
batch_size = 1000
input_length = window_size - label_windth
input_features = daily.shape[1]

epochs = 300
learning_rate = 0.9


def load(): 
  """ Data loading function."""
  
    fremont_bridge = 'https://data.seattle.gov/api/views/65db-xm6k/rows.csv?accessType=DOWNLOAD'
    
    bicycle_weather = 'https://raw.githubusercontent.com/jakevdp/PythonDataScienceHandbook/master/notebooks/data/BicycleWeather.csv'

    counts = pd.read_csv(fremont_bridge, index_col='Date', parse_dates=True, 
                         infer_datetime_format=True)

    weather = pd.read_csv(bicycle_weather, index_col='DATE', parse_dates=True, 
                          infer_datetime_format=True)

    daily = counts.resample('d').sum()
    daily['Total'] = daily.sum(axis=1)
    daily = daily[['Total']] # remove other columns

    weather_columns = ['PRCP', 'SNOW', 'SNWD', 'TMAX', 'TMIN', 'AWND']
    daily = daily.join(weather[weather_columns], how='inner')
    
    daily = daily.drop(index=daily.index[0])
    
    return daily

  
daily = load()

# Split of dataset into Train & Validation. 
n = len(daily)
train_df = daily[0:int(n*0.7)]
val_df = daily[int(n*0.7):]


# Call WindowGenerator and define parameters.
window = WindowGenerator( window_size,  label_windth, batch_size)

# Create dataset windows for each set.
train_dataset = window.make_dataset(train_df, ['Total'])
val_dataset = window.make_dataset(val_df, ['Total'])


# Keras Functional API with LSTM layers.
input = tf.keras.Input( shape=(input_length,input_features))
normalized_input = tf.keras.layers.Normalization(axis=-1)(input)
x = tf.keras.layers.LSTM(240, return_sequences=True)(normalized_input)
x = tf.keras.layers.LSTM(120, return_sequences=True)(x)
x = tf.keras.layers.LSTM(64)(x)
output = tf.keras.layers.Dense(units=(label_windth))(x)
model = tf.keras.Model(input, output, name="lstm")
model.summary()

model.compile(loss=tf.losses.MeanAbsoluteError(),
              metrics=[tf.metrics.MeanSquaredError()], 
              optimizer = tf.keras.optimizers.SGD(learning_rate= learning_rate))

history = model.fit(train_dataset, validation_data = val_dataset,  epochs = epochs, verbose = 0)


# Graphical Representation of Results.
history_dict = history.history

mae_loss = history_dict['loss']
val_mae_loss = history_dict['val_loss']
mse_loss = history_dict['mean_squared_error']
val_mse_loss = history_dict['val_mean_squared_error']


epochs = range(1, len(mae_loss) + 1)
plt.figure(figsize=(10, 6))
#fig.tight_layout()

plt.subplot(2, 1, 1)
# r is for "solid red line"
plt.plot(epochs, mae_loss, 'r', label='Training MAE loss')
# b is for "solid blue line"
plt.plot(epochs, val_mae_loss, 'b', label='Validation MAE loss')
plt.title('Training and validation MAE loss')
# plt.xlabel('Epochs')
plt.ylabel('MAE Loss')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(epochs, mse_loss, 'r', label='Training MSE loss')
plt.plot(epochs, val_mse_loss, 'b', label='Validation MSE loss')
plt.title('Training and validation MSE loss')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
