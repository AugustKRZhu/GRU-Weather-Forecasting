import datetime
import io
import os
import tarfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

download_dir = "data/weather-denmark/"
url = "https://github.com/Hvass-Labs/weather-denmark/raw/master/weather-denmark.tar.gz"
cities = ['Aalborg', 'Aarhus', 'Esbjerg', 'Odense', 'Roskilde']
path = os.path.join(download_dir, "weather-denmark-resampled.pkl")
if os.path.exists(path):
    df = pd.read_pickle(path)
else:
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    r = requests.get(url)
    print("Download finished. Extracting files.")
    tarfile.open(fileobj=io.BytesIO(r.content), mode="r:gz").extractall(download_dir)
    print("Done.")
    df_org = pd.read_pickle(os.path.join(download_dir, "weather-denmark.pkl"))
    df_cities = [df_org.xs(city) for city in cities]
    df_resampled = []
    for df_city in df_cities:
        df_res = df_city.dropna(axis=0, how='all')
        df_res = df_res.dropna(axis=1, how='all')
        df_res = df_res.resample('1T')
        df_res = df_res.interpolate(method='time')
        df_res = df_res.resample('60T')
        df_res = df_res.interpolate()
        df_res = df_res.dropna(how='all')
        df_resampled.append(df_res)
    df = pd.concat(df_resampled, keys=cities, axis=1, join='inner')
    df.to_pickle(path)

df.drop(('Esbjerg', 'Pressure'), axis=1, inplace=True)
df.drop(('Roskilde', 'Pressure'), axis=1, inplace=True)
df['Various', 'Day'] = df.index.dayofyear
df['Various', 'Hour'] = df.index.hour
df['Various', 'Minute'] = df.index.minute
target_city = 'Odense'
target_names = ['Temp', 'WindSpeed', 'Pressure']
shift_days = 1
shift_steps = shift_days * 24
df_targets = df[target_city][target_names].shift(-shift_steps)
x_data = df.values[0:-shift_steps]
y_data = df_targets.values[:-shift_steps]
num_data = len(x_data)
train_split = 0.9
num_train = int(train_split * num_data)
num_test = num_data - num_train
num_x_signals = x_data.shape[1]
num_y_signals = y_data.shape[1]
x_scaler = MinMaxScaler()
x_train_scaled = x_scaler.fit_transform(x_data[0:num_train])
x_test_scaled = x_scaler.transform(x_data[num_train:])
y_scaler = MinMaxScaler()
y_train_scaled = y_scaler.fit_transform(y_data[0:num_train])
y_test_scaled = y_scaler.transform(y_data[num_train:])


def batch_generator(batch_size, sequence_length):
    while True:
        x_shape = (batch_size, sequence_length, num_x_signals)
        x_batch = np.zeros(shape=x_shape, dtype=np.float16)

        y_shape = (batch_size, sequence_length, num_y_signals)
        y_batch = np.zeros(shape=y_shape, dtype=np.float16)

        for i in range(batch_size):
            idx = np.random.randint(num_train - sequence_length)
            x_batch[i] = x_train_scaled[idx:idx + sequence_length]
            y_batch[i] = y_train_scaled[idx:idx + sequence_length]
        yield x_batch, y_batch


batch_size = 32
sequence_length = 24 * 7 * 8
generator = batch_generator(batch_size=batch_size, sequence_length=sequence_length)
validation_data = (np.expand_dims(x_test_scaled, axis=0), np.expand_dims(y_test_scaled, axis=0))
model = tf.keras.Sequential()
model.add(tf.keras.layers.GRU(units=128, return_sequences=True, input_shape=(None, num_x_signals,)))
model.add(tf.keras.layers.GRU(units=128, return_sequences=True))
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(200, activation='relu')))
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_y_signals, activation='sigmoid')))
model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam())
model.summary()

callback_checkpoint = ModelCheckpoint(filepath='weather_checkpoint.keras', monitor='val_loss', verbose=1, save_weights_only=True, save_best_only=True)
callback_early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=1)
callback_tensorboard = TensorBoard(log_dir='.\\weather_logs\\', histogram_freq=0, write_graph=False)
callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, min_lr=1e-4,verbose=1, patience=3)
callbacks = [callback_early_stopping, callback_checkpoint, callback_tensorboard, callback_reduce_lr]
model.fit(generator, epochs=100, steps_per_epoch=100, validation_data=validation_data, verbose=0, callbacks=callbacks)


def plot_comparison(start_idx, length=100, train=True):
    if train:
        x = x_train_scaled
        y_true = y_data[0:num_train]
    else:
        x = x_test_scaled
        y_true = y_data[num_train:]

    end_idx = start_idx + length
    x = x[start_idx:end_idx]
    y_true = y_true[start_idx:end_idx]
    x = np.expand_dims(x, axis=0)
    y_pred = model.predict(x)
    y_pred_rescaled = y_scaler.inverse_transform(y_pred[0])

    for signal in range(len(target_names)):
        signal_pred = y_pred_rescaled[:, signal]
        signal_true = y_true[:, signal]
        plt.figure(figsize=(15, 5))
        plt.plot(signal_true, label='true')
        plt.plot(signal_pred, label='pred')
        plt.ylabel(target_names[signal])
        plt.legend()
        plt.savefig(
            str(target_names[signal]) + "{date:%Y-%m-%d_%H-%M-%S.%f}".format(date=datetime.datetime.now()) + str(
                "train" if train else "test") + ".png")
        plt.close()


plot_comparison(start_idx=100000, length=1000, train=True)
plot_comparison(start_idx=200, length=1000, train=False)
