import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

download_dir = "dataset/"
path = os.path.join(download_dir, "3317898170868.pkl")
if os.path.exists(path):
    df = pd.read_pickle(path)
else:
    df = pd.read_csv(download_dir + '3317898170868.txt', delim_whitespace=True)
    df = df.filter(['YR--MODAHRMN', 'TEMP'])
    df = df.rename(columns={df.columns[0]: "time",  df.columns[1]: 'temp'})
    df['time'] = pd.to_datetime(df['time'], format='%Y%m%d%H%M')
    df = df.set_index(pd.DatetimeIndex(df['time']))
    df = df.drop(['time'], axis=1)
    df = df[df['temp'] != '****']
    df = df.astype(int)
    df.to_pickle(path)
df['Various', 'Day'] = df.index.dayofyear
df['Various', 'Hour'] = df.index.hour
df['Various', 'Minute'] = df.index.minute
print(df.head())
shift_days = 1
shift_steps = shift_days * 24
target_names = ['temp']
df_targets = df[target_names].shift(-shift_steps)
x_data = df.values[0:-shift_steps]
y_data = df_targets.values[:-shift_steps]

num_data = len(x_data)
train_split = 0.8
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
print('Training data:')
print('x_data: ', x_train_scaled.shape)
print('y_data: ', y_train_scaled.shape)
print('Test data:')
print('x_data: ', x_test_scaled.shape)
print('y_data: ', y_test_scaled.shape)


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
sequence_length = 24 * 7 * 16
generator = batch_generator(batch_size=batch_size, sequence_length=sequence_length)
validation_data = (np.expand_dims(x_test_scaled, axis=0), np.expand_dims(y_test_scaled, axis=0))
model = tf.keras.Sequential()
model.add(tf.keras.layers.GRU(units=128, return_sequences=True, input_shape=(None, num_x_signals)))
model.add(tf.keras.layers.GRU(units=128, return_sequences=True))
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(100, activation='relu')))
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_y_signals, activation='sigmoid')))
model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam())
model.summary()

callback_checkpoint = ModelCheckpoint(filepath='hampton_weather_checkpoint.keras', monitor='val_loss', verbose=1, save_weights_only=True, save_best_only=True)
callback_early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, min_lr=1e-4, patience=0, verbose=1)
callback_tensorboard = TensorBoard(log_dir='.\\hampton_weather_logs\\', histogram_freq=0, write_graph=False)
callbacks = [callback_checkpoint, callback_tensorboard, callback_early_stopping, callback_reduce_lr]
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
            str(target_names[signal]) + '_' + str(start_idx) + '_' + str("train" if train else "test") + ".png")
        plt.close()


plot_comparison(start_idx=30000, length=1000, train=True)
plot_comparison(start_idx=40000, length=1000, train=True)

plot_comparison(start_idx=4000, length=1000, train=False)
plot_comparison(start_idx=5000, length=1000, train=False)
plot_comparison(start_idx=6000, length=1000, train=False)
