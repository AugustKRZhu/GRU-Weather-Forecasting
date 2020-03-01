import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import sys
import urllib.request
import datetime
import tarfile
import zipfile
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def print_download_progress(count, block_size, total_size):
    pct_complete = float(count * block_size) / total_size
    pct_complete = min(1.0, pct_complete)
    msg = "\r- Download progress: {0:.1%}".format(pct_complete)
    sys.stdout.write(msg)
    sys.stdout.flush()


def maybe_download_and_extract(url, download_dir):
    filename = url.split('/')[-1]
    file_path = os.path.join(download_dir, filename)
    if not os.path.exists(file_path):
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
        file_path, _ = urllib.request.urlretrieve(url=url, filename=file_path, reporthook=print_download_progress)
        print()
        print("Download finished. Extracting files.")
        if file_path.endswith(".zip"):
            zipfile.ZipFile(file=file_path, mode="r").extractall(download_dir)
        elif file_path.endswith((".tar.gz", ".tgz")):
            tarfile.open(name=file_path, mode="r:gz").extractall(download_dir)
        print("Done.")
    else:
        print("Data has apparently already been downloaded and unpacked.")


def resample(df):
    df_res = df.dropna(axis=0, how='all')
    df_res = df_res.dropna(axis=1, how='all')
    df_res = df_res.resample('1T')
    df_res = df_res.interpolate(method='time')
    df_res = df_res.resample('60T')
    df_res = df_res.interpolate()
    df_res = df_res.dropna(how='all')
    return df_res


def load_resampled_data():
    path = os.path.join("data/weather-denmark/", "weather-denmark-resampled.pkl")
    if os.path.exists(path):
        df = pd.read_pickle(path)
    else:
        df_org = pd.read_pickle(os.path.join("data/weather-denmark/", "weather-denmark.pkl"))
        df_cities = [df_org.xs(city) for city in cities]
        df_resampled = [resample(df_city) for df_city in df_cities]
        df = pd.concat(df_resampled, keys=cities, axis=1, join='inner')
        df.to_pickle(path)
    return df


pd.set_option('display.max_columns', None)

cities = ['Aalborg', 'Aarhus', 'Esbjerg', 'Odense', 'Roskilde']
maybe_download_and_extract("https://github.com/Hvass-Labs/weather-denmark/raw/master/weather-denmark.tar.gz", "data/weather-denmark/")

df = load_resampled_data()

df.drop(('Esbjerg', 'Pressure'), axis=1, inplace=True)
df.drop(('Roskilde', 'Pressure'), axis=1, inplace=True)

df['Various', 'Day'] = df.index.dayofyear
df['Various', 'Hour'] = df.index.hour

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

x_train = x_data[0:num_train]
x_test = x_data[num_train:]

y_train = y_data[0:num_train]
y_test = y_data[num_train:]

num_x_signals = x_data.shape[1]
num_y_signals = y_data.shape[1]

x_scaler = MinMaxScaler()
x_train_scaled = x_scaler.fit_transform(x_train)
x_test_scaled = x_scaler.transform(x_test)

y_scaler = MinMaxScaler()
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)


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


warmup_steps = 50


def loss_mse_warmup(y_true, y_pred):
    y_true_slice = y_true[:, warmup_steps:, :]
    y_pred_slice = y_pred[:, warmup_steps:, :]
    loss = tf.losses.MSE(y_true_slice, y_pred_slice)
    loss_mean = tf.reduce_mean(loss)
    return loss_mean


# optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)
optimizer = tf.keras.optimizers.RMSprop()
model.compile(loss=loss_mse_warmup, optimizer=optimizer)
model.summary()

path_checkpoint = 'weather_checkpoint.keras'
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint, monitor='val_loss', verbose=1, save_weights_only=True, save_best_only=True)
callback_early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=1)
callback_tensorboard = TensorBoard(log_dir='.\\weather_logs\\', histogram_freq=0, write_graph=False)
callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, min_lr=1e-4, patience=3, verbose=1)
callbacks = [callback_early_stopping, callback_checkpoint, callback_tensorboard, callback_reduce_lr]
model.fit(generator, epochs=60, steps_per_epoch=100, validation_data=validation_data, verbose=0, callbacks=callbacks)
result = model.evaluate(x=np.expand_dims(x_test_scaled, axis=0), y=np.expand_dims(y_test_scaled, axis=0))
print("loss (test-set):", result)


def plot_comparison(start_idx, length=100, train=True):
    if train:
        x = x_train_scaled
        y_true = y_train
    else:
        x = x_test_scaled
        y_true = y_test

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
        plt.axvspan(0, warmup_steps, facecolor='black', alpha=0.15)
        plt.ylabel(target_names[signal])
        plt.legend()
        plt.savefig(str(target_names[signal]) + "{date:%Y-%m-%d_%H-%M-%S.%f}".format(date=datetime.datetime.now()) + str("train" if train else "test") + ".png")
        plt.close()


plot_comparison(start_idx=100000, length=1000, train=True)
plot_comparison(start_idx=200, length=1000, train=False)
