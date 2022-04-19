# %tensorflow_version 2.x
import os
from pathlib import Path
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.client import device_lib

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = Path(ROOT, 'output')


def get_device_info():
    device_list = device_lib.list_local_devices()
    print(device_list)


def check_graphviz():
    input = tf.keras.Input(shape=(100,), dtype='int32', name='input')
    x = tf.keras.layers.Embedding(
        output_dim=512, input_dim=10000, input_length=100)(input)
    x = tf.keras.layers.LSTM(32)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    output = tf.keras.layers.Dense(
        1, activation='sigmoid', name='output')(x)
    model = tf.keras.Model(inputs=[input], outputs=[output])
    file_path = Path(OUTPUT_PATH, 'check_graphviz_lib.png')
    keras.utils.plot_model(model, to_file=file_path, show_shapes=True)
    file_path.unlink()  # delete check_graphviz_lib.png
    print('Great! Graphviz already exists.')


if __name__ == '__main__':
    # Get CPU and GPU information.
    get_device_info()
    # Check Graphviz software.
    check_graphviz()
