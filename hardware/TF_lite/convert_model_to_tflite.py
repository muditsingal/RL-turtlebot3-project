import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import os

curr_pwd = os.getcwd()

# Set the path to the directory containing the TensorFlow model
saved_model_dir = os.path.join('model', 'target_actor_td3')

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=20),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2, activation='tanh')
])

# model = tf.keras.Sequential([
#     tf.keras.layers.InputLayer(input_shape=20),
#     tf.keras.layers.Dense(400, activation='relu'),
#     tf.keras.layers.Dense(300, activation='relu'),
#     tf.keras.layers.Dense(2, activation='tanh')
# ])

model.compile(optimizer=Adam(learning_rate=0.0005), loss='mean_squared_error')
model.load_weights(saved_model_dir)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
tflite_model_name = 'lite_model'

open(tflite_model_name + '.tflite', 'wb').write(tflite_model)