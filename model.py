import tensorflow as tf
from tensorflow.keras import layers, optimizers, callbacks

def build_model(input_shape, num_actions):
    model = tf.keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(64, (7, 7), strides=3, activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (5, 5), strides=2, activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_actions, activation='linear')
    ])
    
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mse')
    return model
