# ReLU
# f(x) = max(0, x)
# tf.keras.activations --> sigmoid, tanh, relu, elu, selu

# Load MNIST
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
tf.enable_eager_execution()

def load_mnist():
    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()
    # tensorflow input size: [batch_size, height, width, channel]
    train_data = np.expand_dims(train_data, axis=-1) # Add a channel [N, 28, 28] -> [N, 28, 28, 1]   
    test_data = np.expand_dims(test_data, axis=-1) #[N, 28, 28] -> [N, 28, 28, 1]
    
    train_data, test_data = normalize(train_data, test_data) # [0~255] -> [0~1]
    # One-hot encoding
    train_labels = to_categorical(train_labels, 10) # [N,] -> [N, 10]
    test_labels = to_categorical(test_labels, 10) # [N,] -> [N, 10]
    
def normalize(train_data, test_data):
    train_data = train_data.astype(np.float32) / 255.0
    test_data = test_data.astype(np.float32) / 255.0
    
    return train_data, test_data

# Create Network

def flatten():
    return tf.keras.layers.Flatten()

def dense(channel, weight_init):
    return tf.keras.layers.Dense(units=channel, dense_bias=True, kernel_initializer=weight_init)
    
def relu():
    return tf.keras.layers.Activation(tf.keras.activations.relu)


# Model

class create_model(tf.keras.Model):

