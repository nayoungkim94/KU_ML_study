%matplotlib inline
import tensorflow as tf
import tensorflow.contrib.eager as tfe
tf.enable_eager_execution()  # default in TF2.0
tf.set_random_seed(777)  #

print(tf.__version__)

x_data = [[0, 0],
         [0, 1],
         [1, 0],
         [1, 1]]
         
y_data = [[0],
         [1],
         [1],
         [0]]
         
dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data)).batch(len(x_data))

def preprocess_data(features, labels):
    features = tf.cast(features, tf.float32)
    labels = tf.cast(labels, tf.float32)
    return features, labels
    
W1 = tf.Variable(tf.random_normal([2, 1]), name='weight1')
b1 = tf.Variable(tf.random.normal([1]), name='bias1')

W2 = tf.Variable(tf.random_normal([2, 1]), name='weight2')
b2 = tf.Variable(tf.random.normal([1]), name='bias2')

W3 = tf.Variable(tf.random_normal([2, 1]), name='weight3')
b3 = tf.Variable(tf.random.normal([1]), name='bias3')


# XOR neural net with 3 units
def vector_neural_net(features):
    layer1 = tf.sigmoid(tf.matmul(features, W1) + b1)
    layer2 = tf.sigmoid(tf.matmul(features, W2) + b2)
    hypothesis = tf.sigmoid(tf.matmul(tf.concat([layer1, layer2], -1), W3) + b3)
    return hypothesis


# 2 units
def mat_neural_net(features):
    layer1 = tf.sigmoid(tf.matmul(features, W1) + b1)
    hypothesis = tf.sigmoid(tf.matmul(layer, W2) + b2)
    return hypothesis
    
    
# XOR neural net eager code




