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
    layer3 = tf.concat([layer1, layer2], -1)
    layer3 = tf.reshape(layer3, shape = [-1, 2])
    hypothesis = tf.sigmoid(tf.matmul(layer3, W3) + b3)
    return hypothesis


# 2 units
def mat_neural_net(features):
    layer1 = tf.sigmoid(tf.matmul(features, W1) + b1)
    hypothesis = tf.sigmoid(tf.matmul(layer, W2) + b2)
    return hypothesis
    
    
def loss_fn(hypothesis, labels):
    cost = -tf.reduce_mean(labels * tf.log(hypothesis) + (1 - labels) * tf.log(1 - hypothesis))
    return cost

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

def accuracy_fn(hypothesis, labels):
    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, labels), dtype=tf.float32))
    return accuracy

def grad(features, labels):
    with GradientTape as tape:
        loss_value = loss_fn(neural_net(features), features, labels)
    return tape.gradient(loss_value, [W1, W2, W3, b1, b2, b3])

EPOCHS = 50000

for step in range(EPOCHS):
    




