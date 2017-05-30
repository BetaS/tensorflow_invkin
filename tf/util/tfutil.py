import tensorflow as tf
import numpy as np


def init_weights(shape, init_method='xavier', xavier_params = (None, None)):
    if init_method == 'zeros':
        return tf.Variable(tf.zeros(shape, dtype=tf.float32))
    elif init_method == 'uniform':
        return tf.Variable(tf.random_normal(shape, stddev=0.01, dtype=tf.float32))
    else: #xavier
        (fan_in, fan_out) = xavier_params
        low = -4*np.sqrt(6.0/(fan_in + fan_out)) # {sigmoid:4, tanh:1}
        high = 4*np.sqrt(6.0/(fan_in + fan_out))
        return tf.Variable(tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32))

def _mlp(X, W, b):
    step = len(W)

    result = tf.matmul(X, W[0]) + b[0]

    for i in range(1, step):
        result = tf.matmul(result, W[i])# + b[i]

    return result


def _sigmoid_mlp(X, W, b):
    step = len(W)

    result = tf.nn.sigmoid(tf.matmul(X, W[0]) + b[0])

    for i in range(1, step-1):
        result = tf.nn.sigmoid(tf.matmul(result, W[i]))# +b[i]

    result = tf.matmul(result, W[-1])# + b[-1]

    #result = tf.atan(result) * 180 / 3.141592 * 100

    return result

def mlp(x, input_width, hidden_width, output_width, depth):
    weight = []
    bias = []

    weight.append(init_weights([input_width, hidden_width], 'xavier', xavier_params=(input_width, hidden_width)))
    bias.append(init_weights([1, hidden_width], 'zeros'))
    for i in range(depth):
        weight.append(init_weights([hidden_width, hidden_width], 'xavier', xavier_params=(hidden_width, hidden_width)))
        #bias.append(init_weights([1, hidden_width], 'zeros'))

    weight.append(init_weights([hidden_width, output_width], 'xavier', xavier_params=(hidden_width, output_width)))
    #bias.append(init_weights([1, output_width], 'zeros'))

    return _sigmoid_mlp(x, weight, bias), weight, bias
