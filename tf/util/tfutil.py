import tensorflow as tf
import numpy as np
import math

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
    #result = tf.matmul(X, W[0]) + b[0]

    for i in range(1, step-1):
        result = tf.nn.sigmoid(tf.matmul(result, W[i])) +b[i]

    result = tf.matmul(result, W[-1])# + b[-1]
    #result = tf.nn.sigmoid(result)
    #result = tf.tanh(result) * math.pi
    #result = result * math.pi / 180

    return result


def mlp(x, input_width, hidden_width, output_width, depth):
    weight = []
    bias = []

    weight.append(init_weights([input_width, hidden_width], 'xavier', xavier_params=(input_width, hidden_width)))
    bias.append(init_weights([1, hidden_width], 'zeros'))
    for i in range(depth):
        weight.append(init_weights([hidden_width, hidden_width], 'xavier', xavier_params=(hidden_width, hidden_width)))
        bias.append(init_weights([1, hidden_width], 'zeros'))

    weight.append(init_weights([hidden_width, output_width], 'xavier', xavier_params=(hidden_width, output_width)))
    bias.append(init_weights([1, output_width], 'zeros'))

    return _sigmoid_mlp(x, weight, bias), weight, bias


def clip_radian(r):
    return tf.where(tf.less(r, 0), 2*math.pi + r, r)


def radian_cost(r1, r2):
    r1 = tf.map_fn(clip_radian, r1)
    r2 = tf.map_fn(clip_radian, r2)

    phi = tf.abs(r1-r2)
    return tf.map_fn(lambda x: tf.where(tf.greater(x, math.pi), 2*math.pi - x, x), phi)


def clip_degree(r):
    r = tf.where(tf.greater(r, 360), r - 360, r)
    r = tf.where(tf.less(r, 0), 360.0 + r, r)
    return r


def degree_cost(r1, r2):
    r1 = tf.map_fn(clip_degree, r1)
    r2 = tf.map_fn(clip_degree, r2)

    phi = tf.abs(r1-r2)
    return tf.abs(tf.map_fn(lambda x: tf.where(tf.greater(x, 180), 360 - x, x), phi))


if __name__ == "__main__":
    input_holder = tf.placeholder(tf.float32, [None, 3], name="X")
    output_holder = tf.placeholder(tf.float32, [None, 3], name="Y")

    cost = degree_cost(input_holder, output_holder)

    sess = tf.Session()
    c = sess.run(cost, feed_dict={input_holder: [[0, 20, -90]],
                              output_holder: [[10, 0, 80]]})
    print(c)