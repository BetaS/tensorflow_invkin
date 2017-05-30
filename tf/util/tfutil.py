import tensorflow as tf


def xavier_init(n_inputs, n_outputs, uniform=True):
    if uniform:
        init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)
    else:
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
    return tf.truncated_normal_initializer(stddev=stddev)

def _mlp(_X, _W, _b):
    step = len(_W)

    result = tf.add(tf.matmul(_X, _W[0]), _b[0])

    for i in range(1, step):
        result = tf.add(tf.matmul(result, _W[i]), _b[i])

    return result


def mlp(x, input_width, hidden_width, output_width, depth, sigma_init=0.1):
    weight = []
    bias = []

    weight.append(tf.Variable(tf.random_normal([input_width, hidden_width], stddev = sigma_init)))
    bias.append(tf.Variable(tf.random_normal([hidden_width])))
    for i in range(depth):
        weight.append(tf.Variable(tf.random_normal([hidden_width, hidden_width], stddev = sigma_init)))
        bias.append(tf.Variable(tf.random_normal([hidden_width])))

    weight.append(tf.Variable(tf.random_normal([hidden_width, output_width], stddev = sigma_init)))
    bias.append(tf.Variable(tf.random_normal([output_width])))

    return _mlp(x, weight, bias)
