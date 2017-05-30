import tensorflow as tf
import numpy as np
import math, random
import matplotlib.pyplot as plt

# %matplotlib inline

np.random.seed(1000) # for repro
function_to_learn = lambda x: np.matrix(np.arctan2(x.T[0], x.T[1])).T

NUM_HIDDEN_NODES = 200
NUM_EXAMPLES = 1000
TRAIN_SPLIT = .8
MINI_BATCH_SIZE = 100
NUM_EPOCHS = 1000
"""
all_x = np.float32(
    np.random.uniform(-2*math.pi, 2*math.pi, (2, NUM_EXAMPLES))).T
np.random.shuffle(all_x)
train_size = int(NUM_EXAMPLES*TRAIN_SPLIT)

trainx = all_x[:train_size]
validx = all_x[train_size:]

trainy = function_to_learn(trainx)
validy = function_to_learn(validx)
"""

def load_file(file_path):
    with open(file_path) as f:
        lines = f.readlines()

        if len(lines) > 0:
            print(len(lines), "test case loaded")
            return lines

    return None

loadpath = "../testset_arcsin.txt"
datas = load_file(loadpath)

inputs = []
outputs = []

# Parse Data
for data in datas:
    data = data.split()
    inputs.append(list(map(lambda x: float(x), data[1:])))
    outputs.append(list(map(lambda x: float(x), data[:1])))

train_size = int(len(inputs)*TRAIN_SPLIT)


trainx = np.matrix(inputs[:train_size])
validx = np.matrix(inputs[train_size:])

trainy = np.matrix(outputs[:train_size])
validy = np.matrix(outputs[train_size:])

"""
plt.figure(1)
plt.scatter(trainx, trainy, c='green', label='train')
plt.scatter(validx, validy, c='red', label='validation')
plt.legend()
plt.show()
"""

X = tf.placeholder(tf.float32, [None, 2], name="X")
Y = tf.placeholder(tf.float32, [None, 1], name="Y")

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

def model(X, num_hidden=10):
    w_h = init_weights([2, num_hidden], 'xavier', xavier_params=(2, num_hidden))
    b_h = init_weights([1, num_hidden], 'zeros')
    h = tf.nn.sigmoid(tf.matmul(X, w_h) + b_h)

    w_o = init_weights([num_hidden, 1], 'xavier', xavier_params=(num_hidden, 1))
    b_o = init_weights([1, 1], 'zeros')
    return tf.matmul(h, w_o) + b_o

yhat = model(X, NUM_HIDDEN_NODES)

train_op = tf.train.AdamOptimizer().minimize(tf.nn.l2_loss(yhat - Y))

sess = tf.Session()
sess.run(tf.initialize_all_variables())

errors = []
for i in range(NUM_EPOCHS):
    for start, end in zip(range(0, len(trainx), MINI_BATCH_SIZE), range(MINI_BATCH_SIZE, len(trainx), MINI_BATCH_SIZE)):
        sess.run(train_op, feed_dict={X: trainx[start:end], Y: trainy[start:end]})
    mse = sess.run(tf.nn.l2_loss(yhat - validy),  feed_dict={X: validx})
    errors.append(mse)
    if i%100 == 0: print("epoch %d, validation MSE %g" % (i, mse))

plt.plot(errors)
plt.xlabel('#epochs')
plt.ylabel('MSE')
plt.show()

y_pred = sess.run(yhat, feed_dict={X: validx})

plt.figure(1)
for i in range(len(validx)):
    print(validx[i], y_pred[i], validy[i])
