import numpy as np
import math, random
import matplotlib.pyplot as plt
from tf.numpyNN import NeuralNetwork

# %matplotlib inline

np.random.seed(1000) # for repro
function_to_learn = lambda x: np.matrix(np.arctan2(x.T[0], x.T[1])).T

NUM_HIDDEN_NODES = 200
NUM_EXAMPLES = 1000
TRAIN_SPLIT = .8
MINI_BATCH_SIZE = 100
NUM_EPOCHS = 1000

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

nn = NeuralNetwork(2, 100, 1)
print(trainx.shape[0])
nn.fit(trainx, trainy, epochs = 50, learning_rate = .1, learning_rate_decay = .01, verbose = 1)

y_predicted = nn.predict(validx)

y_predicted = np.argmax(y_predicted, axis=1).astype(int)
y_test = np.argmax(validy, axis=1).astype(int)

print(y_predicted)
print(y_test)