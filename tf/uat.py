# -*- coding: utf-8 -*-

import codecs
import numpy as np
import matplotlib.pyplot as plt

from GenForKin import generate_tc

from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

def read_data():
    test_input = []
    test_output = []
    for i in range(1000):
        case = generate_tc()
        test_input.append(case[7:])
        test_output.append(case[0:1])

    return test_input, test_output

def train_and_evaluate():
    train_x, train_y = read_data()
    test_x, test_y = read_data()
    samples, order_x = train_x.shape
    samples, order_y = train_y.shape
    model = Sequential()
    model.add(Dense(32, activation='sigmoid', init='normal', input_dim=order_x))  # 2-dimensional
    model.add(Dense(32, activation='sigmoid', init='normal'))
    model.add(Dense(32, activation='sigmoid', init='normal'))
    model.add(Dense(order_y, activation='sigmoid', init='normal'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_x, train_y, nb_epoch=100, batch_size=100, verbose=2)
    pred_y = model.predict_classes(test_x, verbose=0)
    diff = test_y.flatten() - pred_y.flatten()
    accuracy = len(diff[diff == 0]) / len(diff)
    print("accuracy: %5.02f%%" % (accuracy * 100))  # 100.00%
    draw_decision_boundary(model)


def draw_decision_boundary(model):
    x1s = np.linspace(-3, 3, num=50)
    x2s = np.linspace(-3, 3, num=50)
    data = []
    for x1 in x1s:
        for x2 in x2s:
            data.append([x1, x2, model.predict_classes(np.array([x1, x2]).reshape(-1, 2), verbose=0)[0]])
    data = np.array(data)
    outers = data[data[:, -1] == 0][:, :-1]
    inners = data[data[:, -1] == 1][:, :-1]
    plt.clf()
    subplot = plt.subplot()
    subplot.scatter(outers[:, 0], outers[:, 1], color='g', alpha=0.75, label='false')
    subplot.scatter(inners[:, 0], inners[:, 1], color='r', alpha=0.75, label='true')
    subplot.set_xlim([-3, 3])
    subplot.set_ylim([-3, 3])
    box = subplot.get_position()
    subplot.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    subplot.legend(loc='upper left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.savefig('decision_boundary.png')


if __name__ == '__main__':
    create_data()
    train_and_evaluate()