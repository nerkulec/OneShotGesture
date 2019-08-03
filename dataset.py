from tensorflow.keras.datasets import mnist
import numpy as np
from random import random
from utils import choice, show_imgs

def load_data(repeats=6):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

    data = zip(x_train, y_train)
    bins = [[] for _ in range(10)]
    for point in data:
        bins[point[1]].append(point[0])

    train_X_1 = np.zeros((x_train.shape[0]*repeats, x_train.shape[1], x_train.shape[2], x_train.shape[3]))
    train_X_2 = np.zeros((x_train.shape[0]*repeats, x_train.shape[1], x_train.shape[2], x_train.shape[3]))
    train_Y = np.zeros(x_train.shape[0]*repeats)
    for example in range(train_X_1.shape[0]):
        train_X_1[example] = x_train[example//repeats]
        if random() < 0.5:
            train_X_2[example] = choice(bins[y_train[example//repeats]])
            train_Y[example] = 1
        else:
            wrong_class = int(random()*9)
            if wrong_class >= y_train[example//repeats]:
                wrong_class += 1
            train_X_2[example] = choice(bins[wrong_class])
            train_Y[example] = 0

    data = zip(x_test, y_test)
    bins = [[] for _ in range(10)]
    for point in data:
        bins[point[1]].append(point[0])
    test_X_1 = np.zeros((x_test.shape[0]*repeats, x_test.shape[1], x_test.shape[2], x_test.shape[3]))
    test_X_2 = np.zeros((x_test.shape[0]*repeats, x_test.shape[1], x_test.shape[2], x_test.shape[3]))
    test_Y = np.zeros(x_test.shape[0]*repeats)
    for example in range(test_X_1.shape[0]):
        test_X_1[example] = x_test[example//repeats]
        if random() < 0.5 :
            test_X_2[example] = choice(bins[y_test[example//repeats]])
            test_Y[example] = 1
        else:
            wrong_class = int(random()*9)
            if wrong_class >= y_test[example//repeats]:
                wrong_class += 1
            test_X_2[example] = choice(bins[wrong_class])
            test_Y[example] = 0
    return ([train_X_1, train_X_2], train_Y), ([test_X_1, test_X_2], test_Y)

if __name__ == '__main__':
    pass



