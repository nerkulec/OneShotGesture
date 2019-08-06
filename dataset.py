import numpy as np
from random import random, sample
from utils import choice, show_imgs
import os

def load_data(dataset='mnist', repeats=6):
    if dataset == 'mnist':
        from tensorflow.keras.datasets import mnist
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
                train_X_2[example] = choice(bins[y_train[example//repeats]], excluding=example//repeats)
                train_Y[example] = 1
            else:
                wrong_class = choice(range(10), excluding=y_train[example//repeats])
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
                test_X_2[example] = choice(bins[y_test[example//repeats]], excluding=example//repeats)
                test_Y[example] = 1
            else:
                wrong_class = choice(range(10), excluding=y_test[example//repeats])
                test_X_2[example] = choice(bins[wrong_class])
                test_Y[example] = 0
        return ([train_X_1, train_X_2], train_Y), ([test_X_1, test_X_2], test_Y)


    if dataset == 'omniglot':
        import cv2
        dataset_path = '~/datasets/omniglot/'
        path_bins = []
        for language in os.listdir(dataset_path):
            for character in os.listdir(dataset_path+language+'/'):
                path_bins.append([])
                for image_name in os.listdir(dataset_path+language+'/'+character+'/'):
                    path = dataset_path+language+'/'+character+'/'+image_name
                    path_bins[-1].append(path)
        test_indexes = sample(range(len(path_bins)), 200)
        train_path_bins = []
        test_path_bins = []
        for i in range(len(path_bins)):
            if i in test_indexes:
                test_path_bins.append(path_bins[i])
            else:
                train_path_bins.append(path_bins[i])
        train_bins = []
        test_bins = []
        for path_bin in train_path_bins:
            train_bins.append([])
            for img_path in path_bin:
                train_bins[-1].append(np.reshape(cv2.imread(img_path), (105, 105, 1)))
        for path_bin in test_path_bins:
            test_bins.append([])
            for img_path in path_bin:
                test_bins[-1].append(np.reshape(cv2.imread(img_path), (105, 105, 1)))
        
        total_number = sum(len(b) for b in train_bins)

        train_X_1 = np.zeros((total_number*repeats, 105, 105, 1))
        train_X_2 = np.zeros((total_number*repeats, 105, 105, 1))
        train_Y = np.zeros(total_number*repeats)
        example = 0
        for img_bin_num, img_bin in enumerate(train_bins):
            for img_num, img in enumerate(img_bin):
                for i in range(repeats):
                    train_X_1[example] = img
                    if random < 0.5:
                        train_X_2[example] = choice(img_bin, excluding=img_num)
                        train_Y[example] = 1
                    else:
                        wrong_bin = choice(train_bins, excluding=img_bin_num)
                        train_X_2[example] = choice(wrong_bin)
                        train_Y[example] = 0
                    example += 1

        total_number = sum(len(b) for b in test_bins)

        test_X_1 = np.zeros((total_number*repeats, 105, 105, 1))
        test_X_2 = np.zeros((total_number*repeats, 105, 105, 1))
        test_Y = np.zeros(total_number*repeats)
        example = 0
        for img_bin_num, img_bin in enumerate(test_bins):
            for img_num, img in enumerate(img_bin):
                for i in range(repeats):
                    test_X_1[example] = img
                    if random < 0.5:
                        test_X_2[example] = choice(img_bin, excluding=img_num)
                        test_Y[example] = 1
                    else:
                        wrong_bin = choice(test_bins, excluding=img_bin_num)
                        test_X_2[example] = choice(wrong_bin)
                        test_Y[example] = 0
                    example += 1
        
        return ([train_X_1, train_X_2], train_Y), ([test_X_1, test_X_2], test_Y)

if __name__ == '__main__':
    pass



