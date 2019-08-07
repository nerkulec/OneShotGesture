import numpy as np
from random import random, sample, shuffle
from utils import choice, show_imgs
import os
from tqdm import tqdm

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

        print('mnist loaded')
        return ([train_X_1, train_X_2], train_Y), ([test_X_1, test_X_2], test_Y)


def get_generators(dataset='omniglot', batch_size=64):
        import cv2
        dataset_path = '/home/bartek/datasets/omniglot/'
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
        
        train_path_bins = np.array(train_path_bins)
        test_path_bins = np.array(test_path_bins)

        def preprocess_img(img):
            img = np.mean(np.reshape(img, (105, 105, 3)), axis=2, keepdims=True)
            return 1-img/255

        def load_img(img_path):
            return preprocess_img(cv2.imread(img_path))

        def generator(batch_size = batch_size, train=True):
            path_bins = train_path_bins if train else test_path_bins
            while True:
                batch_classes = path_bins[np.random.choice(len(path_bins), size = batch_size)]
                correct_class_paths = [np.random.choice(batch_class, size=2) for batch_class in batch_classes[::2]]
                wrong_class_paths = [np.random.choice(batch_class) for batch_class in batch_classes[1::2]]
                batch_x = []
                batch_y = []

                for (img_1_path, img_2_path), img_3_path in zip(correct_class_paths, wrong_class_paths):
                    img_1, img_2, img_3 = (load_img(img_path) for img_path in [img_1_path, img_2_path, img_3_path])
                    batch_x.extend([[img_1.view(), img_2.view()], [img_1.view(), img_3.view()]])
                    batch_y.extend([1, 0])

                batch_x_1, batch_x_2 = np.transpose(np.array(batch_x), [1,0,2,3,4])
                batch_y = np.array(batch_y)
                
                yield [batch_x_1, batch_x_2], batch_y

        return generator(train=True), generator(train=False)

if __name__ == '__main__':
    pass



