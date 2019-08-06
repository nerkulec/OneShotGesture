from random import random
from matplotlib import pyplot as plt
import numpy as np

def choice(a, excluding=None):
    l = len(a)
    if excluding is not None:
        l -= 1
    index = int(random()*l)
    if excluding is not None and l >= excluding:
        index += 1
    return a[index]

def show_imgs(imgs, labels=None, name='img.png'):
    imgs = [np.reshape(img, (28, 28)) for img in imgs]

    f, axarr = plt.subplots(2, len(imgs))
    for i in range(len(imgs)):
        axarr[0,i].imshow(imgs[i])
        if labels is not None:
            axarr[0,i].set_title(str(labels[i]))

    plt.savefig(name)