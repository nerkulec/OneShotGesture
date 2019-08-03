from random import random
from matplotlib import pyplot as plt
import numpy as np

def choice(a):
    try:
        return a[int(random()*a.shape[0])]
    except:
        return a[int(random()*len(a))]

def show_imgs(imgs, labels=None, name='img.png'):
    imgs = [np.reshape(img, (28, 28)) for img in imgs]

    f, axarr = plt.subplots(2, len(imgs))
    for i in range(len(imgs)):
        axarr[0,i].imshow(imgs[i])
        if labels is not None:
            axarr[0,i].set_title(str(labels[i]))

    plt.savefig(name)