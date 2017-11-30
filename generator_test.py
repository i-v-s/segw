import numpy as np
from skimage.io import imread
from matplotlib import pyplot as plt
from generator import train_generator
from pympler import summary, muppy


size = (512, 512)

im = plt.imshow(np.zeros((size[0] * 2, size[1] * 2)))
plt.ion()
plt.show()

train_gen = train_generator(4, ['wnp'], size, 2)

ctr = 0

for i, o in train_gen:
    i1 = np.hstack((i[0], o[0][:, :, :3]))
    i2 = np.hstack((i[1], o[1][:, :, :3]))
    im.set_data(np.vstack((i1, i2)))
    plt.draw()
    plt.pause(0.3)
    ctr += 1
    if ctr > 100:
        ctr = 0
        sum1 = summary.summarize(muppy.get_objects())
        summary.print_(sum1)
