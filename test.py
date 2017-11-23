import numpy as np
from skimage.io import imread
from matplotlib import pyplot as plt
from keras import models

modelName = 'ten'

with open('models/' + modelName + '.json') as model_file:
    model = models.model_from_json(model_file.read())

model.load_weights('models/' + modelName + '.hdf5')

input = np.array([imread('input/lep4.jpg')]) / 255.0

input[0] = np.flip(input[0], 1)

fig = plt.figure()

a1 = fig.add_subplot(221)
a1.axis('off')
a1.imshow(input[0])

output = model.predict(input)

a2 = fig.add_subplot(222)
a2.axis('off')
a2.imshow(output[0][:,:,:3])

plt.show()
