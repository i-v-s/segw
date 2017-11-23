from keras import models
from keras.layers import Conv2D, Activation, Reshape, Permute, MaxPooling2D, UpSampling2D, BatchNormalization
import json

kernel = 5
name = 'seven'

layers = [
    Conv2D(64, kernel, padding='same', input_shape=(None, None, 3), activation='relu'),

    Conv2D(32, kernel, padding='same', activation='relu'),
    #BatchNormalization(),
    #Activation('relu'),

    Conv2D(16, kernel, padding='same', activation='relu'),

    Conv2D(8, kernel, padding='same'),
    BatchNormalization(),
    Activation('relu'),

    Conv2D(8, kernel, padding='same', activation='relu'),

    Conv2D(8, kernel, padding='same', activation='relu'),

    Conv2D(4, kernel, padding='same', activation='softmax'),
    #Activation('softmax')
    #MaxPooling2D()
]

model = models.Sequential()

for l in layers:
    model.add(l)

model.summary()

with open('models/' + name + '.json', 'w') as outfile:
    outfile.write(json.dumps(json.loads(model.to_json()), indent=2))
