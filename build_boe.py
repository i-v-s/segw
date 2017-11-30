from keras import models
from keras.layers import Input, Conv2D, Activation, concatenate, Reshape, Permute, MaxPooling2D, UpSampling2D, BatchNormalization, GlobalMaxPooling2D, Dropout, ZeroPadding2D
import json
from keras.utils import plot_model

name = 'boe'
base = 8

# Input
inputs = Input(shape=(None, None, 3))
l0 = Dropout(0.1)(inputs)

# Convolution
l1 = Conv2D(base, 3, padding='same', activation='relu')(l0)
l1 = Conv2D(base, 3, padding='same', activation='relu')(l1)

l2 = MaxPooling2D()(l1)
l2 = Conv2D(base * 2, 3, padding='same', activation='relu')(l2)
l2 = Conv2D(base * 2, 3, padding='same', activation='relu')(l2)

l3 = MaxPooling2D()(l2)
l3 = Conv2D(base * 4, 3, padding='same', activation='relu')(l3)
l3 = Conv2D(base * 4, 3, padding='same', activation='relu')(l3)

l4 = MaxPooling2D()(l3)
l4 = Conv2D(base * 8, 3, padding='same', activation='relu')(l4)
l4 = Conv2D(base * 8, 3, padding='same', activation='relu')(l4)

l5 = MaxPooling2D()(l4)
l5 = Conv2D(base * 16, 3, padding='same', activation='relu')(l5)
l5 = Conv2D(base * 16, 3, padding='same', activation='relu')(l5)

# Deconvolution
l6 = UpSampling2D()(l5)
l6 = concatenate([l6, l4], axis=3)
l6 = Conv2D(base * 8, 3, padding='same', activation='relu')(l6)
l6 = Conv2D(base * 8, 3, padding='same', activation='relu')(l6)

l7 = UpSampling2D()(l6)
l7 = concatenate([l7, l3], axis=3)
l7 = Conv2D(base * 4, 3, padding='same', activation='relu')(l7)
l7 = Conv2D(base * 4, 3, padding='same', activation='relu')(l7)

l8 = UpSampling2D()(l7)
l8 = concatenate([l8, l2], axis=3)
l8 = Conv2D(base * 2, 3, padding='same', activation='relu')(l8)
l8 = Conv2D(base * 2, 3, padding='same', activation='relu')(l8)

l9 = UpSampling2D()(l8)
l9 = concatenate([l9, l1], axis=3)
l9 = Conv2D(base, 3, padding='same', activation='relu')(l9)
l9 = Conv2D(base, 3, padding='same', activation='relu')(l9)

# Output
outputs = Conv2D(4, 3, padding='same', activation='softmax')(l9)

model = models.Model(inputs=inputs, outputs=outputs)
model.summary()

with open('models/' + name + '.json', 'w') as outfile:
    outfile.write(json.dumps(json.loads(model.to_json()), indent=2))

plot_model(model, to_file='models/' + name + '.png', show_shapes=True)
