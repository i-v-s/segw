from keras import models
from keras.layers import Input, Conv2D, Activation, concatenate, Reshape, Permute, MaxPooling2D, UpSampling2D, BatchNormalization, GlobalMaxPooling2D, Dropout, ZeroPadding2D
import json
from keras.utils import plot_model

name = 'boe'
base = 16

# Input
inputs = Input(shape=(None, None, 3))
l0 = Dropout(0.1)(inputs)

# Convolution
conv1 = Conv2D(base, 3, padding='same', activation='relu')(l0)
conv1 = Conv2D(base, 3, padding='same', activation='relu')(conv1)

conv2 = MaxPooling2D()(conv1)
conv2 = Conv2D(base * 2, 3, padding='same', activation='relu')(conv2)
conv2 = Conv2D(base * 2, 3, padding='same', activation='relu')(conv2)

conv3 = MaxPooling2D()(conv2)
conv3 = Conv2D(base * 4, 3, padding='same', activation='relu')(conv3)
conv3 = Conv2D(base * 4, 3, padding='same', activation='relu')(conv3)

conv4 = MaxPooling2D()(conv3)
conv4 = Conv2D(base * 8, 3, padding='same', activation='relu')(conv4)
conv4 = Conv2D(base * 8, 3, padding='same', activation='relu')(conv4)

conv5 = MaxPooling2D()(conv4)
conv5 = Conv2D(base * 16, 3, padding='same', activation='relu')(conv5)
conv5 = Conv2D(base * 16, 3, padding='same', activation='relu')(conv5)

conv6 = MaxPooling2D()(conv5)
conv6 = Conv2D(base * 32, 3, padding='same', activation='relu')(conv6)
conv6 = Conv2D(base * 32, 3, padding='same', activation='relu')(conv6)

# Deconvolution
dec5 = UpSampling2D()(conv6)
dec5 = concatenate([dec5, conv5], axis=3)
dec5 = Conv2D(base * 16, 3, padding='same', activation='relu')(dec5)
dec5 = Conv2D(base * 16, 3, padding='same', activation='relu')(dec5)

dec4 = UpSampling2D()(conv5)
dec4 = concatenate([dec4, conv4], axis=3)
dec4 = Conv2D(base * 8, 3, padding='same', activation='relu')(dec4)
dec4 = Conv2D(base * 8, 3, padding='same', activation='relu')(dec4)

dec3 = UpSampling2D()(dec4)
dec3 = concatenate([dec3, conv3], axis=3)
dec3 = Conv2D(base * 4, 3, padding='same', activation='relu')(dec3)
dec3 = Conv2D(base * 4, 3, padding='same', activation='relu')(dec3)

dec2 = UpSampling2D()(dec3)
dec2 = concatenate([dec2, conv2], axis=3)
dec2 = Conv2D(base * 2, 3, padding='same', activation='relu')(dec2)
dec2 = Conv2D(base * 2, 3, padding='same', activation='relu')(dec2)

dec1 = UpSampling2D()(dec2)
dec1 = concatenate([dec1, conv1], axis=3)
dec1 = Conv2D(base, 3, padding='same', activation='relu')(dec1)
dec1 = Conv2D(base, 3, padding='same', activation='relu')(dec1)

# Output
outputs = Conv2D(4, 3, padding='same', activation='softmax')(dec1)

model = models.Model(inputs=inputs, outputs=outputs)
model.summary()

with open('models/' + name + '.json', 'w') as outfile:
    outfile.write(json.dumps(json.loads(model.to_json()), indent=2))

plot_model(model, to_file='models/' + name + '.png', show_shapes=True)
