from keras import models
from keras.layers import Input, Conv2D, Activation, concatenate, Reshape, Permute, MaxPooling2D, UpSampling2D, BatchNormalization, GlobalMaxPooling2D
import json

name = 'ten'

# input base
inputs = Input(shape=(None, None, 3))
base = Conv2D(16, 3, padding='same', activation='relu')(inputs)

#global state
#gp = Conv2D(16, )
#gs = GlobalMaxPooling2D()(base)
gs = MaxPooling2D(pool_size=(80, 80), strides=(40, 40))(base)
gs = Conv2D(8, 5, padding='same')(gs)
gs = UpSampling2D(size=40)(gs)

#output_shape=(None, 16)
#local state
ls = Conv2D(16, 3, padding='same', activation='relu')(base)
m = concatenate([base, gs], axis=3)
ls = Conv2D(16, 3, padding='same', activation='relu')(base)
ls = Conv2D(16, 3, padding='same', activation='relu')(ls)
ls = Conv2D(16, 3, padding='same', activation='relu')(ls)
ls = Conv2D(16, 3, padding='same', activation='relu')(ls)
ls = Conv2D(16, 3, padding='same', activation='relu')(ls)
ls = Conv2D(16, 3, padding='same', activation='relu')(ls)
ls = Conv2D(16, 3, padding='same', activation='relu')(ls)
ls = Conv2D(16, 3, padding='same', activation='relu')(ls)
result = Conv2D(4, 3, padding='same', activation='softmax')(ls)

model = models.Model(input=inputs, output=result)

model.summary()

with open('models/' + name + '.json', 'w') as outfile:
    outfile.write(json.dumps(json.loads(model.to_json()), indent=2))
