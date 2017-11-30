from keras import models
from keras.layers import Input, Conv2D, Activation, concatenate, Reshape, Permute, MaxPooling2D, UpSampling2D, BatchNormalization, GlobalMaxPooling2D, Dropout, ZeroPadding2D
import json

name = '20'

# input base
inputs = Input(shape=(None, None, 3))
i = Dropout(0.1)(inputs)
base = Conv2D(16, 3, padding='same', activation='relu')(i)

#global state
#gp = Conv2D(16, )
#gs = GlobalMaxPooling2D()(base)
gs = MaxPooling2D(pool_size=(64, 64), strides=(32, 32), padding='same')(base)
#gs = ZeroPadding2D(((0, 0), (0, 0)))(gs)
gs = Conv2D(128, 3, padding='same')(gs)
gs = Conv2D(16, 3, padding='same')(gs)
gs = UpSampling2D(size=32)(gs)

#output_shape=(None, 16)
#local state
#ls = Conv2D(32, 3, padding='same', activation='relu')(base)
m = concatenate([base, gs], axis=3)
ls = Conv2D(32, 3, padding='same', activation='relu')(m)
ls = Conv2D(64, 3, padding='same', activation='relu')(ls)
ls = Conv2D(128, 3, padding='same', activation='relu')(ls)
ls = Conv2D(64, 3, padding='same', activation='relu')(ls)
ls = Conv2D(32, 3, padding='same', activation='relu')(ls)
ls = Conv2D(16, 3, padding='same', activation='relu')(ls)
ls = Conv2D(16, 3, padding='same', activation='relu')(ls)
ls = Conv2D(16, 3, padding='same', activation='relu')(ls)

ls = Conv2D(32, 3, padding='same', activation='relu')(ls)

ls = Conv2D(32, 3, padding='same', activation='relu')(ls)
ls = Conv2D(64, 3, padding='same', activation='relu')(ls)
ls = Conv2D(32, 3, padding='same', activation='relu')(ls)
ls = Conv2D(32, 3, padding='same', activation='relu')(ls)
ls = Conv2D(16, 3, padding='same', activation='relu')(ls)
ls = Conv2D(16, 3, padding='same', activation='relu')(ls)
ls = Conv2D(16, 3, padding='same', activation='relu')(ls)
ls = Conv2D(16, 3, padding='same', activation='relu')(ls)

result = Conv2D(4, 3, padding='same', activation='softmax')(ls)

model = models.Model(input=inputs, output=result)

model.summary()

with open('models/' + name + '.json', 'w') as outfile:
    outfile.write(json.dumps(json.loads(model.to_json()), indent=2))
