from keras import models
from keras.layers import Input, Conv2D, Activation, concatenate, Reshape, Permute, MaxPooling2D, UpSampling2D, BatchNormalization, GlobalMaxPooling2D, Dropout, ZeroPadding2D
import json
from keras.utils import plot_model


def build_model(base=16, conv_depth=2, deconv_depth=2, depth=6, max_filters=128):
    inputs = Input(shape=(None, None, 3))
    input_layer = Dropout(0.1)(inputs)
    b = base
    convolutions = []
    for n in range(depth):
        convolution = MaxPooling2D(name='max2d' + '_' + str(n))(convolution) if n > 0 else input_layer
        for c in range(conv_depth):
            convolution = Conv2D(min(b, max_filters), 3, padding='same', activation='relu', name='conv2d' + '_' + str(n) + '_' + str(c))(convolution)
        convolutions.append(convolution)
        b *= 2
    b /= 2
    deconvolution = convolution
    for n in range(depth - 2, -1, -1):
        b = int(b / 2)
        deconvolution = UpSampling2D(name='up2d' + '_' + str(n))(deconvolution)
        deconvolution = concatenate([deconvolution, convolutions[n]], axis=3)
        for c in range(deconv_depth):
            deconvolution = Conv2D(min(b, max_filters), 3, padding='same', activation='relu')(deconvolution)
    outputs = Conv2D(4, 3, padding='same', activation='softmax')(deconvolution)
    return models.Model(inputs=inputs, outputs=outputs)


def create_model(base=16, conv_depth=2, deconv_depth=2, depth=6, max_filters=128):
    model_name = 'boe' + str(depth) + '_' + str(conv_depth) + 'x' + str(deconv_depth)\
                 + '_' + str(base) + '-' + str(max_filters)
    print('Building model ' + model_name)
    model = build_model(base=base, conv_depth=conv_depth, deconv_depth=deconv_depth, depth=depth)
    with open('models/' + model_name + '.json', 'w') as outfile:
        outfile.write(json.dumps(json.loads(model.to_json()), indent=2))
    with open('models/' + model_name + '_state.json', 'w') as outfile:
        outfile.write(json.dumps({'epoch': 0, 'size': 2 ** (depth - 1)}, indent=2))
    with open('models/' + model_name + '.txt', 'w') as outfile:
        model.summary(print_fn=lambda x: outfile.write(x + '\n'))
    plot_model(model, to_file='models/' + model_name + '.png', show_shapes=True)
    return model


result = create_model(base=16, depth=8)
result.summary()
