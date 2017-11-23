import numpy as np
from skimage.io import imread
from skimage.transform import resize
from matplotlib import pyplot as plt
from keras import models
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from threading import Thread
from generator import train_generator

modelName = 'three'
width = 512 #640
height = 512 #480

input_samples = np.array(list(map(lambda n: imread('input/' + n), ['data/lep1.jpg', 'data/lep2_s.png', 'data/lep4_s.png', 'data/wi3.png']))) / 255.0


def load_model(model_name):
    with open('models/' + modelName + '.json') as model_file:
        model = models.model_from_json(model_file.read())
    optimizer = SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=False)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
    print('Compiled: OK')
    try:
        model.load_weights('models/' + model_name + '.hdf5')
        print('Weights loaded')
    except OSError:
        pass
    return model


running = True
changed = False
sample_results = []


def handle_close():
    global running
    print("Terminating...")
    running = False
    thread.join()


def prep_plot():
    ims = []
    fig = plt.figure()
    fig.canvas.mpl_connect('close_event', handle_close)
    p = 0
    for n in range(0, 4):
        a1 = fig.add_subplot(221 + p)
        a1.axis('off')
        ims.append(a1.imshow(np.zeros((height, width))))
        p += 1

    plt.ion()
    plt.show()
    return ims


def calc_sample_results(model):
    global changed
    if changed:
        return
    sample_results.clear()
    for n in range(0, 4):
        outp = model.predict(input_samples[n: n + 1])
        sample_results.append(outp[0][:, :, :3])
    changed = True


def train(model_name):
    model = load_model(model_name)
    train_gen = train_generator(model.output_shape[3], ["wnp"], (height, width), 64)
    check_pointer = ModelCheckpoint(
        filepath='models/' + modelName + '.hdf5',
        verbose=1, save_best_only=True, monitor='loss')
    #check_pointer.
    epoch = 0
    epochs = 5
    #while running:
    for i, o in train_gen:
        if not running:
            break
        calc_sample_results(model)
        model.fit(i, o, epochs=epoch + epochs, initial_epoch=epoch, callbacks=[check_pointer], batch_size=4)
        #model.fit_generator(
        #    train_gen, steps_per_epoch=30,
        #    epochs=epoch + epochs, initial_epoch=epoch, callbacks=[check_pointer])
        epoch += epochs


ims = prep_plot()

try:
    thread = Thread(target=train, args=(modelName,))
    thread.start()
    while True:
        if changed:
            for n in range(0, 4):
                ims[n].set_data(sample_results[n])
            plt.draw()
            changed = False
        plt.pause(1)
except KeyboardInterrupt:
    print("Terminating...")
    running = False
    thread.join()


'''
im = plt.imshow(np.zeros((height, width)))
plt.ion()
plt.show()

for i, o in train_gen:
    im.set_data(np.hstack((i[0], o[0][:, :, :3])))
    plt.draw()
    plt.pause(0.01)
'''

'''
input = []
output = []

def load_pair(input_name, output_name):
    input_img = imread('input/' + input_name)
    output_img = imread('output/' + output_name)

    for a in output_img:
        for b in a:
            b[3] = 255 - b[3]

    input.append(input_img)
    input.append(np.flip(input_img, 1))
    output.append(output_img)
    output.append(np.flip(output_img, 1))

load_pair('lep1.jpg', 'lep1.png')
load_pair('lep2_s.png', 'lep2_s.png')
load_pair('lep4_s.png', 'lep4_s.png')
load_pair('wi3.png', 'wi3.png')

input = np.array(input) / 256.0
output = np.array(output) / 256.0


ims = []

fig = plt.figure()
p = 0
for n in range(0, 4):
    a1 = fig.add_subplot(221 + p)
    a1.axis('off')
    ims.append(a1.imshow(np.zeros((453, 600))))
    #ims.append()
    p += 1

plt.ion()
plt.show()


for t in range(0, 10):

    for n in range(0, 4):
        outp = model.predict(input[n * 2 : n * 2 + 1])
        ims[n].set_data(outp[0][:, :, :3])
        #ims[n].draw()

    #plt.draw()
    plt.pause(0.0001)
    fig.canvas.draw()

    history = model.fit(input, output, epochs=10, batch_size=4)
    model.save_weights('models/' + modelName + '.hdf5')

'''
