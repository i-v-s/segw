import os
import psutil
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from matplotlib import pyplot as plt
from keras import models
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from threading import Thread
from generator import train_generator
import imageio
from pympler import summary, muppy


model_name = 'boe8_2x2_16-128'
width = 512
height = 512

input_samples = np.array(list(map(lambda n: imread('input/' + n),
                                  ['valid1.png', 'valid2.png', 'valid3.png', 'valid4.png']))) / 255.0


def load_model(model_name):
    with open('models/' + model_name + '.json') as model_file:
        model = models.model_from_json(model_file.read())
    optimizer = SGD(lr=0.0015, momentum=0.95, decay=0.0005, nesterov=False)
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
    #ims = []
    #fig = plt.figure()
    #fig.canvas.mpl_connect('close_event', handle_close)
    #p = 0
    #for n in range(0, 4):
    #    a1 = fig.add_subplot(221 + p)
    #    a1.axis('off')
    #    ims.append(a1.imshow(np.zeros((height, width))))
    #    p += 1
    plt.axis('off')
    im = plt.imshow(np.zeros((height * 2, width * 2)))
    plt.ion()
    plt.show()
    return im


def calc_sample_results(model):
    global changed
    if changed:
        return
    sample_results.clear()
    for n in range(0, 4):
        outp = model.predict(input_samples[n: n + 1])
        sample_results.append(outp[0][:, :, :3])
    changed = True


def check_memory():
    free = psutil.virtual_memory().free
    if free < 100000000:
        raise MemoryError('Not enough memory')

def train(model_name):
    model = load_model(model_name)
    train_gen = train_generator(model.output_shape[3], (['wnp'], ['fill']), (height, width), 64)
    check_pointer = ModelCheckpoint(
        filepath='models/' + model_name + '.hdf5',
        verbose=1, save_best_only=True, monitor='loss')
    reduce = ReduceLROnPlateau(
        monitor='loss',
        factor=0.3,
        verbose=1,
        cooldown=10
    )
    tensor_board = TensorBoard(
        log_dir='models/' + model_name + '/',
        #write_images=True,
        #write_grads=True,
        #write_graph=True,
        #histogram_freq=1
    )
    tensor_board.validation_data = input_samples
    epoch = 0
    epochs = 20
    #while running:
    try:
        for i, o in train_gen:
            if not running:
                break
            check_memory()
            calc_sample_results(model)
            model.fit(
                i, o,
                epochs=epoch + epochs, initial_epoch=epoch,
                callbacks=[check_pointer, tensor_board],
                batch_size=6)
            #model.fit_generator(
            #    train_gen, steps_per_epoch=30,
            #    epochs=epoch + epochs, initial_epoch=epoch, callbacks=[check_pointer])
            epoch += epochs
    except MemoryError:
        return

im = prep_plot()

try:
    thread = Thread(target=train, args=(model_name,))
    thread.start()
    start = True
    with imageio.get_writer('d:/wires/' + model_name + '.gif', mode='I', fps=5) as writer:
        while thread.isAlive:
            if changed:
                i1 = np.hstack((sample_results[0], sample_results[1]))
                i2 = np.hstack((sample_results[2], sample_results[3]))
                image = np.vstack((i1, i2))[:, :, :3]

                im.set_data(image)
                if start:
                    writer.append_data(np.zeros(image.shape))
                    start = False

                writer.append_data(image)
                plt.draw()
                changed = False

                #summary.print_(summary.summarize(muppy.get_objects()))

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
