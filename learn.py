from argparse import ArgumentParser
import json
import sys
from keras import models
from keras.optimizers import SGD, Adagrad
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from generator import train_generator_queue, shutdown_generator, load_samples
from time import sleep, time
from tensorflow.python.framework.errors import OpError

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
    return model, json.load(open('models/' + model_name + '_state.json'))


def check_memory():
    free = psutil.virtual_memory().free
    if free < 100000000:
        raise MemoryError('Not enough memory')


parser = ArgumentParser(description='Learning neural network')
parser.add_argument('name', type=str, default='boe8_2x2_16-128', help='Name of network')
parser.add_argument('-s', type=int, default=512, help='Height and width of the images')
args = parser.parse_args()
model_name = args.name
size = (args.s, args.s)
model, state = load_model(model_name)
epoch = state.get('epoch', 1)
sample_x = load_samples('validation/input')
sample_y = load_samples('validation/output', inv_channel=3)
# train_gen = train_generator(model.output_shape[3], (['wnp'], ['fill']), size, 8, every_flip=True, workers=2)
train_queue = train_generator_queue(model.output_shape[3], (['wnp'], ['fill']), size, 10, every_flip=True)

best_check = ModelCheckpoint(
    filepath='models/' + model_name + '_best.hdf5',
    verbose=1, save_best_only=True, monitor='loss')

best_loss = state.get('best_loss', float('inf'))

train_data = None


def train_on_queue(queue, epochs=1, updates=50):
    global epoch, train_data, best_loss
    e, u, g = epochs, 0, 0
    loss, acc = 0, 0
    t = None
    while True:
        if u >= updates:
            loss /= u
            u = 0
            validation_loss, validation_acc = model.test_on_batch(sample_x, sample_y)
            print('Epoch %d completed. vl: %.3f, va: %.3f, tl: %.3f, gets: %d, time: %.1fs'
                  % (epoch, validation_loss, validation_acc, loss, g, time() - t))
            t = None
            if validation_loss < best_loss:
                print('Best loss!')
                best_loss = validation_loss
                model.save_weights('models/' + model_name + '_best.hdf5')
            g = 0
            epoch += 1
            e -= 1
            if e <= 0:
                return
        if not queue.empty():
            train_data = queue.get()
            g += 1
        if train_data is None:
            print('Waiting for data...')
            sleep(5)
        else:
            if t is None:
                t = time()
            (l, a) = model.train_on_batch(train_data[0], train_data[1])
            loss += l
            acc += a
            u += 1


def train(count):
    global epoch
    h = model.fit_generator(
        train_gen,
        steps_per_epoch=10,
        verbose=2,
        initial_epoch=epoch,
        validation_data=(sample_x, sample_y),
        epochs=epoch + count,
        callbacks=[best_check],
        max_queue_size=10
    )
    epoch += count
    return h


try:
    while True:
        train_on_queue(train_queue, epochs=5)
        #h = train(1)
        state.update({'epoch': epoch, 'best_loss': best_loss})
        model.save_weights('models/' + model_name + '.hdf5')
        with open('models/' + model_name + '_state.json', 'w') as outfile:
            outfile.write(json.dumps(state, indent=2))
except Exception as e: # KeyboardInterrupt or MemoryError or OpError:
    print('Exception: ', e.message)
    print('Terminating...')
    shutdown_generator()
    sys.exit(1)
