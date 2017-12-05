import numpy as np
import sys
from math import sin, cos, pi
from os import listdir
from skimage.io import imread
import random
from queue import Queue
from threading import Thread
from scipy.ndimage.interpolation import rotate
from pympler import summary, muppy


def train_generator(out_channels, classes, size, batch=None, dump_mem=False):

    def load_classes(folder, class_list):
        fns = []
        for c in class_list:
            fns += [c + '/' + f for f in listdir(folder + c) if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg')]

        def load(fn):
            img = imread(folder + fn) / 255.0
            assert img.shape[0] > size[0] and img.shape[1] > size[1]
            return img
        return list(map(load, fns))

    def prep_out(o):
        result = o[:, :, :out_channels - 1]
        alpha = 1.0 - np.sum(result, 2, keepdims=True)
        alpha[alpha < 0.0] = 0.0
        return np.concatenate((result, alpha), axis=2)

    def test_angle(a, i, o): # angle, (height, width) of input, (height, width) of output
        a *= -pi / 180
        cs, sn = cos(a), sin(a)
        acs, asn = abs(cs), abs(sn)
        rh = i[0] - o[0] * acs - o[1] * asn  # moving range
        rw = i[1] - o[1] * acs - o[0] * asn
        return rh > 0 and rw > 0

    def gen_pos(a, i, o, count): # angle, (height, width) of input, (height, width) of output
        a *= -pi / 180
        cs, sn = cos(a), sin(a)
        acs, asn = abs(cs), abs(sn)
        rh = i[0] - o[0] * acs - o[1] * asn  # moving range
        rw = i[1] - o[1] * acs - o[0] * asn
        ih = i[0] * acs + i[1] * asn  # rotated input size
        iw = i[1] * acs + i[0] * asn
        for c in range(count):
            dh = rh * (random.random() - 0.5)  # select offset
            dw = rw * (random.random() - 0.5)
            rdh = cs * dh - sn * dw  # rotated offset
            rdw = cs * dw + sn * dh
            yield int((ih - o[0]) / 2 - rdh), int((iw - o[1]) / 2 + rdw)

    def rotate_f(i, a):
        img = rotate(i, a, reshape=True)
        img[img > 1.0] = 1.0
        img[img < 0.0] = 0.0
        return img

    def generate_loop(q, dump_mem):
        try:
            class_tuple = classes if isinstance(classes, tuple) else (classes, classes)
            inputs = load_classes('input/', class_tuple[0])
            outputs = list(map(prep_out, load_classes('output/', class_tuple[1])))
            random.seed(1)
            area = size[0] * size[1]
            while True:
                results = []
                metrics = []
                msum = np.ones(out_channels)
                for a in range(3):
                    n = random.randint(0, len(inputs) - 1)
                    i = inputs[n]
                    o = outputs[n]
                    i_shape = i.shape
                    a = -50 + 100.0 * random.random()
                    while not test_angle(a, i_shape, size):
                        a *= 0.5
                    i = rotate_f(i, a)
                    o = rotate_f(o, a)
                    count = int(i_shape[0] * i_shape[1] / area * 3)
                    for p in gen_pos(a, i_shape, size, count):
                        x, y = p[1], p[0]
                        ip = i[y:y + size[0], x:x + size[1], :]
                        op = o[y:y + size[0], x:x + size[1], :]
                        if random.randint(0, 10) > 5:
                            ip = np.flip(ip, 1)
                            op = np.flip(op, 1)
                        m = np.sum(op, axis=(0, 1))
                        msum += m
                        metrics.append((len(results), m))
                        results.append((ip, op))
                metrics = sorted(metrics, key=lambda m: -np.sum((m[1] / msum)[:out_channels - 1]))
                metrics = metrics[: int(len(metrics) / 2)]
                random.shuffle(metrics)
                for a in metrics:
                    q.put(results[a[0]])

                if dump_mem:
                    summary.print_(summary.summarize(muppy.get_objects()))
        except MemoryError:
            sys.exit(2)

    queue = Queue(maxsize=50)
    thread = Thread(target=generate_loop, args=(queue, dump_mem))
    thread.start()

    try:
        while True:
            if batch is None:
                yield queue.get()
            else:
                ia, oa = [], []
                for b in range(batch):
                    (i, o) = queue.get()
                    ia.append(i)
                    oa.append(o)
                yield (np.array(ia), np.array(oa))
    except MemoryError:
        sys.exit(2)
