import sys
import subprocess
from datetime import datetime

#args = sys.argv[1:]
model_name = 'boe9_3x2_16-256' # 'boe8_2x2_16-128'
args = ['C:/Python36-64/python', '-u', 'learn.py', model_name]


def write(string):
    with open('launch_{0}.txt'.format(model_name), 'a') as log:
        out = str(datetime.now()) + ' ' + string.strip()
        log.write(out + '\n')
        print(out)


try:
    while True:
        p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        write('Launched ' + ' '.join(args))
        for line in p.stdout:
            try:
                write(line.decode('utf-8'))
            except UnicodeDecodeError:
                write('[... Unable to decode ...]')
        p.stdout.close()
        p.wait()
        write('Process terminated with exit code ' + str(p.returncode))
except KeyboardInterrupt:
    write('Launcher terminating...')
    p.send_signal(subprocess.signal.CTRL_C_EVENT)
