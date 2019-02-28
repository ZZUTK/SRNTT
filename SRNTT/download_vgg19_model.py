from six.moves import urllib
import sys
import os

DATA_URL = 'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat'


def download_vgg19(model_dir):
    filename = DATA_URL.split('/')[-1]
    if not os.path.exists(model_dir):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, model_dir, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
