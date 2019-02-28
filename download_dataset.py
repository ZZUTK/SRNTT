import requests
import zipfile
import sys
import os
import argparse
from shutil import rmtree


parser = argparse.ArgumentParser('download_dataset')
parser.add_argument('--dataset_name', type=str, default='CUFED5', help='The name of dataset: CUFED5, DIV2K, or CUFED')
args = parser.parse_args()


CUFED5_TEST_DATA_URL = 'https://drive.google.com/uc?export=download&id=1Fa1mopExA9YGG1RxrCZZn7QFTYXLx6ph'
DIV2K_INPUT_PATCH_DATA_URL = 'https://drive.google.com/uc?export=download&id=1nGeoNLVd-zPifH6sLOYvpY9lVYKnUc0w'
DIV2K_REF_PATCH_DATA_URL = 'https://drive.google.com/uc?export=download&id=1sj72-zL3cGjsVqbbnk3PxJxjQWATQx61'
CUFED_INPUT_PATCH_DATA_URL = 'https://drive.google.com/uc?export=download&id=1gN5IPZgPNkjdeXdTySe1Urog5OG8mrLc'
CUFED_REF_PATCH_DATA_URL = 'https://drive.google.com/uc?export=download&id=13BX-UY4jUZu9S--X2Cd6yZ-3nH77nqo_'


datasets = {
    'CUFED5': {'name': 'CUFED5', 'url': CUFED5_TEST_DATA_URL, 'save_dir': 'data/test', 'data_size': 233},
    'DIV2K_input': {'name': 'DIV2K_input', 'url': DIV2K_INPUT_PATCH_DATA_URL, 'save_dir': 'data/train', 'data_size': 1835},
    'DIV2K_ref': {'name': 'DIV2K_ref', 'url': DIV2K_REF_PATCH_DATA_URL, 'save_dir': 'data/train', 'data_size': 1905},
    'CUFED_input': {'name': 'CUFED_input', 'url': CUFED_INPUT_PATCH_DATA_URL, 'save_dir': 'data/train', 'data_size': 567},
    'CUFED_ref': {'name': 'CUFED_ref', 'url': CUFED_REF_PATCH_DATA_URL, 'save_dir': 'data/train', 'data_size': 588}
}


def download_file_from_google_drive(url, save_dir, data_name, data_size=None):
    if not os.path.exists(os.path.join(save_dir, '%s.zip' % data_name)):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with requests.Session() as session:
            response = session.get(url, stream=True)
            token = get_confirm_token(response)
            if token:
                response = session.get(url, params={'confirm': token}, stream=True)
            save_response_content(response, os.path.join(save_dir, '%s.zip' % data_name), data_size)
    else:
        print('[!] %s already exist! Skip download.' % os.path.join(save_dir, '%s.zip' % data_name))

    if os.path.exists(os.path.join(save_dir, data_name.split('_')[-1])):
        rmtree(os.path.join(save_dir, data_name.split('_')[-1]))

    zip_ref = zipfile.ZipFile(os.path.join(save_dir, '%s.zip' % data_name), 'r')
    if 'train' in save_dir:
        print('>> Unzip %s --> %s' % (os.path.join(save_dir, '%s.zip' % data_name),
                                      os.path.join(save_dir, data_name.split('_')[0], data_name.split('_')[-1])))
        zip_ref.extractall(os.path.join(save_dir, data_name.split('_')[0]))
    else:
        print('>> Unzip %s --> %s' % (os.path.join(save_dir, '%s.zip' % data_name),
                                      os.path.join(save_dir, data_name.split('_')[-1])))
        zip_ref.extractall(save_dir)
    zip_ref.close()


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save_response_content(response, save_dir, data_size=None):
    chunk_size = 1024 * 1024  # in byte
    with open(save_dir, "wb") as f:
        len_content = 0
        for chunk in response.iter_content(chunk_size):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                len_content += len(chunk)
                if data_size is not None:
                    sys.stdout.write('\r>> Downloading %s %.1f%%' % (save_dir, min(len_content / 1024. / 1024. / data_size * 100, 100)))
                    sys.stdout.flush()
                else:
                    sys.stdout.write('\r>> Downloading %s %.1f MB' % (save_dir, len_content / 1024. / 1024.))
                    sys.stdout.flush()
        print('')


if __name__ == "__main__":
    is_downloaded = False
    for key in datasets:
        if args.dataset_name == key.split('_')[0]:
            dataset = datasets[key]
            download_file_from_google_drive(
                url=dataset['url'],
                save_dir=dataset['save_dir'],
                data_name=dataset['name'],
                data_size=dataset['data_size']
            )
            is_downloaded = True
    if not is_downloaded:
        print('''[!] Unrecognized dataset name "%s"''' % args.dataset_name)
