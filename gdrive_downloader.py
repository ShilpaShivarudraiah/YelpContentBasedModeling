import gdown, sys

def g_downloader(file_id):
    prefix = 'https://drive.google.com/uc?/export=download&id='

    gdown.download(prefix+file_id)