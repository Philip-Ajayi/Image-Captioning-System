import os
import requests
from tqdm import tqdm
import zipfile

FLICKR8K_URL = 'https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip'
CAPTIONS_URL = 'https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip'
DATA_DIR = './data'

def download_file(url, output_path):
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    with open(output_path, 'wb') as file, tqdm(
            desc=f"Downloading {os.path.basename(output_path)}",
            total=total,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

if __name__ == '__main__':
    os.makedirs(DATA_DIR, exist_ok=True)
    dataset_zip = os.path.join(DATA_DIR, 'Flickr8k_Dataset.zip')
    captions_zip = os.path.join(DATA_DIR, 'Flickr8k_text.zip')
    
    if not os.path.exists(dataset_zip):
        download_file(FLICKR8K_URL, dataset_zip)
        extract_zip(dataset_zip, DATA_DIR)
    else:
        print("Dataset already downloaded.")
    
    if not os.path.exists(captions_zip):
        download_file(CAPTIONS_URL, captions_zip)
        extract_zip(captions_zip, DATA_DIR)
    else:
        print("Captions already downloaded.")
      
