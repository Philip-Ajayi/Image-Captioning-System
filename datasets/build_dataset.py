import os
import nltk
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

nltk.download('punkt')

class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, tokenizer=None, max_length=50):
        self.root_dir = root_dir
        self.transform = transform
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.image_ids = []
        self.captions = []
        
        with open(captions_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines[1:]:
            tokens = line.strip().split('\t')
            if len(tokens) == 2:
                img_id, caption = tokens
                img_id = img_id[:-2]  # remove #0, #1 etc.
                self.image_ids.append(img_id)
                self.captions.append(caption)

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_ids[idx])
        image = Image.open(img_path).convert("RGB")
        caption = self.captions[idx]
        
        if self.transform is not None:
            image = self.transform(image)
        
        tokens = self.tokenizer.encode(caption)
        tokens = tokens[:self.max_length]
        return image, torch.tensor(tokens)
