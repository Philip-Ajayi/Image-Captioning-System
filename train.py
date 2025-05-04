import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from datasets.build_dataset import FlickrDataset
from utils.tokenizer import SimpleTokenizer
from models.captioning_model import CaptioningModel
from torchvision import transforms
from utils import config

def collate_fn(batch):
    images, captions = zip(*batch)
    images = torch.stack(images, 0)
    lengths = [len(cap) for cap in captions]
    padded_captions = nn.utils.rnn.pad_sequence(captions, batch_first=True, padding_value=0)
    return images, padded_captions, lengths

def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    tokenizer = SimpleTokenizer(freq_threshold=config.FREQ_THRESHOLD)

    # Build vocab
    with open('./data/Flickr8k_text/Flickr8k.token.txt', 'r') as f:
        sentences = [line.strip().split('\t')[1] for line in f.readlines()]
    tokenizer.build_vocab(sentences)

    dataset = FlickrDataset(
        root_dir='./data/Flicker8k_Dataset',
        captions_file='./data/Flickr8k_text/Flickr8k.token.txt',
        transform=transform,
        tokenizer=tokenizer
    )

    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    model = CaptioningModel(config.EMBED_SIZE, config.HIDDEN_SIZE, len(tokenizer))

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    params = model.parameters()
    optimizer = optim.Adam(params, lr=config.LEARNING_RATE)

    model.encoder.train()
    model.decoder.train()

    for epoch in range(config.NUM_EPOCHS):
        for idx, (imgs, captions, lengths) in enumerate(loader):
            outputs = model.decoder(model.encoder(imgs), captions[:, :-1])
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions[:, 1:].reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx % 100 == 0:
                print(f"Epoch [{epoch}/{config.NUM_EPOCHS}] Batch {idx}/{len(loader)} Loss {loss.item():.4f}")

if __name__ == '__main__':
    main()
