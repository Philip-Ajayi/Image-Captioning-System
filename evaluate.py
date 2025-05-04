import torch
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import corpus_bleu
from datasets.build_dataset import FlickrDataset
from utils.tokenizer import SimpleTokenizer
from models.captioning_model import CaptioningModel
from torchvision import transforms
from utils import config

def evaluate(model, data_loader, tokenizer):
    model.eval()  # Set the model to evaluation mode
    references = []
    hypotheses = []

    with torch.no_grad():
        for images, captions, lengths in data_loader:
            outputs = model.decoder(model.encoder(images), captions[:, :-1])

            # Decode the generated captions
            predicted_ids = torch.argmax(outputs, dim=-1)
            
            for idx in range(predicted_ids.size(0)):
                # Convert predicted tokens to words
                predicted_caption = [tokenizer.idx2word[token.item()] for token in predicted_ids[idx]]
                references.append([captions[idx][1:-1].tolist()])  # excluding <SOS> and <EOS>
                hypotheses.append(predicted_caption)

    # Calculate BLEU score
    bleu_score = corpus_bleu(references, hypotheses)
    return bleu_score

def main():
    # Load the model and tokenizer
    tokenizer = SimpleTokenizer(freq_threshold=config.FREQ_THRESHOLD)
    tokenizer.build_vocab(sentences)

    # Load the test dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    test_dataset = FlickrDataset(
        root_dir='./data/Flicker8k_Dataset',
        captions_file='./data/Flickr8k_text/Flickr8k.token.txt',
        transform=transform,
        tokenizer=tokenizer
    )

    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Load the trained model
    model = CaptioningModel(config.EMBED_SIZE, config.HIDDEN_SIZE, len(tokenizer))
    model.load_state_dict(torch.load('captioning_model.pth'))
    model.to(device)  # Ensure the model is on the right device (CPU/GPU)

    bleu_score = evaluate(model, test_loader, tokenizer)
    print(f"BLEU Score: {bleu_score:.4f}")

if __name__ == '__main__':
    main()
