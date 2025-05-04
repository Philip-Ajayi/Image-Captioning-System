import torch
from PIL import Image
from utils.tokenizer import SimpleTokenizer
from models.captioning_model import CaptioningModel
from torchvision import transforms
from utils import config

def generate_caption(model, image, tokenizer, max_length=50):
    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to the right device

    # Encode the image with the encoder
    features = model.encoder(image)

    # Generate the caption using the decoder
    caption = [tokenizer.word2idx["<SOS>"]]
    for _ in range(max_length):
        input_caption = torch.tensor(caption).unsqueeze(0).to(device)
        output = model.decoder(features, input_caption)
        predicted_token = torch.argmax(output[0, -1, :]).item()

        if predicted_token == tokenizer.word2idx["<EOS>"]:
            break

        caption.append(predicted_token)

    caption_words = [tokenizer.idx2word[token] for token in caption[1:]]
    return ' '.join(caption_words)

def main(image_path):
    # Load the model and tokenizer
    tokenizer = SimpleTokenizer(freq_threshold=config.FREQ_THRESHOLD)
    tokenizer.build_vocab(sentences)

    # Load the trained model
    model = CaptioningModel(config.EMBED_SIZE, config.HIDDEN_SIZE, len(tokenizer))
    model.load_state_dict(torch.load('captioning_model.pth'))
    model.to(device)  # Ensure the model is on the right device (CPU/GPU)

    # Load and process the image
    image = Image.open(image_path).convert("RGB")

    # Generate caption for the image
    caption = generate_caption(model, image, tokenizer)
    print(f"Generated Caption: {caption}")

if __name__ == '__main__':
    image_path = 'path_to_your_image.jpg'  # Path to the image file
    main(image_path)
