from .encoder import EncoderCNN
from .decoder import DecoderRNN

class CaptioningModel:
    def __init__(self, embed_size, hidden_size, vocab_size):
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
    
    def parameters(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())
