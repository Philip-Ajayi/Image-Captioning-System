from collections import Counter
import nltk

class SimpleTokenizer:
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold
        self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}

    def __len__(self):
        return len(self.word2idx)

    def tokenize(self, text):
        return nltk.word_tokenize(text.lower())

    def build_vocab(self, sentence_list):
        frequencies = Counter()
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenize(text)
        return [self.word2idx.get(word, self.word2idx["<UNK>"]) for word in tokenized_text]

    def encode(self, text):
        return [self.word2idx["<SOS>"]] + self.numericalize(text) + [self.word2idx["<EOS>"]]
