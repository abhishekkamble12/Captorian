import re
from collections import Counter


class Vocabulary:
    def __init__(self, freq_threshold=2):
        self.itos = {0: '<pad>', 1: '<start>', 2: '<end>', 3: '<unk>'}
        self.stoi = {token: index for index, token in self.itos.items()}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer(text):
        return re.sub(r'[^a-zA-Z\\s]', '', text.lower()).strip().split()

    def build_vocabulary(self, captions):
        frequencies = Counter()
        for caption in captions:
            frequencies.update(self.tokenizer(caption))
        for word in sorted(word for word, count in frequencies.items() if count >= self.freq_threshold):
            index = len(self.itos)
            self.stoi[word] = index
            self.itos[index] = word

    def numericalize(self, text):
        unknown = self.stoi['<unk>']
        return [self.stoi.get(token, unknown) for token in self.tokenizer(text)]
