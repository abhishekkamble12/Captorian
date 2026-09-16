import os

import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchvision import transforms


class RawImageCaptionDataset(Dataset):
    def __init__(self, image_dir, dataframe, vocab, transform=None):
        self.image_dir = image_dir
        self.dataframe = dataframe.reset_index(drop=True)
        self.vocab = vocab
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image = Image.open(os.path.join(self.image_dir, row['image'])).convert('RGB')
        caption = [self.vocab.stoi['<start>']]
        caption += self.vocab.numericalize(row['caption'])
        caption.append(self.vocab.stoi['<end>'])
        return self.transform(image), torch.tensor(caption, dtype=torch.long)


class CollateBatch:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        images, captions = zip(*batch)
        return torch.stack(images), pad_sequence(captions, batch_first=True, padding_value=self.pad_idx)
