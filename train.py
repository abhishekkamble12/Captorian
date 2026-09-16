import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import DecoderWithAttention, DecoderWithoutAttention, SpatialEncoderCNN
from utils import CollateBatch, RawImageCaptionDataset, Vocabulary


def load_captions(caption_file):
    rows = []
    with open(caption_file, encoding='utf-8') as handle:
        for line in handle:
            image_id, caption = line.rstrip().split('\t', maxsplit=1)
            rows.append({'image': image_id.split('#')[0], 'caption': caption})
    dataframe = pd.DataFrame(rows)
    dataframe['word_count'] = dataframe['caption'].str.split().str.len()
    return dataframe


def load_split(path):
    return [line.strip() for line in open(path, encoding='utf-8') if line.strip()]


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    captions = load_captions(args.caption_file)
    train_images = load_split(args.train_split)
    val_images = load_split(args.val_split)
    train_df = captions[captions.image.isin(train_images)]
    val_df = captions[captions.image.isin(val_images)]
    vocab = Vocabulary(args.frequency_threshold)
    vocab.build_vocabulary(train_df.caption.tolist())
    train_set = RawImageCaptionDataset(args.image_dir, train_df, vocab)
    val_set = RawImageCaptionDataset(args.image_dir, val_df, vocab)
    collate = CollateBatch(vocab.stoi['<pad>'])
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_set, args.batch_size, shuffle=False, collate_fn=collate)
    encoder = SpatialEncoderCNN(trainable=False).to(device)
    decoder_class = DecoderWithAttention if args.model == 'attention' else DecoderWithoutAttention
    decoder = decoder_class(len(vocab)).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi['<pad>'])
    optimizer = torch.optim.Adam(decoder.parameters(), lr=args.learning_rate)
    for epoch in range(args.epochs):
        encoder.eval()
        decoder.train()
        train_loss = 0.0
        for images, captions_batch in train_loader:
            images, captions_batch = images.to(device), captions_batch.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                features = encoder(images)
            outputs = decoder(features, captions_batch[:, :-1])
            loss = criterion(outputs.reshape(-1, len(vocab)), captions_batch[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f'epoch={epoch + 1} train_loss={train_loss / max(1, len(train_loader)):.4f}')
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model_type': args.model, 'encoder_state_dict': encoder.state_dict(), 'decoder_state_dict': decoder.state_dict(), 'vocab_stoi': vocab.stoi, 'vocab_itos': vocab.itos, 'max_length': int(train_df.word_count.max()) + 2}, args.checkpoint)
    print(f'saved checkpoint: {args.checkpoint}')


def main():
    parser = argparse.ArgumentParser(description='Train the Captorian attention model.')
    parser.add_argument('--image-dir', type=Path, default='data/Flickr8k_Dataset')
    parser.add_argument('--caption-file', type=Path, default='data/Flickr8k_text/Flickr8k.token.txt')
    parser.add_argument('--train-split', type=Path, default='data/Flickr8k_text/Flickr_8k.trainImages.txt')
    parser.add_argument('--val-split', type=Path, default='data/Flickr8k_text/Flickr_8k.devImages.txt')
    parser.add_argument('--checkpoint', type=Path, default='checkpoints/caption_attention_checkpoint.pt')
    parser.add_argument('--model', choices=['attention', 'baseline'], default='attention')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--frequency-threshold', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    required = [args.image_dir, args.caption_file, args.train_split, args.val_split]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        parser.error('Missing dataset paths: ' + ', '.join(missing))
    train(args)


if __name__ == '__main__':
    main()
