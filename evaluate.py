import argparse
from pathlib import Path

import pandas as pd

from inference import generate_caption, load_checkpoint
from utils.metrics import bleu_scores


def main():
    parser = argparse.ArgumentParser(description='Evaluate generated captions with BLEU.')
    parser.add_argument('--caption-file', type=Path, required=True)
    parser.add_argument('--image-dir', type=Path, required=True)
    parser.add_argument('--test-split', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, nargs='+', required=True)
    args = parser.parse_args()
    test_images = {line.strip() for line in open(args.test_split, encoding='utf-8') if line.strip()}
    dataframe = pd.read_csv(args.caption_file, sep='\t', names=['image_id', 'caption'])
    dataframe['image'] = dataframe.image_id.str.split('#').str[0]
    test_groups = list(dataframe[dataframe.image.isin(test_images)].groupby('image'))
    results = []
    for checkpoint_path in args.checkpoint:
        encoder, decoder, checkpoint = load_checkpoint(checkpoint_path)
        references, hypotheses = [], []
        for image_name, group in test_groups:
            references.append([caption.lower().split() for caption in group.caption])
            caption = generate_caption(
                args.image_dir / image_name,
                encoder,
                decoder,
                checkpoint['vocab_stoi'],
                checkpoint['vocab_itos'],
                checkpoint['max_length'],
            )
            hypotheses.append(caption.lower().split())
        model_name = 'ResNet50 + Attention + LSTM' if checkpoint.get('model_type', 'attention') == 'attention' else 'ResNet50 + LSTM'
        results.append((model_name, bleu_scores(references, hypotheses)))

    print(f'{"Model":36} BLEU-1   BLEU-2   BLEU-3   BLEU-4')
    print('-' * 68)
    for model_name, scores in results:
        print(f'{model_name:36} ' + ' '.join(f'{scores[name]:.4f}  ' for name in scores))


if __name__ == '__main__':
    main()
