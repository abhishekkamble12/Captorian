import argparse
from pathlib import Path

import pandas as pd

from inference import generate_caption
from utils.metrics import bleu_scores


def main():
    parser = argparse.ArgumentParser(description='Evaluate generated captions with BLEU.')
    parser.add_argument('--caption-file', type=Path, required=True)
    parser.add_argument('--image-dir', type=Path, required=True)
    parser.add_argument('--test-split', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args()
    test_images = {line.strip() for line in open(args.test_split, encoding='utf-8') if line.strip()}
    dataframe = pd.read_csv(args.caption_file, sep='\t', names=['image_id', 'caption'])
    dataframe['image'] = dataframe.image_id.str.split('#').str[0]
    references, hypotheses = [], []
    for image_name, group in dataframe[dataframe.image.isin(test_images)].groupby('image'):
        references.append([caption.lower().split() for caption in group.caption])
        hypotheses.append(generate_caption(args.image_dir / image_name, args.checkpoint).lower().split())
    for name, score in bleu_scores(references, hypotheses).items():
        print(f'{name}: {score:.4f}')


if __name__ == '__main__':
    main()
