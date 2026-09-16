import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from inference import generate_caption, load_checkpoint


def main():
    parser = argparse.ArgumentParser(description='Render token-level Bahdanau attention overlays.')
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('attention_visualization.png'))
    parser.add_argument('--max-words', type=int, default=5)
    args = parser.parse_args()

    encoder, decoder, checkpoint = load_checkpoint(args.checkpoint)
    caption, word_maps = generate_caption(
        args.image,
        encoder,
        decoder,
        checkpoint['vocab_stoi'],
        checkpoint['vocab_itos'],
        checkpoint['max_length'],
        return_attention=True,
    )
    if not word_maps:
        raise SystemExit('This checkpoint does not expose spatial attention weights.')

    image = Image.open(args.image).convert('RGB')
    selected = word_maps[:args.max_words]
    figure, axes = plt.subplots(1, len(selected), figsize=(3 * len(selected), 3.5), squeeze=False)
    for axis, (word, weights) in zip(axes[0], selected):
        axis.imshow(image)
        axis.imshow(np.asarray(Image.fromarray(weights.numpy()).resize(image.size)), cmap='jet', alpha=0.45)
        axis.set_title(word)
        axis.axis('off')
    figure.suptitle(f'Generated caption: {caption}', fontsize=11)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches='tight')
    print(f'saved attention visualization: {args.output}')


if __name__ == '__main__':
    main()