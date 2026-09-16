import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from models import DecoderWithAttention, DecoderWithoutAttention, SpatialEncoderCNN


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    encoder = SpatialEncoderCNN().to(DEVICE)
    decoder_class = DecoderWithAttention if checkpoint.get('model_type', 'attention') == 'attention' else DecoderWithoutAttention
    decoder = decoder_class(len(checkpoint['vocab_itos'])).to(DEVICE)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    encoder.eval()
    decoder.eval()
    return encoder, decoder, checkpoint


def generate_caption(image_path, checkpoint_path=None, decoder=None, vocab_stoi=None, vocab_itos=None, max_length=None, return_attention=False):
    if isinstance(checkpoint_path, torch.nn.Module):
        encoder = checkpoint_path
        checkpoint = {'vocab_stoi': vocab_stoi, 'vocab_itos': vocab_itos, 'max_length': max_length}
    else:
        encoder, decoder, checkpoint = load_checkpoint(checkpoint_path)
    stoi = checkpoint['vocab_stoi']
    itos = checkpoint['vocab_itos']
    max_length = checkpoint['max_length']
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    source_image = image_path if isinstance(image_path, Image.Image) else Image.open(image_path)
    image = preprocess(source_image.convert('RGB')).unsqueeze(0).to(DEVICE)
    tokens = [stoi['<start>']]
    attention_maps = []
    with torch.no_grad():
        encoder_out = encoder(image)
        hidden, cell = decoder.init_hidden_state(encoder_out)
        for _ in range(max_length):
            current = torch.tensor([tokens[-1]], device=DEVICE)
            embedding = decoder.embedding(current)
            if hasattr(decoder, 'attention'):
                context, weights = decoder.attention(encoder_out, hidden)
                attention_maps.append(weights[0, :, 0].reshape(8, 8).cpu())
            else:
                context = encoder_out.mean(dim=1)
            hidden, cell = decoder.lstm_cell(torch.cat([embedding, context], dim=1), (hidden, cell))
            next_token = decoder.fc(hidden).argmax(dim=1).item()
            if next_token == stoi['<end>']:
                break
            tokens.append(next_token)
    special = {stoi['<pad>'], stoi['<start>'], stoi['<end>']}
    caption = ' '.join(itos[token] for token in tokens[1:] if token not in special)
    if return_attention:
        return caption, [(itos[token], attention_maps[index]) for index, token in enumerate(tokens[1:]) if token not in special]
    return caption


def main():
    parser = argparse.ArgumentParser(description='Generate a caption for an image.')
    parser.add_argument('--image_path', required=True, type=Path)
    parser.add_argument('--checkpoint', default='caption_attention_checkpoint.pt', type=Path)
    args = parser.parse_args()
    if not args.image_path.exists():
        parser.error(f'Image not found: {args.image_path}')
    if not args.checkpoint.exists():
        parser.error('Checkpoint not found. Run train.py first or provide --checkpoint.')
    print(generate_caption(args.image_path, args.checkpoint))


if __name__ == '__main__':
    main()
