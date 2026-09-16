# Captorian: Image Caption Generator

Captorian is a multimodal deep-learning project that generates natural-language captions for images. It uses a pretrained ResNet50 visual encoder, Bahdanau attention over spatial image features, and an LSTM language decoder trained on Flickr8k.

## Project Summary

Given an image, the model learns to generate a sequence of words:

```text
image -> spatial visual features -> attention -> LSTM decoder -> caption
```

The notebook includes dataset preparation, training/validation loss tracking, checkpoint creation, and greedy caption generation. The Streamlit application provides a simple upload-and-caption interface.

## Repository Contents

```text
Captorian/
├── models/                 # ResNet encoder, attention, and LSTM decoder
├── utils/                  # Vocabulary, dataset, collation, and metrics
├── checkpoints/            # Local model artifacts, excluded from Git
├── Caption_generator.ipynb # Original exploration and visual analysis
├── train.py                # Reproducible command-line training
├── evaluate.py             # Test-set BLEU evaluation
├── inference.py            # Command-line inference
├── app.py                  # Streamlit demo
├── requirements.txt
├── pyproject.toml
└── README.md
```

The notebook remains at the repository root for easy review. The Python modules and scripts are the canonical implementation for reuse and deployment.

## Dataset

This project uses the Flickr8k dataset. Download it from Kaggle or another authorized source and place the extracted files in the project directory. The notebook expects:

- `Flickr8k_Dataset/Flicker8k_Dataset/` or `Flickr8k_Dataset/`
- `Flickr8k.token.txt`
- `Flickr_8k.trainImages.txt`
- `Flickr_8k.devImages.txt`
- `Flickr_8k.testImages.txt`

Do not commit `kaggle.json`, the dataset, or trained weights to a public repository.

## Setup

```bash
python -m venv .venv
.venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the modular trainer after placing the dataset under `data/`:

```bash
python train.py
```

Train the CNN-LSTM baseline with the same Flickr8k split:

```bash
python train.py --model baseline --checkpoint checkpoints/caption_baseline_checkpoint.pt
```

Train the attention model explicitly when producing a fresh comparison checkpoint:

```bash
python train.py --model attention --checkpoint checkpoints/caption_attention_checkpoint.pt
```

The notebook can also be opened in VS Code or Jupyter for exploration. Its recommended section is titled **Reproducible attention-model pipeline**. Both paths create:

```text
caption_attention_checkpoint.pt
```

The first run may download pretrained ResNet50 weights. A GPU is recommended, but CPU execution is possible with a smaller batch size and fewer epochs.

## Run the Demo

After training and creating `checkpoints/caption_attention_checkpoint.pt`:

```bash
python inference.py --image_path path/to/image.jpg --checkpoint checkpoints/caption_attention_checkpoint.pt
streamlit run app.py
```

Upload an image in the browser and select **Generate caption**.

## Evaluation

Run test evaluation after training:

```bash
python evaluate.py --caption-file data/Flickr8k_text/Flickr8k.token.txt --image-dir data/Flickr8k_Dataset --test-split data/Flickr8k_text/Flickr_8k.testImages.txt --checkpoint checkpoints/caption_attention_checkpoint.pt
```

Compare both models on the identical test split. The command reports BLEU-1 through BLEU-4:

```bash
python evaluate.py --caption-file data/Flickr8k_text/Flickr8k.token.txt --image-dir data/Flickr8k_Dataset --test-split data/Flickr8k_text/Flickr_8k.testImages.txt --checkpoint checkpoints/caption_baseline_checkpoint.pt checkpoints/caption_attention_checkpoint.pt
```

The output is shaped for a README benchmark table:

```text
Model                                BLEU-1   BLEU-2   BLEU-3   BLEU-4
ResNet50 + LSTM                     <run>    <run>    <run>    <run>
ResNet50 + Attention + LSTM         <run>    <run>    <run>    <run>
```

Generate a token-level attention visualization for up to five words:

```bash
python visualize_attention.py --image path/to/image.jpg --checkpoint checkpoints/caption_attention_checkpoint.pt --output reports/attention_example.png
```

The generated PNG overlays each word's 8x8 Bahdanau spatial attention map on the input image. Add three to five representative PNGs to the project README after running this command.

The current modular evaluator reports BLEU-1 through BLEU-4. METEOR, ROUGE-L, and CIDEr are available in the notebook's optional evaluator, but no benchmark values are claimed because the current notebook has not completed a reproducible end-to-end execution.

Qualitative evaluation is also important. Inspect examples where the model gets objects, actions, counting, or spatial relationships wrong. Caption metrics measure word overlap and should not be treated as a complete measure of image understanding.

## Limitations

- Flickr8k is a small dataset and does not represent all image domains.
- Captions are limited to the vocabulary and style of the training data.
- Greedy decoding can produce generic or repetitive captions.
- The system may miss small objects, count incorrectly, or describe ambiguous images poorly.
- The application is a demonstration, not a safety-critical visual recognition system.

## Future Improvements

- Add beam-search decoding.
- Fine-tune the final ResNet layers with a lower learning rate.
- Add CIDEr-based checkpoint selection.
- Add automated tests for vocabulary, dataset collation, checkpoint loading, and inference.
- Add automated tests for vocabulary, dataset collation, checkpoint loading, and inference.
- Add GitHub Actions for syntax checks and tests.

## LinkedIn Project Description

> Built Captorian, an image-captioning system using transfer learning and multimodal deep learning. The project combines a pretrained ResNet50 visual encoder, Bahdanau spatial attention, and an LSTM language decoder trained on Flickr8k. I implemented reproducible train/validation splitting, shifted teacher-forcing targets, checkpointing, caption generation, and a Streamlit demo for image uploads. The project evaluates generated captions with standard NLP metrics and includes qualitative error analysis and documented limitations.
