Project: VQA (ResNet50 + LSTM)

Structure (created by conversion):
- `src/data` - dataset and preprocessing utilities
- `src/models` - model components and VQA model
- `src/utils` - metrics, text utilities, training and plotting helpers
- `src/scripts` - runnable scripts (`train.py`, `test.py`)

Quick start (after installing dependencies from `requirements.txt`):

Train (example):

```bash
python -m src.scripts.train --train_csv /path/to/train.csv --val_csv /path/to/val.csv --image_folder /path/to/images --device cpu
```

Test single sample:

```bash
python -m src.scripts.test --model_checkpoint ./vqa_best.pth --question "What is in the image?" --image /path/to/image.png --vocab_csv /path/to/train.csv
```

Notes:
- The package expects `en_core_web_sm` to be available for spaCy. Install with `python -m spacy download en_core_web_sm`.
- Adjust batch sizes and num_workers for your machine.
