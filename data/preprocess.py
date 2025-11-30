import spacy
from collections import Counter
import torch
import torchvision.transforms as transforms

# Load spaCy tokenizer (ensure 'en_core_web_sm' is installed in the environment)
nlp = spacy.load("en_core_web_sm")

# Image transforms used in notebook
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def spacy_tokenizer(text: str):
    tokens = []
    for token in nlp(text):
        if not token.is_punct and not token.is_space:
            tokens.append(token.text.lower())
    return tokens


def build_vocab(texts, min_freq: int = 1):
    counter = Counter()
    for text in texts:
        counter.update(spacy_tokenizer(text))

    vocab = {
        "<unk>": 0,
        "<pad>": 1,
        "<sos>": 2,
        "<eos>": 3,
    }

    index = 4
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = index
            index += 1
    return vocab


def get_max_len(texts):
    max_len = 0
    for text in texts:
        tokens = spacy_tokenizer(text)
        if len(tokens) > max_len:
            max_len = len(tokens)
    return max_len


def text_to_tensor(text: str, vocab: dict, max_len: int):
    tokens = spacy_tokenizer(text)
    tokens = ["<sos>"] + tokens + ["<eos>"]

    indices = [vocab.get(token, vocab["<unk>"]) for token in tokens]

    if len(indices) < max_len:
        indices += [vocab["<pad>"]] * (max_len - len(indices))
    else:
        indices = indices[:max_len]

    return torch.tensor(indices, dtype=torch.long)


def decode_answer(tensor, vocab_dict):
    return " ".join([vocab_dict[idx] for idx in tensor if idx not in {0, 1}])
