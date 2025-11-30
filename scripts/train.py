import argparse
import os
import torch
import torch.nn as nn
from torch.optim import AdamW

from data.preprocess import build_vocab
from models.vqa_model import VQA_Model
from utils.train import train_vqa_from_csv


def main(train_csv, val_csv, image_folder, device='cpu'):
    import pandas as pd

    train_df = pd.read_csv(train_csv)

    vocab_questions = build_vocab(train_df['question'])
    vocab_answers = build_vocab(train_df['response'])

    questions_vocab_size = len(vocab_questions)
    answers_vocab_size = len(vocab_answers)

    model = VQA_Model(questions_vocab_size, answers_vocab_size)

    criterion = nn.CrossEntropyLoss(ignore_index=1)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    history, best_model_path = train_vqa_from_csv(train_csv, val_csv, image_folder, model, optimizer, criterion, batch_size=16, num_epochs=10, device=device)
    print('Training finished. Best model at:', best_model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', required=True)
    parser.add_argument('--val_csv', required=True)
    parser.add_argument('--image_folder', required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    main(args.train_csv, args.val_csv, args.image_folder, args.device)
