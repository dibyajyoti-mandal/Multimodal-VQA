import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from .preprocess import text_to_tensor


class VQADataset(Dataset):
    def __init__(self, csv_path, image_folder, vocab_questions, vocab_answers, len_max_question=24, len_max_answer=36, transform=None):
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        self.df['image_id'] = self.df['image_id'].astype(str) + '.png'
        self.image_folder = image_folder
        self.transform = transform
        self.vocab_questions = vocab_questions
        self.vocab_answers = vocab_answers
        self.len_max_question = len_max_question
        self.len_max_answer = len_max_answer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_folder, row['image_id'])
        question = text_to_tensor(row['question'], self.vocab_questions, self.len_max_question)
        answer = text_to_tensor(row['response'], self.vocab_answers, self.len_max_answer)

        if os.path.exists(image_path):
            img = Image.open(image_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
        else:
            img = torch.zeros((3, 224, 224))
            print(f"Image not found: {image_path}")

        return img, question, answer
