import time
import os
import torch
import torch.nn as nn

from utils.text import tensor_to_text
from utils.metrics import compute_bleu


def train_model(model, train_loader, eval_loader, criterion, optimizer, best_model_path, num_epochs=10, patience=5, device='cpu'):
    model.to(device)
    best_loss = float('inf')
    no_improve_epochs = 0

    history = {
        "train_loss": [],
        "eval_loss": [],
        "bleu_score": []
    }

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        # Training
        model.train()
        total_loss = 0.0
        for images, questions, answers in train_loader:
            images, questions, answers = images.to(device), questions.to(device), answers.to(device)

            optimizer.zero_grad()
            output = model(images, questions, answers[:, :-1])
            loss = criterion(output.view(-1, output.size(-1)), answers[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0

        # Evaluation
        model.eval()
        eval_loss = 0.0
        bleu_scores = []
        with torch.no_grad():
            for images, questions, answers in eval_loader:
                images, questions, answers = images.to(device), questions.to(device), answers.to(device)
                output = model(images, questions, answers[:, :-1])

                loss = criterion(output.view(-1, output.size(-1)), answers[:, 1:].reshape(-1))
                eval_loss += loss.item()

                preds = tensor_to_text(model(images, questions), getattr(model, 'idx2word_answers', {}))
                gts = tensor_to_text(answers, getattr(model, 'idx2word_answers', {}))
                bleu_scores.append(compute_bleu(preds, gts))

        avg_eval_loss = eval_loss / len(eval_loader) if len(eval_loader) > 0 else 0
        avg_bleu_score = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0

        epoch_time = time.time() - start_time

        print(f"Epoch [{epoch}/{num_epochs}] Train Loss: {avg_train_loss:.4f} Eval Loss: {avg_eval_loss:.4f} BLEU: {avg_bleu_score:.4f} Time: {epoch_time:.2f}s")

        history['train_loss'].append(avg_train_loss)
        history['eval_loss'].append(avg_eval_loss)
        history['bleu_score'].append(avg_bleu_score)

        # Early stopping / checkpoint
        if avg_eval_loss < best_loss:
            best_loss = avg_eval_loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model to {best_model_path}")
        else:
            no_improve_epochs += 1
            print(f"No improvement for {no_improve_epochs}/{patience} epochs")
            if no_improve_epochs >= patience:
                print("Early stopping triggered")
                break

    return history


def train_vqa_from_csv(train_csv, val_csv, image_folder, model, optimizer, criterion, batch_size=16, num_epochs=10, device='cpu'):
    import pandas as pd
    from data.dataset import VQADataset
    from data.preprocess import build_vocab, get_max_len, image_transform
    from torch.utils.data import DataLoader

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    vocab_questions = build_vocab(train_df['question'])
    vocab_answers = build_vocab(train_df['response'])

    len_max_question = 24
    len_max_answer = 36

    train_dataset = VQADataset(train_csv, image_folder, vocab_questions, vocab_answers, len_max_question, len_max_answer, transform=image_transform)
    val_dataset = VQADataset(val_csv, image_folder, vocab_questions, vocab_answers, len_max_question, len_max_answer, transform=image_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    best_model_path = os.path.join(os.getcwd(), 'vqa_best.pth')

    history = train_model(model, train_loader, val_loader, criterion, optimizer, best_model_path, num_epochs=num_epochs, patience=5, device=device)

    return history, best_model_path
