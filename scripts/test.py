import argparse
import torch
from PIL import Image
from data.preprocess import text_to_tensor, image_transform, build_vocab
from models.vqa_model import VQA_Model
from utils.text import tensor_to_text


def load_vocab_from_csv(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path)
    return build_vocab(df['question']), build_vocab(df['response'])


def load_model_checkpoint(model, checkpoint_path, device='cpu'):
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def inference_single(model, question, image_path, idx2word_answers, device='cpu'):
    image = Image.open(image_path).convert('RGB')
    image_tensor = image_transform(image).unsqueeze(0).to(device)

    question_tensor = text_to_tensor(question, vocab_questions, 24).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor, question_tensor)

    predicted = tensor_to_text(output, idx2word_answers)[0]
    return predicted


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_checkpoint', required=True)
    parser.add_argument('--question', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--vocab_csv', required=True, help='CSV used to build vocab (train CSV)')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    device = torch.device(args.device)

    vocab_questions, vocab_answers = load_vocab_from_csv(args.vocab_csv)
    idx2word_answers = {idx: w for w, idx in vocab_answers.items()}

    model = VQA_Model(len(vocab_questions), len(vocab_answers))
    model = load_model_checkpoint(model, args.model_checkpoint, device=device)

    predicted = inference_single(model, args.question, args.image, idx2word_answers, device=device)
    print('Predicted answer:', predicted)
