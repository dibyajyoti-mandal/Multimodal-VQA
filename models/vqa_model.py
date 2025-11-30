import torch
import torch.nn as nn
from .cnn import CNN_Feature_Extractor_pretrained
from .encoder import Question_Encoder
from .decoder import Answer_Decoder


class VQA_Model(nn.Module):
    def __init__(self, questions_vocab_size, answers_vocab_size, k_beam=3, device=None):
        super(VQA_Model, self).__init__()

        self.device = device
        self.image_encoder_resnet50_pretrained = CNN_Feature_Extractor_pretrained()
        self.question_encoder = Question_Encoder(questions_vocab_size)

        self.answer_decoder = Answer_Decoder(answers_vocab_size, k_beam=k_beam)

    def forward(self, image, question, answer_seq=None):
        image_feat = self.image_encoder_resnet50_pretrained(image)
        question_feat = self.question_encoder(question)
        output = self.answer_decoder(question_feat, image_feat, answer_seq)
        return output
