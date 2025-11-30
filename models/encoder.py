import torch
import torch.nn as nn
import torch.nn.functional as F


class Question_Encoder(nn.Module):
    def __init__(self, questions_vocab_size, embedding_dim=256, hidden_dim=512):
        super(Question_Encoder, self).__init__()
        self.embedding = nn.Embedding(questions_vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, dropout=0.2, batch_first=True)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        return hidden[-1]


class Attention(nn.Module):
    def __init__(self, hidden_dim=512):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 3, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, combined_feat):
        if hidden is None:
            # If no hidden (rare), expand zeros
            hidden = torch.zeros((combined_feat.size(0), combined_feat.size(1) // 2), device=combined_feat.device)

        if hidden.dim() > 2:
            hidden = hidden.squeeze(0)

        if hidden.dim() == 1:
            hidden = hidden.unsqueeze(0)

        energy = torch.tanh(self.attn(torch.cat((hidden, combined_feat), dim=1)))
        attention_weights = F.softmax(self.v(energy), dim=1)

        context = attention_weights * combined_feat
        return context, attention_weights
