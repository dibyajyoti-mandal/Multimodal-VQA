import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import Attention


class Answer_Decoder(nn.Module):
    def __init__(self, answer_vocab_size, embedding_size=256, hidden_dim=512, k_beam=3):
        super(Answer_Decoder, self).__init__()
        self.embedding = nn.Embedding(answer_vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size + 1024, hidden_dim, num_layers=3, dropout=0.2, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, answer_vocab_size)

        self.k_beam = k_beam

    def forward(self, question_feat, image_feat, answer_seq=None, answer_maxlength=36):
        combined_feat = torch.cat((question_feat, image_feat), dim=1)

        if answer_seq is not None:
            x = self.embedding(answer_seq)
            hidden_state = None
            outputs = []

            for i in range(x.size(1)):
                context, _ = self.attention(hidden_state[0][-1] if hidden_state else question_feat, combined_feat)
                lstm_input = torch.cat((x[:, i, :], context), dim=1).unsqueeze(1)
                output, hidden_state = self.lstm(lstm_input, hidden_state)
                outputs.append(self.fc(output.squeeze(1)))

            return torch.stack(outputs, dim=1)

        else:
            batch_size = combined_feat.size(0)
            device = image_feat.device
            end_token = 3

            all_results = []

            for b in range(batch_size):
                b_question_feat = question_feat[b:b+1]
                b_combined_feat = combined_feat[b:b+1]

                beams = [(torch.tensor([[2]], dtype=torch.long, device=device),
                          0.0,
                          None)]

                completed_beams = []

                for _ in range(answer_maxlength):
                    candidates = []

                    for seq, score, hidden_state in beams:
                        if seq[0, -1].item() == end_token:
                            completed_beams.append((seq, score, hidden_state))
                            continue

                        x = self.embedding(seq[:, -1])

                        prev_hidden = hidden_state[0][-1] if hidden_state else b_question_feat
                        context, _ = self.attention(prev_hidden, b_combined_feat)

                        lstm_input = torch.cat((x, context), dim=1).unsqueeze(1)

                        output, new_hidden = self.lstm(lstm_input, hidden_state)

                        logits = self.fc(output.squeeze(1))
                        log_probs = F.log_softmax(logits, dim=1)

                        topk_log_probs, topk_indices = log_probs.topk(self.k_beam)

                        for i in range(self.k_beam):
                            next_token = topk_indices[:, i:i+1]
                            next_score = score + topk_log_probs[:, i].item()
                            next_seq = torch.cat([seq, next_token], dim=1)
                            candidates.append((next_seq, next_score, new_hidden))

                    if not candidates:
                        break

                    candidates.sort(key=lambda x: x[1], reverse=True)
                    beams = candidates[:self.k_beam]

                    if all(beam[0][0, -1].item() == end_token for beam in beams):
                        completed_beams.extend(beams)
                        break

                if completed_beams:
                    completed_beams.sort(key=lambda x: x[1], reverse=True)
                    best_seq = completed_beams[0][0]
                else:
                    beams.sort(key=lambda x: x[1], reverse=True)
                    best_seq = beams[0][0]

                all_results.append(best_seq)

            max_len = max(seq.size(1) for seq in all_results)
            padded_results = []

            for seq in all_results:
                if seq.size(1) < max_len:
                    padding = torch.full((1, max_len - seq.size(1)), end_token, dtype=torch.long, device=device)
                    padded_seq = torch.cat([seq, padding], dim=1)
                    padded_results.append(padded_seq)
                else:
                    padded_results.append(seq)

            return torch.cat(padded_results, dim=0)
