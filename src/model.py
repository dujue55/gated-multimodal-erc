import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

FUSIONS = ('text', 'speech', 'gated', 'fixed')


class EmotionModel(nn.Module):
    """Utterance features -> (fusion) -> GRU over the context -> classifier.

    All variants share the GRU and the classifier head; they differ only in the input:
      text   : text features only
      speech : speech features only
      gated  : g * text + (1 - g) * speech, with a learned per-dimension gate g in (0, 1)
      fixed  : 0.5 * text + 0.5 * speech (no learned gate)
    """

    def __init__(self, fusion, text_dim=768, speech_dim=768, hidden_size=256, num_classes=4,
                 dropout=0.3):
        super().__init__()
        if fusion not in FUSIONS:
            raise ValueError(f'fusion must be one of {FUSIONS}')
        self.fusion = fusion

        if fusion in ('gated', 'fixed'):
            # Map speech features to the text dimension so the two can be mixed
            self.speech_proj = nn.Linear(speech_dim, text_dim) if speech_dim != text_dim else nn.Identity()
            input_dim = text_dim
        else:
            input_dim = text_dim if fusion == 'text' else speech_dim

        if fusion == 'gated':
            self.gate = nn.Sequential(
                nn.Linear(2 * text_dim, text_dim // 2),
                nn.ReLU(),
                nn.LayerNorm(text_dim // 2),
                nn.Linear(text_dim // 2, text_dim),
                nn.Sigmoid(),
            )

        self.gru = nn.GRU(input_dim, hidden_size, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, F_t, F_s, lengths):
        """F_t, F_s: [B, L, D] padded sequences; lengths: [B]. Returns (logits, gate or None)."""
        gate = None
        if self.fusion == 'text':
            x = F_t
        elif self.fusion == 'speech':
            x = F_s
        else:
            F_s = self.speech_proj(F_s)
            if self.fusion == 'gated':
                gate = self.gate(torch.cat((F_t, F_s), dim=-1))   # [B, L, D], weight of text
                x = gate * F_t + (1 - gate) * F_s
            else:
                x = 0.5 * F_t + 0.5 * F_s

        # Packing makes the GRU stop at each sequence's true length, so padding never
        # reaches the final hidden state.
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)
        return self.classifier(h_n[-1]), gate
