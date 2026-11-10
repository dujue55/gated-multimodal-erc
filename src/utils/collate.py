import torch
from torch.nn.utils.rnn import pad_sequence


def collate_epc(batch):
    """Pad variable-length context sequences to the longest one in the batch."""
    lengths = torch.tensor([b['F_t'].shape[0] for b in batch], dtype=torch.long)
    mask = torch.arange(int(lengths.max()))[None, :] < lengths[:, None]   # [B, L], True = real
    return {
        'F_t': pad_sequence([b['F_t'] for b in batch], batch_first=True),   # [B, L, D_t]
        'F_s': pad_sequence([b['F_s'] for b in batch], batch_first=True),   # [B, L, D_s]
        'label': torch.stack([b['label'] for b in batch]),
        'lengths': lengths,
        'mask': mask,
        'utt_id': [b['utt_id'] for b in batch],
    }
