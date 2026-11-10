"""IEMOCAP loading for emotion recognition in conversation (ERC).

Each sample is one target utterance with one of the four target emotions. The input is the
target utterance plus up to `history_len` preceding utterances of the same dialogue, in
chronological order. Context utterances can have any label.
"""

import os
import re
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset

SESSIONS = [f'Session{i}' for i in range(1, 6)]

# Four-class setting: excited is merged into happy
EMO_MAP = {'hap': 0, 'exc': 0, 'ang': 1, 'sad': 2, 'neu': 3}
CLASS_NAMES = ['Happy', 'Angry', 'Sad', 'Neutral']

# "[6.2901 - 8.2357]	Ses01F_impro01_F000	neu	[2.5000, 2.5000, 2.5000]"
EMO_LINE = re.compile(r'^\[(\d+\.\d+)\s*-\s*(\d+\.\d+)\]\s+(Ses\S+)\s+(\w+)\s+\[', re.M)
# "Ses01F_impro01_F000 [006.2901-008.2357]: Excuse me."
TRANS_LINE = re.compile(r'^(Ses\S+)\s*\[[\d.]+-[\d.]+\]:\s*(.*)$', re.M)


def _read(path):
    try:
        with open(path, encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, encoding='latin-1') as f:
            return f.read()


def _txt_files(directory):
    # Skip macOS metadata files ("._*") that some IEMOCAP copies contain
    return sorted(f for f in os.listdir(directory) if f.endswith('.txt') and not f.startswith('.'))


def dialogue_of(utt_id):
    return utt_id.rsplit('_', 1)[0]


def load_corpus(data_root, sessions=SESSIONS):
    """Return {dialogue_id: {'session': str, 'utterances': [dict, ...]}}, sorted by start time.

    The utterance list comes from the EmoEvaluation files and keeps every label.
    Utterances without a transcript are left out and reported in `dropped`.
    """
    corpus, dropped = {}, []
    for session in sessions:
        dialog_dir = os.path.join(data_root, session, 'dialog')

        texts = {}
        trans_dir = os.path.join(dialog_dir, 'transcriptions')
        for name in _txt_files(trans_dir):
            for utt_id, text in TRANS_LINE.findall(_read(os.path.join(trans_dir, name))):
                texts[utt_id] = text.strip()

        emo_dir = os.path.join(dialog_dir, 'EmoEvaluation')
        for name in _txt_files(emo_dir):
            for start, end, utt_id, label in EMO_LINE.findall(_read(os.path.join(emo_dir, name))):
                if utt_id not in texts:
                    dropped.append(utt_id)
                    continue
                dialogue = dialogue_of(utt_id)
                entry = corpus.setdefault(dialogue, {'session': session, 'utterances': []})
                entry['utterances'].append({
                    'utt_id': utt_id,
                    'label': label.lower(),
                    'start': float(start),
                    'end': float(end),
                    'text': texts[utt_id],
                    'audio_path': os.path.join(data_root, session, 'sentences', 'wav', dialogue,
                                               f'{utt_id}.wav'),
                })

    for entry in corpus.values():
        # Timestamps restart in every dialogue, so sort within each dialogue only
        entry['utterances'].sort(key=lambda u: (u['start'], u['utt_id']))
    return corpus, dropped


def split_dialogues(corpus, test_session, val_fraction=0.2, seed=0):
    """Split dialogues into train / val / test for one leave-one-session-out fold.

    Validation dialogues are held out from every training session. The split depends only on
    `test_session` and `seed`, so all models and training seeds share it.
    """
    by_session = defaultdict(list)
    for dialogue, entry in corpus.items():
        by_session[entry['session']].append(dialogue)

    rng = np.random.RandomState(seed)
    splits = {'train': [], 'val': [], 'test': sorted(by_session[test_session])}
    for session in sorted(by_session):
        if session == test_session:
            continue
        dialogues = sorted(by_session[session])
        rng.shuffle(dialogues)
        n_val = max(1, round(val_fraction * len(dialogues)))
        splits['val'] += sorted(dialogues[:n_val])
        splits['train'] += sorted(dialogues[n_val:])
    return splits


def build_samples(corpus, dialogues, history_len):
    """One sample per target-emotion utterance: (context ids + target id, label)."""
    samples = []
    for dialogue in dialogues:
        utterances = corpus[dialogue]['utterances']
        for j, utt in enumerate(utterances):
            if utt['label'] not in EMO_MAP:
                continue
            context = utterances[max(0, j - history_len):j]
            samples.append({
                'utt_id': utt['utt_id'],
                'ids': [u['utt_id'] for u in context] + [utt['utt_id']],
                'label': EMO_MAP[utt['label']],
            })
    return samples


class FeatureStore:
    """Cached utterance features from one or more cache directories.

    Each directory holds, per session: {session}_utt_ids.npy, {session}_F_t.npy and
    {session}_F_s_{speech_feature}.npy. Later directories override earlier ones.
    """

    def __init__(self, cache_dirs, speech_feature):
        if speech_feature not in ('e2v', 'wavlm'):
            raise ValueError("speech_feature must be 'e2v' or 'wavlm'")
        self.speech_feature = speech_feature
        self.text, self.speech = {}, {}
        for cache_dir in cache_dirs:
            for session in SESSIONS:
                ids_path = os.path.join(cache_dir, f'{session}_utt_ids.npy')
                if not os.path.exists(ids_path):
                    continue
                ids = np.load(ids_path, allow_pickle=True)
                F_t = np.load(os.path.join(cache_dir, f'{session}_F_t.npy'))
                F_s = np.load(os.path.join(cache_dir, f'{session}_F_s_{speech_feature}.npy'))
                assert len(ids) == len(F_t) == len(F_s), f'size mismatch in {cache_dir}/{session}'
                for i, utt_id in enumerate(ids):
                    self.text[str(utt_id)] = F_t[i]
                    self.speech[str(utt_id)] = F_s[i]

    def missing(self, utt_ids):
        return sorted(u for u in set(utt_ids) if u not in self.text)

    def get(self, utt_ids):
        return (np.stack([self.text[u] for u in utt_ids]),
                np.stack([self.speech[u] for u in utt_ids]))


def audit_cache(corpus, dropped, store):
    """Report utterances that the cache does not cover. Returns a dict of lists."""
    all_utts = [u for entry in corpus.values() for u in entry['utterances']]
    missing = set(store.missing([u['utt_id'] for u in all_utts]))
    return {
        'n_utterances': len(all_utts),
        'n_targets': sum(u['label'] in EMO_MAP for u in all_utts),
        'missing': [(u['utt_id'], u['label']) for u in all_utts if u['utt_id'] in missing],
        'no_transcript': dropped,
    }


class ERCDataset(Dataset):
    def __init__(self, samples, store):
        missing = store.missing([u for s in samples for u in s['ids']])
        if missing:
            raise KeyError(f'{len(missing)} utterances have no cached features, e.g. {missing[:5]}')
        self.samples = samples
        self.store = store

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        F_t, F_s = self.store.get(sample['ids'])
        return {
            'F_t': torch.from_numpy(F_t).float(),
            'F_s': torch.from_numpy(F_s).float(),
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'utt_id': sample['utt_id'],
        }
