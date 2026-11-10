"""Utterance-level feature extraction with frozen pre-trained encoders.

text  : BERT (bert-base-uncased), [CLS] token of the last layer
e2v   : emotion2vec base (self-supervised, no emotion labels), utterance embedding
wavlm : WavLM Base+, mean over frames of the last layer

Do not use emotion2vec_plus models here: they are fine-tuned on labelled emotion data.
"""

import os
import sys
from contextlib import contextmanager

import numpy as np
import torch

TEXT_MODEL_ID = 'bert-base-uncased'
EMOTION2VEC_MODEL_ID = 'iic/emotion2vec_base'
WAVLM_MODEL_ID = 'microsoft/wavlm-base-plus'
FEATURE_DIM = 768
MODES = ('text', 'e2v', 'wavlm')


@contextmanager
def _quiet_stderr():
    stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    try:
        yield
    finally:
        sys.stderr.close()
        sys.stderr = stderr


def load_extractor(mode, device):
    """Load one encoder. Encoders are loaded one at a time to keep GPU memory low."""
    if mode == 'text':
        from transformers import AutoModel, AutoTokenizer
        return {'tokenizer': AutoTokenizer.from_pretrained(TEXT_MODEL_ID),
                'model': AutoModel.from_pretrained(TEXT_MODEL_ID).to(device).eval()}
    if mode == 'e2v':
        from funasr import AutoModel
        return {'model': AutoModel(model=EMOTION2VEC_MODEL_ID)}
    if mode == 'wavlm':
        from transformers import AutoFeatureExtractor, AutoModel
        return {'processor': AutoFeatureExtractor.from_pretrained(WAVLM_MODEL_ID),
                'model': AutoModel.from_pretrained(WAVLM_MODEL_ID).to(device).eval()}
    raise ValueError(f'mode must be one of {MODES}')


@torch.no_grad()
def extract_one(extractor, mode, utterance, device):
    """Feature vector [768] for one utterance dict (needs 'text' and 'audio_path')."""
    if mode == 'text':
        inputs = extractor['tokenizer'](utterance['text'], return_tensors='pt', truncation=True,
                                        max_length=512).to(device)
        return extractor['model'](**inputs).last_hidden_state[0, 0].cpu().numpy()

    if mode == 'e2v':
        with _quiet_stderr():
            res = extractor['model'].generate(input=utterance['audio_path'], granularity='utterance',
                                              extract_embedding=True)
        return np.asarray(res[0]['feats'], dtype=np.float32).reshape(-1)

    import torchaudio
    waveform, sr = torchaudio.load(utterance['audio_path'])
    waveform = waveform.mean(dim=0)                       # mono
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    inputs = extractor['processor'](waveform.numpy(), sampling_rate=16000, return_tensors='pt').to(device)
    return extractor['model'](**inputs).last_hidden_state[0].mean(dim=0).cpu().numpy()


def extract_features(utterances, device, modes=MODES):
    """Return {mode: array [N, 768]} for a list of utterance dicts.

    Failures raise an error instead of silently returning zero vectors.
    """
    out = {}
    for mode in modes:
        extractor = load_extractor(mode, device)
        feats = [extract_one(extractor, mode, u, device) for u in utterances]
        out[mode] = np.stack(feats).astype(np.float32)
        assert out[mode].shape[1] == FEATURE_DIM, f'{mode}: unexpected dim {out[mode].shape[1]}'
        del extractor
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return out


def save_cache(cache_dir, session, utt_ids, features):
    """Write features in the cache layout that dataset.FeatureStore reads."""
    os.makedirs(cache_dir, exist_ok=True)
    np.save(os.path.join(cache_dir, f'{session}_utt_ids.npy'), np.array(utt_ids))
    np.save(os.path.join(cache_dir, f'{session}_F_t.npy'), features['text'])
    np.save(os.path.join(cache_dir, f'{session}_F_s_e2v.npy'), features['e2v'])
    np.save(os.path.join(cache_dir, f'{session}_F_s_wavlm.npy'), features['wavlm'])
