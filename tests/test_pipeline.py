"""Checks for data construction, splits, padding and training.

Runs without the IEMOCAP release: a small fake corpus is written in the IEMOCAP file layout.
Utterances are shuffled in the files and the two speakers alternate in time, so the tests
catch wrong ordering. Run with:  python -m pytest tests -q
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dataset import (EMO_MAP, ERCDataset, FeatureStore, audit_cache, build_samples,  # noqa: E402
                     load_corpus, split_dialogues)
from model import EmotionModel  # noqa: E402
from trainer import DEFAULT_CONFIG, SETTINGS, run_fold, summarize  # noqa: E402
from utils.collate import collate_epc  # noqa: E402

LABELS = ['neu', 'hap', 'exc', 'ang', 'sad', 'fru', 'sur', 'fea', 'dis', 'oth', 'xxx']
N_DIALOGUES, TURNS = 5, 10          # per session; each turn = one F and one M utterance


@pytest.fixture(scope='module')
def fake_data(tmp_path_factory):
    root = tmp_path_factory.mktemp('iemocap')
    cache = tmp_path_factory.mktemp('cache')
    rng = np.random.RandomState(0)
    for s in range(1, 6):
        session = f'Session{s}'
        emo_dir = root / session / 'dialog' / 'EmoEvaluation'
        trans_dir = root / session / 'dialog' / 'transcriptions'
        emo_dir.mkdir(parents=True)
        trans_dir.mkdir(parents=True)
        cached_ids = []
        for d in range(N_DIALOGUES):
            dialogue = f'Ses0{s}F_impro{d:02d}'
            lines, trans = [], []
            for t in range(TURNS):
                for k, spk in enumerate('FM'):
                    utt_id = f'{dialogue}_{spk}{t:03d}'
                    start = 10.0 * t + 5.0 * k            # F then M within each turn
                    label = LABELS[rng.randint(len(LABELS))]
                    lines.append(f'[{start:.4f} - {start + 4:.4f}]\t{utt_id}\t{label}\t[2.5, 2.5, 2.5]\n'
                                 f'C-E2:\tNeutral;\t()\n\n')
                    trans.append(f'{utt_id} [{start:08.4f}-{start + 4:08.4f}]: words {utt_id}\n')
                    if utt_id != f'{dialogue}_M009':      # one utterance missing from the cache
                        cached_ids.append(utt_id)
            rng.shuffle(lines)
            rng.shuffle(trans)
            (emo_dir / f'{dialogue}.txt').write_text('% header line\n\n' + ''.join(lines))
            (trans_dir / f'{dialogue}.txt').write_text(''.join(trans))
            (emo_dir / f'._{dialogue}.txt').write_bytes(b'\x00\x05junk')   # macOS metadata file
        n = len(cached_ids)
        np.save(cache / f'{session}_utt_ids.npy', np.array(cached_ids))
        for name in ('F_t', 'F_s_e2v', 'F_s_wavlm'):
            np.save(cache / f'{session}_{name}.npy', rng.randn(n, 768).astype(np.float32))
    corpus, dropped = load_corpus(str(root))
    return corpus, dropped, str(cache)


def test_corpus_is_chronological(fake_data):
    corpus, dropped, _ = fake_data
    assert len(corpus) == 5 * N_DIALOGUES and dropped == []
    for dialogue, entry in corpus.items():
        ids = [u['utt_id'] for u in entry['utterances']]
        assert len(ids) == 2 * TURNS
        # Speakers alternate: F000, M000, F001, M001, ...
        assert ids[:4] == [f'{dialogue}_F000', f'{dialogue}_M000', f'{dialogue}_F001', f'{dialogue}_M001']
        starts = [u['start'] for u in entry['utterances']]
        assert starts == sorted(starts)


def test_samples_stay_inside_dialogue(fake_data):
    corpus, _, _ = fake_data
    dialogues = sorted(corpus)
    for history_len in (0, 3):
        samples = build_samples(corpus, dialogues, history_len)
        n_targets = sum(u['label'] in EMO_MAP for e in corpus.values() for u in e['utterances'])
        assert len(samples) == n_targets
        for s in samples:
            assert len({i.rsplit('_', 1)[0] for i in s['ids']}) == 1   # one dialogue only
            assert s['ids'][-1] == s['utt_id']
            assert 1 <= len(s['ids']) <= history_len + 1
            utts = corpus[s['utt_id'].rsplit('_', 1)[0]]['utterances']
            j = [u['utt_id'] for u in utts].index(s['utt_id'])
            assert len(s['ids']) == min(j, history_len) + 1              # first utterance -> length 1


def test_splits(fake_data):
    corpus, _, _ = fake_data
    for test_session in ('Session1', 'Session5'):
        split = split_dialogues(corpus, test_session, seed=0)
        sets = {k: set(v) for k, v in split.items()}
        assert not (sets['train'] & sets['val'] or sets['train'] & sets['test'] or sets['val'] & sets['test'])
        assert sets['train'] | sets['val'] | sets['test'] == set(corpus)
        assert all(corpus[d]['session'] == test_session for d in sets['test'])
        assert all(corpus[d]['session'] != test_session for d in sets['train'] | sets['val'])
        assert split == split_dialogues(corpus, test_session, seed=0)     # reproducible


def test_audit_and_missing_features(fake_data):
    corpus, dropped, cache = fake_data
    store = FeatureStore([cache], 'e2v')
    report = audit_cache(corpus, dropped, store)
    assert [m[0] for m in report['missing']] == sorted(f'Ses0{s}F_impro{d:02d}_M009'
                                                       for s in range(1, 6) for d in range(N_DIALOGUES))
    samples = build_samples(corpus, sorted(corpus), 3)
    with pytest.raises(KeyError):
        ERCDataset(samples, store)


@pytest.mark.parametrize('fusion', ['text', 'speech', 'gated', 'fixed'])
def test_padding_does_not_change_outputs(fusion):
    torch.manual_seed(0)
    model = EmotionModel(fusion).eval()
    seqs = [(torch.randn(L, 768), torch.randn(L, 768)) for L in (1, 4, 2, 3)]
    batch = collate_epc([{'F_t': t, 'F_s': s, 'label': torch.tensor(0), 'utt_id': str(i)}
                         for i, (t, s) in enumerate(seqs)])
    with torch.no_grad():
        batched, _ = model(batch['F_t'], batch['F_s'], batch['lengths'])
        for i, (t, s) in enumerate(seqs):
            single, _ = model(t[None], s[None], torch.tensor([len(t)]))
            assert torch.allclose(single[0], batched[i], atol=1e-5)


def test_fixed_fusion_has_no_gate():
    names = [n for n, _ in EmotionModel('fixed').named_parameters()]
    assert not any(n.startswith('gate') for n in names)
    assert any(n.startswith('gate') for n, _ in EmotionModel('gated').named_parameters())


def test_one_fold_runs(fake_data):
    corpus, _, cache = fake_data
    store = FeatureStore([cache], 'e2v')
    # Give the missing utterances features so every context is complete
    for u in store.missing([u['utt_id'] for e in corpus.values() for u in e['utterances']]):
        store.text[u] = store.speech[u] = np.zeros(768, dtype=np.float32)
    cfg = {**DEFAULT_CONFIG, **SETTINGS['Gated Fusion (E2V)'], 'max_epochs': 3, 'patience': 2}
    summary, preds = run_fold(corpus, store, 'Session1', cfg, seed=0, device=torch.device('cpu'))
    assert summary['n_test'] == len(preds) > 0
    assert set(preds['utt_id']).isdisjoint(
        {s['utt_id'] for s in build_samples(corpus, split_dialogues(corpus, 'Session1')['train'], 3)})
    assert 1 <= summary['best_epoch'] <= summary['epochs_run'] <= 3
    assert {'gate_mean', 'gate_target'} <= set(preds.columns)
    # Same seed, same result
    again, _ = run_fold(corpus, store, 'Session1', cfg, seed=0, device=torch.device('cpu'))
    assert again['test_macro_f1'] == summary['test_macro_f1']
    folds = pd.DataFrame([{'setting': 'Gated Fusion (E2V)', **summary}])
    assert summarize(folds).loc['Gated Fusion (E2V)', 'n_seeds'] == 1
