"""Training and leave-one-session-out evaluation.

Protocol per fold: train on the training dialogues, pick the epoch with the best validation
macro-F1 (early stopping), then evaluate that model once on the held-out test session.
"""

import copy
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.utils.data import DataLoader

from dataset import SESSIONS, ERCDataset, build_samples, split_dialogues
from model import EmotionModel
from utils.collate import collate_epc

# Fixed before the final runs; not tuned on test results
DEFAULT_CONFIG = {
    'history_len': 3,
    'lambda_balance': 5e-4,   # weight of the (gate - 0.5)^2 penalty, gated fusion only
    'hidden_size': 256,
    'num_classes': 4,
    'batch_size': 32,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'max_epochs': 50,
    'patience': 10,
    'val_fraction': 0.2,
    'split_seed': 0,
}

# The eight settings of the study. Unlisted keys come from DEFAULT_CONFIG.
SETTINGS = {
    'Text-Only':                {'fusion': 'text',   'speech_feature': 'e2v'},
    'Speech-Only (E2V)':        {'fusion': 'speech', 'speech_feature': 'e2v'},
    'Speech-Only (WavLM)':      {'fusion': 'speech', 'speech_feature': 'wavlm'},
    'Gated Fusion (E2V)':       {'fusion': 'gated',  'speech_feature': 'e2v'},
    'Gated Fusion (WavLM)':     {'fusion': 'gated',  'speech_feature': 'wavlm'},
    'Fixed Fusion (E2V)':       {'fusion': 'fixed',  'speech_feature': 'e2v'},
    'Gated Fusion (E2V), current utterance only': {'fusion': 'gated', 'speech_feature': 'e2v',
                                                   'history_len': 0},
    'Gated Fusion (E2V), no gate penalty':        {'fusion': 'gated', 'speech_feature': 'e2v',
                                                   'lambda_balance': 0.0},
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def masked_mean(values, mask):
    """Mean of values [B, L, D] over the real (unpadded) positions."""
    m = mask.unsqueeze(-1).float()
    return (values * m).sum() / (m.sum() * values.shape[-1])


def train_epoch(model, loader, optimizer, lambda_balance, device):
    model.train()
    for batch in loader:
        mask = batch['mask'].to(device)
        logits, gate = model(batch['F_t'].to(device), batch['F_s'].to(device), batch['lengths'])
        loss = F.cross_entropy(logits, batch['label'].to(device))
        if gate is not None and lambda_balance > 0:
            loss = loss + lambda_balance * masked_mean((gate - 0.5) ** 2, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


@torch.no_grad()
def evaluate(model, loader, device):
    """Metrics plus per-sample predictions. Gate statistics use real positions only."""
    model.eval()
    rows = []
    for batch in loader:
        lengths = batch['lengths']
        logits, gate = model(batch['F_t'].to(device), batch['F_s'].to(device), lengths)
        preds = logits.argmax(dim=1).cpu().numpy()
        for i, utt_id in enumerate(batch['utt_id']):
            row = {'utt_id': utt_id, 'label': int(batch['label'][i]), 'pred': int(preds[i])}
            if gate is not None:
                g = gate[i, :lengths[i]]                       # [L_i, D], text weight
                row['gate_mean'] = g.mean().item()             # over positions and dimensions
                row['gate_target'] = g[-1].mean().item()       # target utterance only
                row['gate_dim_std'] = g[-1].std().item()       # spread across dimensions
                row['gate_time_std'] = g.std(dim=0).mean().item() if len(g) > 1 else float('nan')
            rows.append(row)

    df = pd.DataFrame(rows)
    metrics = {
        'macro_f1': f1_score(df['label'], df['pred'], average='macro', zero_division=0),
        'uar': recall_score(df['label'], df['pred'], average='macro', zero_division=0),
        'accuracy': accuracy_score(df['label'], df['pred']),
    }
    return metrics, df


def make_loader(samples, store, batch_size, shuffle, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(ERCDataset(samples, store), batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_epc, generator=generator)


def run_fold(corpus, store, test_session, cfg, seed, device, verbose=False):
    """Train one model for one fold. Returns (summary dict, per-sample test predictions)."""
    split = split_dialogues(corpus, test_session, cfg['val_fraction'], cfg['split_seed'])
    samples = {k: build_samples(corpus, v, cfg['history_len']) for k, v in split.items()}

    set_seed(seed)
    loaders = {
        'train': make_loader(samples['train'], store, cfg['batch_size'], True, seed),
        'val': make_loader(samples['val'], store, cfg['batch_size'], False, seed),
        'test': make_loader(samples['test'], store, cfg['batch_size'], False, seed),
    }
    model = EmotionModel(cfg['fusion'], hidden_size=cfg['hidden_size'],
                         num_classes=cfg['num_classes']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'],
                                  weight_decay=cfg['weight_decay'])

    start = time.time()
    best_f1, best_epoch, best_state, waited = -1.0, 0, None, 0
    for epoch in range(1, cfg['max_epochs'] + 1):
        train_epoch(model, loaders['train'], optimizer, cfg['lambda_balance'], device)
        val_f1 = evaluate(model, loaders['val'], device)[0]['macro_f1']
        if verbose:
            print(f'  epoch {epoch:2d}: val macro-F1 {val_f1:.4f}')
        if val_f1 > best_f1:
            best_f1, best_epoch, best_state, waited = val_f1, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            waited += 1
            if waited >= cfg['patience']:
                break
    train_time = time.time() - start

    # The test session is used exactly once, with the selected model
    model.load_state_dict(best_state)
    test_metrics, predictions = evaluate(model, loaders['test'], device)

    summary = {
        'test_session': test_session, 'seed': seed, 'best_epoch': best_epoch,
        'epochs_run': epoch, 'val_macro_f1': best_f1, 'train_time_s': train_time,
        'n_train': len(samples['train']), 'n_val': len(samples['val']), 'n_test': len(samples['test']),
        **{f'test_{k}': v for k, v in test_metrics.items()},
    }
    predictions.insert(0, 'seed', seed)
    predictions.insert(0, 'test_session', test_session)
    return summary, predictions


def run_setting(name, corpus, stores, seeds, device, sessions=SESSIONS, verbose=False):
    """All folds and seeds of one setting. `stores` maps 'e2v' / 'wavlm' to a FeatureStore."""
    cfg = {**DEFAULT_CONFIG, **SETTINGS[name]}
    store = stores[cfg['speech_feature']]
    folds, predictions = [], []
    for seed in seeds:
        for session in sessions:
            summary, preds = run_fold(corpus, store, session, cfg, seed, device, verbose)
            folds.append({'setting': name, **summary})
            predictions.append(preds.assign(setting=name))
            print(f"{name} | seed {seed} | test {session}: macro-F1 {summary['test_macro_f1']:.4f} "
                  f"(best epoch {summary['best_epoch']}, {summary['train_time_s']:.0f}s)")
    return pd.DataFrame(folds), pd.concat(predictions, ignore_index=True)


def summarize(folds):
    """Mean over the five folds for each seed, then mean and std across seeds."""
    metrics = ['test_macro_f1', 'test_uar', 'test_accuracy']
    per_seed = folds.groupby(['setting', 'seed'])[metrics].mean()
    summary = per_seed.groupby('setting').agg(['mean', 'std'])
    summary.columns = [f'{m}_{s}' for m, s in summary.columns]
    summary['n_seeds'] = per_seed.groupby('setting').size()
    return summary.reindex([s for s in SETTINGS if s in summary.index])
