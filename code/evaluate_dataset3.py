"""The design on dataset 3 (Welter, Sokolovskii et al., Zenodo 10943281; GenScript,
MinION Kit 14, Guppy). Reuses the dataset-2 estimation (channel_fit_dataset2) and evaluation (evaluate_dataset2);
only the loader, output folder, strand length and window offset differ.
  python -X utf8 evaluate_dataset3.py --stage describe|align|fit|eval|summary [--budget 240]
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import channel_model as ch  # noqa: E402
import evaluate_dataset1 as ev  # noqa: E402
import dataset_loaders as dat  # noqa: E402
import evaluate_dataset2 as e4  # noqa: E402
import channel_fit_dataset2 as m4  # noqa: E402

DL = HERE.parent / 'downloads' / 'welter_sokolovskii_nanopore' / 'extracted'
OUT = HERE.parent / 'results' / 'dataset3'
SUB = None          # chosen sub-archive folder, set in CONFIG below after describe
CONFIG = HERE.parent / 'results' / 'dataset3_config.txt'


def _read_cfg():
    kv = dict(l.split('=', 1) for l in CONFIG.read_text().split('\n') if '=' in l)
    return {k.strip(): v.strip() for k, v in kv.items()}


def load_third():
    cfg = _read_cfg()
    tx, rx = DL / cfg['tx'], DL / cfg['rx']
    centers = [l.strip() for l in tx.read_text(encoding='ascii').split('\n') if l.strip() and not l.startswith('>')]
    clusters, cur = [], None
    with open(rx, encoding='ascii') as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            if set(s) == {'='}:
                if cur is not None:
                    clusters.append(cur)
                cur = []
            else:
                if cur is None:
                    cur = []
                cur.append(s)
    if cur is not None:
        clusters.append(cur)
    return centers, clusters


def reads_for(source, n, centers, clusters):
    """Same as evaluate_dataset2.reads_for, except that empty clusters (549 here) are not drawn."""
    half = 1 if source == 'real_heldout' else 0
    rng = np.random.default_rng([929, half])
    cl_ids = [ci for ci in range(half, len(centers), 2) if len(centers[ci]) == e4.LD and clusters[ci]]
    out = []
    while len(out) < n:
        ci = int(rng.choice(cl_ids)); ri = int(rng.integers(len(clusters[ci])))
        y = [ev.B2I[c] for c in clusters[ci][ri]]
        if abs(len(y) - e4.LD) <= ch.D:
            out.append((ci, y))
    return out


def patch():
    cfg = _read_cfg()
    e4.reads_for = reads_for
    dat.load_dnaformer = load_third
    OUT.mkdir(parents=True, exist_ok=True)
    m4.OUT = OUT
    e4.OUT = OUT
    e4.LD = int(cfg['ld'])
    e4.OFFSET = int(cfg.get('offset', 0))
    return cfg


def summary(M):
    ev.OUT = OUT
    i1 = int(np.where(ev.TAUS == 1.0)[0][0])
    LN2 = math.log(2)
    res = {}
    for source, dec, n in e4.CONFIGS:
        nm = f'{source}__{dec.replace(":", "-")}'
        e = ev.load_eval(nm)
        if source == 'real_design':
            fit = np.arange(len(e['rank'])) % 2 == 0
            it = int(np.argmin((-e['lq_tau'][fit]).mean(0)))
            evd = {k: (v[~fit] if k != 't_per_read' else v) for k, v in e.items()}
        else:
            ed = ev.load_eval(nm.replace('real_heldout', 'real_design'))
            it = int(np.argmin((-ed['lq_tau']).mean(0)))
            evd = e
        u, c = ev.metrics(evd, i1, M), ev.metrics(evd, it, M)
        res[nm] = (evd, it)
        print(f'{nm}: n={len(evd["rank"])} eps={u["eps"]:.4f} hard={u["hard"]:.3f} | uncal soft={u["soft"]:.3f} '
              f'ECE={u["ece"]:.4f} | tau*={ev.TAUS[it]} soft={c["soft"]:.3f} ECE={c["ece"]:.4f} | '
              f'{1e3 * e["t_per_read"].mean():.1f} ms/read', flush=True)
    (eb, itb), (ed_, _) = res['real_heldout__marg-iid'], res['real_heldout__joint-hmm2x']
    d = (-ed_['lq_tau'][:, i1] + eb['lq_tau'][:, itb]) / LN2
    print(f'held-out advantage (baseline tau {ev.TAUS[itb]} soft - design tau 1 soft): {-d.mean():.3f} +- {d.std() / math.sqrt(len(d)):.3f}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='describe')
    ap.add_argument('--budget', type=float, default=240)
    a = ap.parse_args()
    if a.stage == 'describe':
        c, cl = load_third()
        print(dat.describe(c, cl))
        print('done describe')
    else:
        cfg = patch()
        if a.stage == 'align':
            m4.stage_align(a.budget)
        elif a.stage == 'fit':
            m4.stage_fit()
        elif a.stage == 'eval':
            e4.run(a.budget)
        else:
            summary(int(cfg['m']))
