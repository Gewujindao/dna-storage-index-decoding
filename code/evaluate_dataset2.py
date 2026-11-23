"""The design on dataset 2 (DNAformer pilot nanopore). Same statistics and decoders as evaluate_dataset1;
only the dataset loader, strand length 140 and index window offset differ.
Runs all configurations in sequence, resumable, stops after --budget seconds.
  python -X utf8 evaluate_dataset2.py --budget 240
  python -X utf8 evaluate_dataset2.py --summary
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import channel_model as ch  # noqa: E402
import evaluate_dataset1 as ev  # noqa: E402
import dataset_loaders as dat  # noqa: E402

OUT = HERE.parent / 'results' / 'dataset2'
NC, OFFSET, LD = 10, 60, 140          # window fixed on the design half (positions 0-9 had near-zero error)
B2I = ev.B2I
CHUNK = 100
CONFIGS = [('real_design', 'marg:iid', 1000), ('real_design', 'joint:hmm2x', 1000),
           ('real_heldout', 'marg:iid', 2000), ('real_heldout', 'joint:hmm2x', 2000)]


def models():
    m = np.load(OUT / 'measure.npz')
    r = m['rates']
    pd = r[[2, 5, 8]].sum(); pi = r[3:].sum(); ps = r[[1, 4, 7]].sum() / max(1 - pd, 1e-9)
    sm = m['subm'] / m['subm'].sum(1, keepdims=True)
    isame = float(m['ins_eq'][1] / m['ins_eq'][0])
    h2 = np.load(OUT / 'hmm_S2.npz')
    return {'iid': ch.Model.iid(pi, pd, ps), 'hmm2x': ch.Model(h2['pi0'], h2['A'], h2['E'], SM=sm, ins_same=isame)}


def reads_for(source, n, centers, clusters):
    half = 1 if source == 'real_heldout' else 0
    rng = np.random.default_rng([929, half])
    cl_ids = [ci for ci in range(half, len(centers), 2) if len(centers[ci]) == LD]
    out = []
    while len(out) < n:                      # clusters are large: draw cluster, then read
        ci = int(rng.choice(cl_ids)); ri = int(rng.integers(len(clusters[ci])))
        y = [B2I[c] for c in clusters[ci][ri]]
        if abs(len(y) - LD) <= ch.D:
            out.append((ci, y))
    return out


def run(budget):
    ch.L = LD
    centers, clusters = dat.load_dnaformer()
    mods = models()
    C = np.array([[B2I[c] for c in s[OFFSET:OFFSET + NC]] for s in centers], dtype=np.int64)
    trie = ch.Trie(C)
    t0 = time.perf_counter()
    for source, dec, n in CONFIGS:
        name = f'{source}__{dec.replace(":", "-")}'
        od = OUT / 'eval' / name
        od.mkdir(parents=True, exist_ok=True)
        items = None
        kind, mname = dec.split(':')
        model = mods[mname]
        for ci in range(n // CHUNK):
            f = od / f'chunk_{ci:03d}.npz'
            if f.exists():
                continue
            if time.perf_counter() - t0 > budget:
                print('budget reached; rerun to continue', flush=True)
                return
            if items is None:
                items = reads_for(source, n, centers, clusters)
            part = items[ci * CHUNK:(ci + 1) * CHUNK]
            reads, true = [y for _, y in part], np.array([c for c, _ in part])
            tb = time.perf_counter()
            res = []
            if kind == 'marg':
                lp = ch.marginal_logapp(model, reads, NC, offset=OFFSET)
                for b in range(len(reads)):
                    s = lp[b, 0][C[:, 0]].copy()
                    for t in range(1, NC):
                        s += lp[b, t][C[:, t]]
                    s -= s.max() + math.log(np.exp(s - s.max()).sum())
                    res.append(ev.stats(s, true[b]))
            else:
                Y, lens = ch.pad(reads)
                beta = ch.backward(model, Y, lens, t_stop=OFFSET + NC)[:, OFFSET + NC]
                for b in range(len(reads)):
                    res.append(ev.stats(ch.joint_logpost_trie(model, Y[b:b + 1], lens[b:b + 1], beta[b], trie,
                                                              offset=OFFSET), true[b]))
            tper = (time.perf_counter() - tb) / len(reads)
            np.savez(f, lq_tau=np.stack([r[0] for r in res]), conf_tau=np.stack([r[1] for r in res]),
                     rank=np.array([r[2] for r in res]), logS=np.stack([r[3] for r in res]), t_per_read=tper)
            print(f'{name} chunk {ci}: {tper * 1e3:.1f} ms/read', flush=True)
    print('done eval4', flush=True)


def summary():
    ev.OUT = OUT
    M = 1000
    i1 = int(np.where(ev.TAUS == 1.0)[0][0])
    for source, dec, n in CONFIGS:
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
        print(f'{nm}: n={len(evd["rank"])} eps={u["eps"]:.4f} hard={u["hard"]:.3f} | uncal soft={u["soft"]:.3f} '
              f'ECE={u["ece"]:.4f} | tau*={ev.TAUS[it]} soft={c["soft"]:.3f} ECE={c["ece"]:.4f} | '
              f'{1e3 * e["t_per_read"].mean():.1f} ms/read', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--budget', type=float, default=240)
    ap.add_argument('--summary', action='store_true')
    a = ap.parse_args()
    summary() if a.summary else run(a.budget)
