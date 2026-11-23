"""Index-posterior evaluation on real or simulated reads (resumable chunks, short runs).

  python -X utf8 evaluate_dataset1.py --source real_design --decoder marg:iid --n 1500 --budget 25
  python -X utf8 evaluate_dataset1.py --summary NAME [NAME ...] [--tau_from NAME]

Sources: real_design (even clusters), real_heldout (odd clusters), sim_<variant>_design with variant in
  iid, burst, subst, pos, homo, all, fit2, fit3 (reads simulated from the design-half strands).
Decoders: marg:<model>, joint:<model> with model in iid, hmm2, hmm3, hmm2s (hmm2 + substitution matrix).
Index set: first NC = 10 symbols of all 10,000 design strands (enumerated, M = 10,000).
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import real_reads_dataset1 as rr  # noqa: E402
import channel_model as ch  # noqa: E402

OUT = HERE.parent / 'results' / 'dataset1'
NC = 10
TAUS = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2])
B2I = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
CHUNK = 100


def models():
    m = np.load(OUT / 'measure.npz')
    r = m['rates']                                             # event frequencies (m, kind)
    pd = r[[2, 5, 8]].sum()
    pi = r[3:].sum()                                           # share of positions with >= 1 insertion
    ps = r[[1, 4, 7]].sum() / max(1 - pd, 1e-9)
    sm = m['subm'] / m['subm'].sum(1, keepdims=True)
    prof = m['prof'] / m['prof'].mean()
    ch_h, ch_n = m['ctx_homo'], m['ctx_non']
    eh = (ch_h[1:].sum() / ch_h[0]) / (ch_n[1:].sum() / ch_n[0])
    out = {'iid': ch.Model.iid(pi, pd, ps)}
    for S in (2, 3):
        h = np.load(OUT / f'hmm_S{S}.npz')
        out[f'hmm{S}'] = ch.Model(h['pi0'], h['A'], h['E'])
    h2 = np.load(OUT / 'hmm_S2.npz')
    h3 = np.load(OUT / 'hmm_S3.npz')
    isame = float(m['ins_eq'][1] / m['ins_eq'][0])
    out['hmm2s'] = ch.Model(h2['pi0'], h2['A'], h2['E'], SM=sm)
    out['iidx'] = ch.Model(*_iid_parts(pi, pd, ps), SM=sm, ins_same=isame)
    out['hmm2x'] = ch.Model(h2['pi0'], h2['A'], h2['E'], SM=sm, ins_same=isame)
    out['hmm3x'] = ch.Model(h3['pi0'], h3['A'], h3['E'], SM=sm, ins_same=isame)
    sims = {
        'iid': ch.Model.iid(pi, pd, ps),
        'burst': ch.Model(h2['pi0'], h2['A'], h2['E']),
        'subst': ch.Model.iid(pi, pd, ps, SM=sm),
        'insdup': ch.Model(*_iid_parts(pi, pd, ps), ins_same=isame),
        'pos': ch.Model(*_iid_parts(pi, pd, ps), pos_scale=prof),
        'homo': ch.Model(*_iid_parts(pi, pd, ps), homo_scale=(1.0, eh)),
        'all': ch.Model(h2['pi0'], h2['A'], h2['E'], SM=sm, pos_scale=prof, homo_scale=(1.0, eh), ins_same=isame),
        'fit2x': ch.Model(h2['pi0'], h2['A'], h2['E'], SM=sm, ins_same=isame),
    }
    return out, sims


def _iid_parts(pi, pd, ps):
    m = ch.Model.iid(pi, pd, ps)
    return m.pi0, m.A, m.E


def reads_for(source, n, centers, clusters, sims):
    half = 1 if source == 'real_heldout' else 0
    rng = np.random.default_rng([919, half])
    pool = [(ci, ri) for ci in range(half, 10000, 2) for ri in range(len(clusters[ci]))]
    pick = rng.choice(len(pool), size=3 * n, replace=False)
    out = []
    for p in pick:
        ci, ri = pool[p]
        if source.startswith('real'):
            y = [B2I[c] for c in clusters[ci][ri]]
        else:
            variant = source.split('_')[1]
            srng = np.random.default_rng([77, int(p)])
            y = ch.simulate([B2I[c] for c in centers[ci]], sims[variant], srng)
        if abs(len(y) - ch.L) <= ch.D:
            out.append((ci, y))
        if len(out) == n:
            break
    return out


def stats(logpost, true):
    """Per-read statistics from a log-posterior vector over the index set."""
    s = logpost
    st = s[true]
    mx = s.max()
    lq_tau = np.array([tau * (st - mx) - math.log(np.exp(tau * (s - mx)).sum()) for tau in TAUS])
    conf_tau = np.array([-math.log(np.exp(tau * (s - mx)).sum()) for tau in TAUS])   # log max posterior
    rank = int((s > st).sum())
    top = np.sort(np.partition(s, -64)[-64:])[::-1]
    cs = np.logaddexp.accumulate(top) - (mx + math.log(np.exp(s - mx).sum()))
    return lq_tau, conf_tau, rank, cs[[0, 3, 15, 63]]


def run(a):
    centers, clusters = rr.load()
    mods, sims = models()
    C = np.array([[B2I[c] for c in s[:NC]] for s in centers], dtype=np.int64)
    name = f'{a.source}__{a.decoder.replace(":", "-")}'
    od = OUT / 'eval' / name
    od.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    items, trie = None, None
    kind, mname = a.decoder.split(':')
    model = mods[mname]
    for ci in range(a.n // CHUNK):
        f = od / f'chunk_{ci:03d}.npz'
        if f.exists():
            continue
        if time.perf_counter() - t0 > a.budget:
            print('budget reached; rerun to continue', flush=True)
            return
        if items is None:
            items = reads_for(a.source, a.n, centers, clusters, sims)
        part = items[ci * CHUNK:(ci + 1) * CHUNK]
        reads, true = [y for _, y in part], np.array([c for c, _ in part])
        tb = time.perf_counter()
        res = []
        if kind == 'marg':
            lp = ch.marginal_logapp(model, reads, NC)
            for b in range(len(reads)):
                s = lp[b, 0][C[:, 0]].copy()
                for t in range(1, NC):
                    s += lp[b, t][C[:, t]]
                s -= s.max() + math.log(np.exp(s - s.max()).sum())
                res.append(stats(s, true[b]))
        else:
            if trie is None:
                trie = ch.Trie(C)
            Y, lens = ch.pad(reads)
            beta = ch.backward(model, Y, lens, t_stop=NC)[:, NC]
            for b in range(len(reads)):
                res.append(stats(ch.joint_logpost_trie(model, Y[b:b + 1], lens[b:b + 1], beta[b], trie), true[b]))
        tper = (time.perf_counter() - tb) / len(reads)
        np.savez(f, lq_tau=np.stack([r[0] for r in res]), conf_tau=np.stack([r[1] for r in res]),
                 rank=np.array([r[2] for r in res]), logS=np.stack([r[3] for r in res]), t_per_read=tper)
        print(f'{name} chunk {ci}: {len(reads)} reads, {tper * 1e3:.1f} ms/read', flush=True)
    print('done', name, flush=True)


def load_eval(name):
    d = [np.load(f) for f in sorted((OUT / 'eval' / name).glob('chunk_*.npz'))]
    return {k: np.concatenate([x[k] for x in d]) if d[0][k].ndim else np.array([float(x[k]) for x in d])
            for k in ('lq_tau', 'conf_tau', 'rank', 'logS', 't_per_read')}


def ece(conf, correct, nb=10):
    bins = np.minimum((conf * nb).astype(int), nb - 1)
    e = 0.0
    for b in range(nb):
        m = bins == b
        if m.any():
            e += abs(correct[m].mean() - conf[m].mean()) * m.mean()
    return e


def metrics(e, it, M=10000):
    LN2 = math.log(2)
    rank = e['rank']
    eps = (rank >= 1).mean()
    hard = (-(1 - eps) * math.log2(1 - eps) - eps * math.log2(eps / (M - 1))) if 0 < eps < 1 else 0.0
    soft = float((-e['lq_tau'][:, it] / LN2).mean())
    conf = np.exp(e['conf_tau'][:, it])
    return dict(eps=eps, soft=soft, hard=hard, gain=hard - soft, ece=ece(conf, rank == 0),
                se_gain=float(np.std(-e['lq_tau'][:, it] / LN2, ddof=1) / math.sqrt(len(rank))))


def summary(names, tau_from):
    i1 = int(np.where(TAUS == 1.0)[0][0])
    for nm in names:
        e = load_eval(nm)
        n = len(e['rank'])
        if tau_from:
            ef = load_eval(tau_from)
            it = int(np.argmin((-ef['lq_tau']).mean(0)))
            ev = e
        else:                                                   # choose tau on even reads, report odd reads
            fit = np.arange(n) % 2 == 0
            it = int(np.argmin((-e['lq_tau'][fit]).mean(0)))
            ev = {k: (v[~fit] if k != 't_per_read' else v) for k, v in e.items()}
        u, c = metrics(ev, i1), metrics(ev, it)
        print(f'{nm}: n={len(ev["rank"])} eps={u["eps"]:.4f} hard={u["hard"]:.3f} | uncal soft={u["soft"]:.3f} '
              f'gain={u["gain"]:.3f} ECE={u["ece"]:.4f} | tau*={TAUS[it]} soft={c["soft"]:.3f} gain={c["gain"]:.3f} '
              f'(se~{c["se_gain"]:.3f}) ECE={c["ece"]:.4f} | {1e3 * e["t_per_read"].mean():.1f} ms/read', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--source')
    ap.add_argument('--decoder')
    ap.add_argument('--n', type=int, default=1500)
    ap.add_argument('--budget', type=float, default=25)
    ap.add_argument('--summary', nargs='*')
    ap.add_argument('--tau_from')
    a = ap.parse_args()
    if a.summary:
        summary(a.summary, a.tau_from)
    else:
        run(a)
