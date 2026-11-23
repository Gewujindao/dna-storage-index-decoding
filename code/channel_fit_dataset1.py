"""Measure how real nanopore reads violate the i.i.d. IDS assumptions.
Uses the design half: clusters with EVEN index.

Per read: banded unit-cost alignment to its design strand -> per input position t an event
(m = insertions before t, capped at 2; kind in {M, S, D}). Measures: rates, burst run lengths,
event-class transitions, position profile, homopolymer context, substitution / inserted-base
distributions; fits discrete HMMs (1, 2, 3 states) on the event sequences (Baum-Welch).
  python -X utf8 channel_fit_dataset1.py
"""
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import real_reads_dataset1 as rr  # noqa: E402

OUT = HERE.parent / 'results' / 'dataset1'
NREAD = 4000
KINDS = 'MSD'
B2I = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


def events_of(x, y):
    dist, ops, ins = rr.align(x, y)
    ins = list(ins)
    ins[len(x) - 1] += ins[len(x)]                    # trailing insertions folded into the last position
    ev = np.array([min(ins[t], 2) * 3 + KINDS.index(ops[t]) for t in range(len(x))])
    return dist, ops, ins, ev


def baum_welch(seqs, S, iters=60, seed=0, start=None, it0=0, stop_after=None):
    """Discrete HMM on event symbols 0..8; returns (pi0, A, E, loglik[, iterations done]).
    start/it0/stop_after allow resuming the identical iteration sequence in short segments."""
    rng = np.random.default_rng(seed)
    V = 9
    pi0 = np.full(S, 1.0 / S)
    A = np.full((S, S), 0.1 / max(S - 1, 1)) + np.eye(S) * (0.9 - 0.1 / max(S - 1, 1)) if S > 1 else np.ones((1, 1))
    E = rng.dirichlet(np.ones(V), size=S) * 0.2 + 0.8 * np.bincount(np.concatenate(seqs), minlength=V) / sum(len(s) for s in seqs)
    E[:, 0] += np.linspace(0.3, -0.3, S) if S > 1 else 0          # break symmetry: state 0 'good'
    E = np.clip(E, 1e-6, None)
    E /= E.sum(1, keepdims=True)
    if start is not None:
        pi0, A, E = start
    X = np.stack(seqs)
    N, T = X.shape
    done = it0
    for it in range(it0, iters):
        if stop_after is not None and it - it0 >= stop_after:
            break
        done = it + 1
        em = E[:, X].transpose(1, 2, 0)                # (N, T, S)
        al = np.zeros((N, T, S)); c = np.zeros((N, T))
        a = pi0 * em[:, 0]
        c[:, 0] = a.sum(1); al[:, 0] = a / c[:, :1]
        for t in range(1, T):
            a = (al[:, t - 1] @ A) * em[:, t]
            c[:, t] = a.sum(1); al[:, t] = a / c[:, t:t + 1]
        be = np.ones((N, T, S))
        for t in range(T - 2, -1, -1):
            b = (em[:, t + 1] * be[:, t + 1]) @ A.T
            be[:, t] = b / c[:, t + 1:t + 2]
        g = al * be
        g /= g.sum(2, keepdims=True)
        xi = np.zeros((S, S))
        for t in range(T - 1):
            m = al[:, t][:, :, None] * A[None] * (em[:, t + 1] * be[:, t + 1])[:, None, :] / c[:, t + 1][:, None, None]
            xi += m.sum(0)
        pi0 = g[:, 0].mean(0)
        A = xi / xi.sum(1, keepdims=True)
        E = np.zeros((S, V))
        for v in range(V):
            E[:, v] = (g * (X == v)[:, :, None]).sum((0, 1))
        E = np.clip(E, 1e-9, None)
        E /= E.sum(1, keepdims=True)
    if stop_after is not None:
        return pi0, A, E, float(np.log(c).sum()) if done > it0 else float('nan'), done
    return pi0, A, E, float(np.log(c).sum())


def align_cached(centers, clusters, pick, pool, budget=20.0):
    """Align in chunks of 500 reads, cached on disk, so each invocation stays short. Returns True when complete."""
    import time
    t0 = time.perf_counter()
    cd = OUT / 'align_cache'
    cd.mkdir(parents=True, exist_ok=True)
    for c0 in range(0, len(pick), 500):
        f = cd / f'chunk_{c0:05d}.npz'
        if f.exists():
            continue
        if time.perf_counter() - t0 > budget:
            return False
        rows = []
        for p in pick[c0:c0 + 500]:
            ci, ri = pool[p]
            x, y = centers[ci], clusters[ci][ri]
            if abs(len(y) - len(x)) > rr.BAND:
                continue
            dist, ops, ins = rr.align(x, y)
            if dist / len(x) > 0.3:
                continue
            rows.append((ci, ri, [KINDS.index(o) for o in ops], list(ins[:len(x)])))
        np.savez(f, ci=np.array([r[0] for r in rows]), ri=np.array([r[1] for r in rows]),
                 kind=np.array([r[2] for r in rows]), ins=np.array([r[3] for r in rows]))
    return True


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    centers, clusters = rr.load()
    pool = [(ci, ri) for ci in range(0, 10000, 2) for ri in range(len(clusters[ci]))]
    rng = np.random.default_rng(303)
    pick = rng.choice(len(pool), size=NREAD, replace=False)
    if not align_cached(centers, clusters, pick, pool):
        print('budget reached; rerun to continue', flush=True)
        return
    cached = [np.load(f) for f in sorted((OUT / 'align_cache').glob('chunk_*.npz'))]
    recs = [(int(c['ci'][i]), int(c['ri'][i]), c['kind'][i], c['ins'][i]) for c in cached for i in range(len(c['ci']))]
    seqs, xs = [], []
    run_hist = np.zeros(12)
    trans = np.zeros((4, 4))                           # classes: 0 M(no ins), 1 S, 2 D, 3 ins-before-M
    pos_err = np.zeros(110); n_pos = 0
    ctx = {k: np.zeros(4) for k in ('homo', 'nonhomo')}   # counts: n, sub, del, ins
    subm = np.zeros((4, 4)); ins_eq_next = np.zeros(2)
    for ci, ri, kind, ins_raw in recs:
        x, y = centers[ci], clusters[ci][ri]
        ops = [KINDS[k] for k in kind]
        ins = [int(v) for v in ins_raw] + [0]
        ev = np.array([min(ins[t], 2) * 3 + int(kind[t]) for t in range(len(x))])
        seqs.append(ev); xs.append(x)
        err = np.array([(o != 'M') or (ins[t] > 0) for t, o in enumerate(ops)], dtype=int)
        pos_err += err; n_pos += 1
        r = 0
        for e in list(err) + [0]:
            if e:
                r += 1
            elif r:
                run_hist[min(r, 11)] += 1; r = 0
        cls = np.array([1 if o == 'S' else 2 if o == 'D' else (3 if ins[t] > 0 else 0) for t, o in enumerate(ops)])
        for t in range(len(cls) - 1):
            trans[cls[t], cls[t + 1]] += 1
        # alignment to read positions for substitution matrix / inserted bases
        j = 0
        for t, o in enumerate(ops):
            hom = (t > 0 and x[t] == x[t - 1]) or (t + 1 < len(x) and x[t] == x[t + 1])
            c = ctx['homo' if hom else 'nonhomo']
            c[0] += 1; c[1] += o == 'S'; c[2] += o == 'D'; c[3] += ins[t] > 0
            for _ in range(ins[t] if t < len(x) - 1 else 0):
                ins_eq_next[0] += 1; ins_eq_next[1] += (j < len(y) and y[j] == x[t]); j += 1
            if o in 'MS':
                if o == 'S' and j < len(y):
                    subm[B2I[x[t]], B2I[y[j]]] += 1
                j += 1
    X = np.stack(seqs)
    n = len(seqs)
    pe = X.size and (X != 0).mean()
    print(f'design reads aligned={n}; per-position error (any event)={pe:.4f}', flush=True)
    rates = np.bincount(X.ravel(), minlength=9) / X.size
    print('event frequencies (m,kind):', {f'{m}{k}': round(float(rates[m * 3 + i]), 5) for m in range(3) for i, k in enumerate(KINDS)}, flush=True)
    # burst runs
    tot_runs = run_hist.sum()
    p = pe
    print('run length distribution (observed / iid geometric):', flush=True)
    for L in range(1, 8):
        iid = (1 - p) * p ** (L - 1)
        print(f'  len {L}: {run_hist[L] / tot_runs:.4f} / {iid:.4f}', flush=True)
    print(f'  mean run length: {np.dot(np.arange(12), run_hist) / tot_runs:.3f} (iid {1 / (1 - p):.3f})', flush=True)
    tr = trans / trans.sum(1, keepdims=True)
    print('class transitions P(next | current), classes M, S, D, I:', flush=True)
    for i, nm in enumerate('MSDI'):
        print(f'  {nm}: ' + ' '.join(f'{v:.4f}' for v in tr[i]), flush=True)
    prof = pos_err / n_pos
    print('position profile (10-nt bins):', ' '.join(f'{prof[i:i + 10].mean():.4f}' for i in range(0, 110, 10)), flush=True)
    for k, c in ctx.items():
        print(f'context {k}: n={int(c[0])} sub={c[1] / c[0]:.4f} del={c[2] / c[0]:.4f} ins={c[3] / c[0]:.4f}', flush=True)
    sm = subm / subm.sum(1, keepdims=True)
    print('substitution matrix rows x=A,C,G,T -> y:', [' '.join(f'{v:.3f}' for v in row) for row in sm], flush=True)
    print(f'inserted base equals current strand base: {ins_eq_next[1] / max(ins_eq_next[0], 1):.3f} (uniform 0.25)', flush=True)
    fits = {}
    for S in (1, 2, 3):
        pi0, A, E, ll = baum_welch(seqs, S)
        k = (S - 1) + S * (S - 1) + S * 8
        bic = -2 * ll + k * math.log(X.size)
        fits[S] = (pi0, A, E, ll)
        print(f'HMM S={S}: loglik={ll:.1f} BIC={bic:.1f}', flush=True)
        for s in range(S):
            err_s = 1 - E[s, 0]
            print(f'   state {s}: pi0={pi0[s]:.3f} stay={A[s, s]:.4f} P(any event)={err_s:.4f} '
                  f'sub={E[s, 1]:.4f} del={E[s, 2]:.4f} ins1={E[s, 3:6].sum():.4f} ins2={E[s, 6:9].sum():.4f}', flush=True)
        np.savez(OUT / f'hmm_S{S}.npz', pi0=pi0, A=A, E=E, ll=ll)
    np.savez(OUT / 'measure.npz', run_hist=run_hist, trans=trans, prof=prof, subm=subm,
             ctx_homo=ctx['homo'], ctx_non=ctx['nonhomo'], ins_eq=ins_eq_next, rates=rates)
    print('done measure', flush=True)


if __name__ == '__main__':
    main()
