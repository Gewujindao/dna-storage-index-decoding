"""The dataset-1 estimation procedure (channel_fit_dataset1) applied to the design half (even clusters) of a new
dataset. Short resumable segments: alignment cached in chunks; then a two-state HMM fit.
  python -X utf8 channel_fit_dataset2.py [--stage align|fit]
"""
import argparse
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import real_reads_dataset1 as rr  # noqa: E402
import channel_fit_dataset1 as m3  # noqa: E402
import dataset_loaders as dat  # noqa: E402

OUT = HERE.parent / 'results' / 'dataset2'
KINDS = 'MSD'
B2I = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


def design_length(centers):
    return Counter(len(c) for c in centers).most_common(1)[0][0]


def pool_design(centers, clusters, Ld):
    return [(ci, ri) for ci in range(0, len(centers), 2) if len(centers[ci]) == Ld
            for ri in range(len(clusters[ci]))]


def stage_align(budget):
    centers, clusters = dat.load_dnaformer()
    Ld = design_length(centers)
    pool = pool_design(centers, clusters, Ld)
    rng = np.random.default_rng(303)
    pick = rng.choice(len(pool), size=4000, replace=False)
    cd = OUT / 'align_cache'
    cd.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    for c0 in range(0, 4000, 250):
        f = cd / f'chunk_{c0:05d}.npz'
        if f.exists():
            continue
        if time.perf_counter() - t0 > budget:
            print('budget reached; rerun to continue', flush=True)
            return
        rows, nskip = [], 0
        for p in pick[c0:c0 + 250]:
            ci, ri = pool[p]
            x, y = centers[ci], clusters[ci][ri]
            if abs(len(y) - len(x)) > rr.BAND:
                nskip += 1
                continue
            dist, ops, ins = rr.align(x, y)
            if dist / len(x) > 0.3:
                nskip += 1
                continue
            rows.append((ci, ri, [KINDS.index(o) for o in ops], list(ins[:len(x)])))
        np.savez(f, ci=np.array([r[0] for r in rows]), ri=np.array([r[1] for r in rows]),
                 kind=np.array([r[2] for r in rows]), ins=np.array([r[3] for r in rows]), skipped=nskip)
        print(f'chunk {c0}: kept {len(rows)} skipped {nskip}', flush=True)
    print('done align', flush=True)


def stage_fit():
    centers, clusters = dat.load_dnaformer()
    Ld = design_length(centers)
    cached = []
    for f in sorted((OUT / 'align_cache').glob('chunk_*.npz')):
        with np.load(f) as z:
            cached.append({k: z[k] for k in ('ci', 'ri', 'kind', 'ins', 'skipped')})   # read each array once
    skipped = sum(int(c['skipped']) for c in cached)
    recs = [(int(c['ci'][i]), int(c['ri'][i]), c['kind'][i], c['ins'][i]) for c in cached for i in range(len(c['ci']))]
    seqs = []
    run_hist = np.zeros(12); trans = np.zeros((4, 4)); pos_err = np.zeros(Ld)
    subm = np.zeros((4, 4)); ins_eq = np.zeros(2)
    ctx = {k: np.zeros(4) for k in ('homo', 'nonhomo')}
    for ci, ri, kind, ins_raw in recs:
        x, y = centers[ci], clusters[ci][ri]
        ops = [KINDS[k] for k in kind]
        ins = [int(v) for v in ins_raw] + [0]
        seqs.append(np.array([min(ins[t], 2) * 3 + int(kind[t]) for t in range(Ld)]))
        err = np.array([(o != 'M') or (ins[t] > 0) for t, o in enumerate(ops)], dtype=int)
        pos_err += err
        r = 0
        for e in list(err) + [0]:
            if e:
                r += 1
            elif r:
                run_hist[min(r, 11)] += 1; r = 0
        cls = np.array([1 if o == 'S' else 2 if o == 'D' else (3 if ins[t] > 0 else 0) for t, o in enumerate(ops)])
        for t in range(Ld - 1):
            trans[cls[t], cls[t + 1]] += 1
        j = 0
        for t, o in enumerate(ops):
            hom = (t > 0 and x[t] == x[t - 1]) or (t + 1 < Ld and x[t] == x[t + 1])
            c = ctx['homo' if hom else 'nonhomo']
            c[0] += 1; c[1] += o == 'S'; c[2] += o == 'D'; c[3] += ins[t] > 0
            for _ in range(ins[t]):
                ins_eq[0] += 1; ins_eq[1] += (j < len(y) and y[j] == x[t]); j += 1
            if o in 'MS':
                if o == 'S' and j < len(y):
                    subm[B2I[x[t]], B2I[y[j]]] += 1
                j += 1
    X = np.stack(seqs)
    n = len(seqs)
    pe = (X != 0).mean()
    print(f'design length {Ld}; aligned {n}, skipped {skipped}; per-position error {pe:.4f}', flush=True)
    rates = np.bincount(X.ravel(), minlength=9) / X.size
    print('event frequencies:', {f'{mm}{k}': round(float(rates[mm * 3 + i]), 5) for mm in range(3) for i, k in enumerate(KINDS)}, flush=True)
    tot = run_hist.sum()
    print('run length (obs / iid):', ' '.join(f'{L}:{run_hist[L] / tot:.4f}/{(1 - pe) * pe ** (L - 1):.4f}' for L in range(1, 6)),
          f'mean {np.dot(np.arange(12), run_hist) / tot:.3f} (iid {1 / (1 - pe):.3f})', flush=True)
    tr = trans / trans.sum(1, keepdims=True)
    print('transitions M,S,D,I:', [' '.join(f'{v:.4f}' for v in row) for row in tr], flush=True)
    prof = pos_err / n
    print('position profile first 15:', ' '.join(f'{v:.3f}' for v in prof[:15]), flush=True)
    print('position profile 10-bins:', ' '.join(f'{prof[i:i + 10].mean():.4f}' for i in range(0, Ld, 10)), f'median {np.median(prof):.4f}', flush=True)
    for k, c in ctx.items():
        print(f'context {k}: sub={c[1] / c[0]:.4f} del={c[2] / c[0]:.4f} ins={c[3] / c[0]:.4f}', flush=True)
    sm = subm / subm.sum(1, keepdims=True)
    print('substitution matrix:', [' '.join(f'{v:.3f}' for v in row) for row in sm], flush=True)
    print(f'inserted base = strand base: {ins_eq[1] / max(ins_eq[0], 1):.3f}', flush=True)
    ck = OUT / 'bw_ckpt.npz'
    if ck.exists():
        c = np.load(ck)
        start, it0 = (c['pi0'], c['A'], c['E']), int(c['it'])
    else:
        start, it0 = None, 0
    if it0 < 60:
        pi0, A, E, ll, done = m3.baum_welch(seqs, 2, start=start, it0=it0, stop_after=24)
        np.savez(ck, pi0=pi0, A=A, E=E, it=done, ll=ll)
        print(f'Baum-Welch iterations done: {done}/60; budget reached; rerun to continue', flush=True)
        if done < 60:
            return
    c = np.load(ck)
    pi0, A, E, ll = c['pi0'], c['A'], c['E'], float(c['ll'])
    for s in range(2):
        print(f'HMM state {s}: pi0={pi0[s]:.3f} stay={A[s, s]:.4f} P(event)={1 - E[s, 0]:.4f} sub={E[s, 1]:.4f} del={E[s, 2]:.4f}', flush=True)
    np.savez(OUT / 'hmm_S2.npz', pi0=pi0, A=A, E=E, ll=ll)
    np.savez(OUT / 'measure.npz', run_hist=run_hist, trans=trans, prof=prof, subm=subm, ctx_homo=ctx['homo'],
             ctx_non=ctx['nonhomo'], ins_eq=ins_eq, rates=rates, Ld=Ld)
    print('done fit', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='align')
    ap.add_argument('--budget', type=float, default=18)
    a = ap.parse_args()
    if a.stage == 'describe':
        c, cl = dat.load_dnaformer()
        print(dat.describe(c, cl))
    elif a.stage == 'align':
        stage_align(a.budget)
    else:
        stage_fit()
