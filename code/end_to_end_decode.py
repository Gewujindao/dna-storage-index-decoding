"""Decode every read of the end-to-end file once (decoders with the dataset-1 parameters).
Per read: baseline (per-symbol i.i.d. product) and design (joint two-state trie) index posteriors over the
10,000-strand codebook (first 10 nt) and over the file-only codebook (2,000 strands); payload symbol APPs
(positions 10..109) from the dataset-1 two-state channel. Window job: 2,000 file reads at positions 50..59.
Resumable chunks, stops after --budget seconds.
  python -X utf8 end_to_end_decode.py --job file|window --budget 240
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
import evaluate_dataset1 as ev  # noqa: E402

OUT = HERE.parent / 'results' / 'end_to_end'
B2I = ev.B2I
NC, NFILE, NEMPTY, NBLOCK = 10, 2000, 3, 250
CHUNK = 100
TAU_B = 0.6
I_TAU_B = int(np.where(np.isclose(ev.TAUS, TAU_B))[0][0])


def make_file():
    f = OUT / 'file.npz'
    if f.exists():
        z = np.load(f)
        return z['strands']
    fr = np.load(OUT / 'fresh_clusters.npz')
    rng = np.random.default_rng(505)
    s = np.concatenate([rng.choice(fr['fresh'], NFILE - NEMPTY, replace=False),
                        rng.choice(fr['empty_fresh'], NEMPTY, replace=False)])
    s = rng.permutation(s)                          # block b = s[250 b : 250 (b + 1)]
    np.savez(f, strands=s)
    return s


def dev_strands(clusters):
    """Pipeline/timing check only: uses 2,000 even clusters from the dataset-1 evaluation, outside the fresh set."""
    fresh = set(np.load(OUT / 'fresh_clusters.npz')['fresh'].tolist())
    return np.array([ci for ci in range(0, 10000, 2) if ci not in fresh and len(clusters[ci]) > 0][:NFILE])


def file_reads(strands, clusters):
    """All reads of the file clusters, in a fixed order; length filter applied later (flag)."""
    items = [(int(ci), ri) for ci in strands for ri in range(len(clusters[ci]))]
    return items


def app_with_beta(model, Y, lens, beta, npos):
    """Symbol APPs (log) at positions 0..npos-1 given the backward pass (same as ch.marginal_logapp)."""
    B, S = Y.shape[0], model.S
    al = np.zeros((B, ch.NS, S))
    al[:, ch.D, :] = model.pi0
    out = np.zeros((B, npos, 4))
    for t in range(npos):
        G = ch.emissions(model, Y, lens, t)
        bz = beta[:, t + 1] @ model.A.T
        app = np.zeros((B, 4))
        nal = np.zeros((B, ch.NS, S))
        for z in range(S):
            for l in range(4):
                bs = np.zeros((B, ch.NS))
                ch._shift_back(bs, bz[:, :, z], l - 1)
                app += (al[:, :, z][:, :, None] * G[z, l] * bs[:, :, None]).sum(1)
                w = al[:, :, z] * G[z, l].mean(2)
                tmp = np.zeros((B, ch.NS))
                ch._shift(tmp, w, l - 1)
                nal += tmp[:, :, None] * model.A[z][None, None, :]
        out[:, t] = np.log(np.maximum(app / app.sum(1, keepdims=True), 1e-300))
        al = nal / nal.sum((1, 2), keepdims=True)
    return out


def summarize(s, true):
    """argmax, log max posterior and log posterior of the true index (s is a normalised log posterior)."""
    return int(np.argmax(s)), float(s.max()), (float(s[true]) if true >= 0 else np.nan)


def tempered(s, tau):
    v = tau * s
    return v - (v.max() + math.log(np.exp(v - v.max()).sum()))


def run(job, budget):
    OUT.mkdir(parents=True, exist_ok=True)
    centers, clusters = rr.load()
    strands = make_file() if job != 'dev' else dev_strands(clusters)
    mods, _ = ev.models()
    m_b, m_d = mods['iid'], mods['hmm2x']
    items = file_reads(strands, clusters)
    pos_in_file = {int(ci): k for k, ci in enumerate(strands)}
    if job == 'window':
        rng = np.random.default_rng(606)
        ok = [k for k, (ci, ri) in enumerate(items) if abs(len(clusters[ci][ri]) - 110) <= ch.D]
        items = [items[k] for k in rng.choice(ok, 2000, replace=False)]
        offset = 50
    else:
        offset = 0
    C_all = np.array([[B2I[c] for c in s[offset:offset + NC]] for s in centers], dtype=np.int64)
    C_file = C_all[strands]
    trie_all, trie_file = None, None
    od = OUT / f'decode_{job}'
    od.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    nchunk = (len(items) + CHUNK - 1) // CHUNK
    for k in range(nchunk):
        f = od / f'chunk_{k:04d}.npz'
        if f.exists():
            continue
        if time.perf_counter() - t0 > budget:
            print('budget reached; rerun to continue', flush=True)
            return
        if trie_all is None:
            trie_all = ch.Trie(C_all)
            trie_file = ch.Trie(C_file) if job in ('file', 'dev') else None
        part = items[k * CHUNK:(k + 1) * CHUNK]
        ci_arr = np.array([ci for ci, _ in part])
        keep = np.array([abs(len(clusters[ci][ri]) - 110) <= ch.D for ci, ri in part])
        reads = [[B2I[c] for c in clusters[ci][ri]] for (ci, ri), kp in zip(part, keep) if kp]
        tci = ci_arr[keep]
        tf = np.array([pos_in_file[int(c)] for c in tci])
        nb = len(reads)
        rec = {key: np.full(len(part), np.nan) for key in
               ('b_arg', 'b_lmax', 'b_ltrue', 'd_arg', 'd_lmax', 'd_ltrue',
                'bf_arg', 'bf_lmax', 'df_arg', 'df_lmax')}
        lq_b, lq_d = np.full((len(part), len(ev.TAUS)), np.nan), np.full((len(part), len(ev.TAUS)), np.nan)
        app = np.zeros((len(part), 100, 4), dtype=np.float32)
        tb = time.perf_counter()
        if nb:
            Y, lens = ch.pad(reads)
            lp = ch.marginal_logapp(m_b, reads, NC, offset=offset)        # baseline symbol APPs
            beta = ch.backward(m_d, Y, lens)                              # design channel, all positions
            if job in ('file', 'dev'):
                pay = app_with_beta(m_d, Y, lens, beta, 110)[:, 10:]
            idx = np.nonzero(keep)[0]
            for b in range(nb):
                j = idx[b]
                s = lp[b, 0][C_all[:, 0]].copy()
                for t in range(1, NC):
                    s += lp[b, t][C_all[:, t]]
                s -= s.max() + math.log(np.exp(s - s.max()).sum())
                st = ev.stats(s, int(tci[b]))
                lq_b[j] = st[0]
                rec['b_arg'][j], rec['b_lmax'][j], rec['b_ltrue'][j] = summarize(tempered(s, TAU_B), int(tci[b]))
                sd = ch.joint_logpost_trie(m_d, Y[b:b + 1], lens[b:b + 1], beta[b, offset + NC], trie_all, offset=offset)
                st = ev.stats(sd, int(tci[b]))
                lq_d[j] = st[0]
                rec['d_arg'][j], rec['d_lmax'][j], rec['d_ltrue'][j] = summarize(sd, int(tci[b]))
                if job in ('file', 'dev'):
                    sf = s[strands] - (s[strands].max() + math.log(np.exp(s[strands] - s[strands].max()).sum()))
                    rec['bf_arg'][j], rec['bf_lmax'][j], _ = summarize(tempered(sf, TAU_B), int(tf[b]))
                    sdf = ch.joint_logpost_trie(m_d, Y[b:b + 1], lens[b:b + 1], beta[b, offset + NC], trie_file, offset=offset)
                    rec['df_arg'][j], rec['df_lmax'][j], _ = summarize(sdf, int(tf[b]))
                    app[j] = pay[b]
        tper = (time.perf_counter() - tb) / max(nb, 1)
        np.savez(f, ci=ci_arr, keep=keep, lq_b=lq_b, lq_d=lq_d, app=app, t_per_read=tper, **rec)
        print(f'{job} chunk {k}/{nchunk}: {nb} reads, {tper * 1e3:.1f} ms/read', flush=True)
    print(f'done decode {job}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--job', default='file')
    ap.add_argument('--budget', type=float, default=240)
    a = ap.parse_args()
    run(a.job, a.budget)
