"""End-to-end read-back of the 2,000-strand file (8 RS(250,k) blocks over GF(256)) from the
decoded fresh reads. Variants A1 / A2 / B, codebooks 'all' (M = 10,000) and 'file' (M = 2,000), arms baseline /
design, coverage grid x 40 subsamples x theta grid. Resumable per coverage; stops after --budget seconds.
  python -X utf8 end_to_end_readback.py --budget 240        (main grid)
  python -X utf8 end_to_end_readback.py --rscheck           (real RS decoding on sampled real error patterns)
  python -X utf8 end_to_end_readback.py --summary
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

OUT = HERE.parent / 'results' / 'end_to_end'
B2I = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
NFILE, NB = 2000, 250
CS = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8]
THETAS = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]
NREP = 40
RATES = [0.6, 0.7, 0.8, 0.9]


def load():
    strands = np.load(OUT / 'file.npz')['strands']
    fs = sorted((OUT / 'decode_file').glob('chunk_*.npz'))
    d = [np.load(f) for f in fs]
    keys = ('ci', 'keep', 'b_arg', 'b_lmax', 'd_arg', 'd_lmax', 'bf_arg', 'bf_lmax', 'df_arg', 'df_lmax', 'app',
            'lq_b', 'lq_d', 'b_ltrue', 'd_ltrue')
    z = {k: np.concatenate([x[k] for x in d]) for k in keys}
    z['t_per_read'] = np.array([float(x['t_per_read']) for x in d])
    return strands, z


def setup():
    centers, clusters = rr.load()
    strands, z = load()
    pos = -np.ones(10000, dtype=np.int64)
    pos[strands] = np.arange(NFILE)
    src = pos[z['ci']]
    truth = np.array([[B2I[c] for c in centers[ci][10:110]] for ci in strands])            # (2000, 100)
    tbytes = (truth.reshape(NFILE, 25, 4) * np.array([64, 16, 4, 1])).sum(2)
    kp = z['keep']
    arms = {}
    for arm, a_key, l_key, cb in (('b_all', 'b_arg', 'b_lmax', 'all'), ('d_all', 'd_arg', 'd_lmax', 'all'),
                                  ('b_file', 'bf_arg', 'bf_lmax', 'file'), ('d_file', 'df_arg', 'df_lmax', 'file')):
        arg = np.where(kp, z[a_key], 0).astype(np.int64)
        tgt = pos[arg] if cb == 'all' else arg                    # -1: landed outside the file
        correct = kp & ((arg == z['ci']) if cb == 'all' else (arg == src))
        conf = np.where(kp, np.exp(np.where(kp, z[l_key], -1e9)), -1.0)
        arms[arm] = (tgt, correct, conf)
    return strands, z, src, tbytes, arms, int(len(z['ci']))


def blocks_W(strand_err, strand_era):
    """strand_err, strand_era: (2000, 25) bool -> per row (8 blocks x 25) counts 2e + x."""
    e = strand_err.reshape(8, NB, 25).sum(1)
    x = strand_era.reshape(8, NB, 25).sum(1)
    return 2 * e + x                                               # (8, 25)


def evaluate(variant, tgt, correct, conf, acc_base, land_rand, src, app, tbytes, theta):
    acc = acc_base & (conf >= theta)
    t = np.where(tgt >= 0, tgt, land_rand)
    a_idx = np.nonzero(acc)[0]
    tt, cc = t[a_idx], correct[a_idx]
    if variant == 'A1':
        cont = np.zeros(NFILE, bool); cont[tt[~cc]] = True
        good = np.zeros(NFILE, bool); good[tt[cc]] = True
        err = cont
        era = ~cont & ~good
        return blocks_W(np.repeat(err[:, None], 25, 1), np.repeat(era[:, None], 25, 1))
    if variant == 'A2':
        ncor = np.bincount(tt[cc], minlength=NFILE)
        wrong = tt[~cc]
        wsrc = src[a_idx][~cc]
        if len(wrong):
            pair = wrong * 10000 + wsrc
            up, cnt = np.unique(pair, return_counts=True)
            mw = np.zeros(NFILE, dtype=np.int64)
            np.maximum.at(mw, up // 10000, cnt)
        else:
            mw = np.zeros(NFILE, dtype=np.int64)
        ok = ncor > mw
        era = (ncor == mw)                                          # no reads at all, or a tie
        err = ~ok & ~era
        return blocks_W(np.repeat(err[:, None], 25, 1), np.repeat(era[:, None], 25, 1))
    # B: real payload consensus
    order = np.argsort(tt, kind='stable')
    tts = tt[order]
    acc_app = app[a_idx][order]                                     # (na, 100, 4)
    S = np.zeros((NFILE, 100, 4))
    if len(tts):
        starts = np.r_[0, np.nonzero(np.diff(tts))[0] + 1]
        S[tts[starts]] = np.add.reduceat(acc_app, starts, axis=0)
    has = np.zeros(NFILE, bool); has[tts] = True
    base = S.argmax(2)
    rb = (base.reshape(NFILE, 25, 4) * np.array([64, 16, 4, 1])).sum(2)
    era = np.repeat(~has[:, None], 25, 1)
    err = (rb != tbytes) & ~era
    return blocks_W(err, era)


def run(budget):
    strands, z, src, tbytes, arms, nread = setup()
    app = z['app']
    od = OUT / 'e2e'
    od.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    for ic, c in enumerate(CS):
        f = od / f'c_{ic:02d}.npz'
        if f.exists():
            continue
        if time.perf_counter() - t0 > budget:
            print('budget reached; rerun to continue', flush=True)
            return
        p = c * NFILE / nread
        res = {}
        for rep in range(NREP):
            rng = np.random.default_rng([707, rep])
            u = rng.random(nread)                                   # same subsample key for every c (nested)
            land = rng.integers(NFILE, size=nread)
            acc_base = u < p
            for arm, (tgt, correct, conf) in arms.items():
                for v in ('A1', 'A2', 'B'):
                    for it, th in enumerate(THETAS):
                        W = evaluate(v, tgt, correct, conf, acc_base, land, src, app, tbytes, th)
                        res.setdefault(f'{v}_{arm}', np.zeros((NREP, len(THETAS), 8, 25), dtype=np.int16))[rep, it] = W
        np.savez(f, c=c, p=p, **res)
        print(f'c={c}: done ({time.perf_counter() - t0:.0f} s)', flush=True)
    print('done e2e', flush=True)


def file_ok(W, nsym):
    """W (..., 8, 25) -> file recovered (all rows), and fraction of rows (outer codewords) recovered."""
    return (W <= nsym).all((-1, -2)), (W <= nsym).mean((-1, -2))


def summary():
    od = OUT / 'e2e'
    data = [np.load(f) for f in sorted(od.glob('c_*.npz'))]
    cs = [float(d['c']) for d in data]
    names = [k for k in data[0].files if k not in ('c', 'p')]
    i_fixed = {'b': THETAS.index(0.5), 'd': THETAS.index(0.95)}
    for nm in names:
        v, arm = nm.split('_', 1)
        print(f'== {nm}', flush=True)
        cmin, cmin_fixed = {}, {}
        for R in RATES:
            nsym = NB - round(R * NB)
            line = []
            for ic, d in enumerate(data):
                fo, ro = file_ok(d[nm], nsym)                      # (NREP, NTH)
                fr, rw = fo.mean(0), ro.mean(0)
                best = int(np.lexsort((rw, fr))[-1])
                ff = fr[i_fixed[arm[0]]]
                line.append(f'{cs[ic]:g}:{fr[best]:.2f}/{rw[best]:.3f}(th{THETAS[best]})|fx{ff:.2f}')
                if fr[best] >= 0.9 and R not in cmin:
                    cmin[R] = cs[ic]
                if ff >= 0.9 and R not in cmin_fixed:
                    cmin_fixed[R] = cs[ic]
            print(f'  R={R}: ' + ' '.join(line), flush=True)
        print('  c_min (best theta): ' + ', '.join(f'R{R}={cmin.get(R, ">8")}' for R in RATES)
              + ' | fixed theta: ' + ', '.join(f'R{R}={cmin_fixed.get(R, ">8")}' for R in RATES), flush=True)
        rs = []
        for ic, d in enumerate(data):                               # max rate with file recovery >= 0.9
            Wmax = d[nm].max((-1, -2))                              # (NREP, NTH)
            q = np.sort(Wmax, 0)[int(math.ceil(0.9 * NREP)) - 1]    # 36th smallest per theta
            rs.append(f'{cs[ic]:g}:{max(NB - q.min(), 0) / NB:.3f}')
        print('  R* (file recovery >= 0.9): ' + ' '.join(rs), flush=True)


def index_metrics():
    """Index quality on all decoded fresh reads (M = 10,000), and the window comparison."""
    LN2 = math.log(2)
    _, z = load()
    k = z['keep']
    for arm, it in (('b', 4), ('d', 8)):
        ar = z[f'{arm}_arg'][k]; ci = z['ci'][k]
        lq = z[f'lq_{arm}'][k][:, it]
        conf = np.exp(z[f'{arm}_lmax'][k])
        cor = ar == ci
        eps = 1 - cor.mean()
        bins = np.minimum((conf * 10).astype(int), 9)
        ece = sum(abs(cor[bins == b].mean() - conf[bins == b].mean()) * (bins == b).mean() for b in range(10) if (bins == b).any())
        print(f'fresh reads arm {arm}: n={k.sum()} eps={eps:.4f} soft={-lq.mean() / LN2:.3f} (se {lq.std() / LN2 / math.sqrt(len(lq)):.3f}) ECE={ece:.4f}')
    dl = -(z['lq_d'][k][:, 8] - z['lq_b'][k][:, 4]) / LN2
    print(f'  paired advantage design vs baseline: {dl.mean():.3f} +- {dl.std() / math.sqrt(len(dl)):.3f} bits/read')
    print(f'  decode time per read (file job, all parts): {1e3 * z["t_per_read"].mean():.1f} ms')
    wf = sorted((OUT / 'decode_window').glob('chunk_*.npz'))
    if wf:
        w = [np.load(f) for f in wf]
        wci = np.concatenate([x['ci'] for x in w])
        # pair with the same reads in the file job: the window job drew reads from the same item list
        sel = np.random.default_rng(606).choice(np.nonzero(z['keep'])[0], 2000, replace=False)   # same reads, window 0-9
        assert (z['ci'][sel] == wci).all()
        adv = {}
        for arm, it in (('b', 4), ('d', 8)):
            lq = np.concatenate([x[f'lq_{arm}'][:, it] for x in w]); ar = np.concatenate([x[f'{arm}_arg'] for x in w])
            lq0, ar0 = z[f'lq_{arm}'][sel, it], z[f'{arm}_arg'][sel]
            adv[arm] = (lq, lq0)
            print(f'same 2000 reads, arm {arm}: window 0-9 eps={np.mean(ar0 != wci):.4f} soft={-lq0.mean() / LN2:.3f} | '
                  f'window 50-59 eps={np.mean(ar != wci):.4f} soft={-lq.mean() / LN2:.3f}')
        for wname, i in (('0-9', 1), ('50-59', 0)):
            dl = (adv['d'][i] - adv['b'][i]) / LN2
            print(f'  advantage at window {wname}: {dl.mean():.3f} +- {dl.std() / math.sqrt(len(dl)):.3f} bits/read')
        dd = ((adv['d'][0] - adv['b'][0]) - (adv['d'][1] - adv['b'][1])) / LN2
        print(f'  window effect on the advantage (mid - start): {dd.mean():.3f} +- {dd.std() / math.sqrt(len(dd)):.3f}')


def rscheck(nsample=100):
    """Real RS decoding on real error patterns (variant B, both arms, c = 3, theta 0.9, rep 0)."""
    import reed_solomon as rs
    strands, z, src, tbytes, arms, nread = setup()
    app = z['app'].astype(np.float64)
    rng = np.random.default_rng([707, 0]); u = rng.random(nread); land = rng.integers(NFILE, size=nread)
    p = 3 * NFILE / nread
    agree, tot, succ = 0, 0, 0
    crng = np.random.default_rng(9)
    for arm in ('b_all', 'd_all'):
        tgt, correct, conf = arms[arm]
        acc = (u < p) & (conf >= 0.9)
        t = np.where(tgt >= 0, tgt, land)
        a_idx = np.nonzero(acc)[0]
        S = np.zeros((NFILE, 100, 4))
        np.add.at(S, t[a_idx], app[a_idx])
        has = np.zeros(NFILE, bool); has[t[a_idx]] = True
        rb = (S.argmax(2).reshape(NFILE, 25, 4) * np.array([64, 16, 4, 1])).sum(2)
        for _ in range(nsample // 2):
            b, r = int(crng.integers(8)), int(crng.integers(25))
            sl = slice(b * NB, (b + 1) * NB)
            nsym = int(crng.choice([25, 50, 75, 100]))
            c = rs.encode([int(v) for v in crng.integers(256, size=NB - nsym)], nsym)
            e = rb[sl, r] ^ tbytes[sl, r]                          # real error pattern (GF(256) addition = xor)
            recv = [ci ^ int(ei) for ci, ei in zip(c, e)]
            er = [i for i in range(NB) if not has[b * NB + i]]
            out = rs.decode(recv, nsym, er)
            ok = out == c
            ne = int(((e != 0) & has[sl]).sum())
            agree += ok == (2 * ne + len(er) <= nsym); tot += 1; succ += ok
    print(f'RS check on real B patterns: decoder outcome matches 2e+x<=n-k in {agree}/{tot}; decoded {succ}/{tot}')
    # A2 patterns near the threshold (design and baseline, c = 3.5 / 4, theta 0.8 / 0.5): an erroneous strand
    # carries the payload bytes of its plurality wrong source (real contents), empty strands are erasures
    agree, tot, succ = 0, 0, 0
    for arm, c_, th in (('d_all', 3.5, 0.8), ('b_all', 4.0, 0.5)):
        tgt, correct, conf = arms[arm]
        for rep in range(4):
            rng = np.random.default_rng([707, rep]); u = rng.random(nread); land = rng.integers(NFILE, size=nread)
            acc = (u < c_ * NFILE / nread) & (conf >= th)
            t = np.where(tgt >= 0, tgt, land)
            a_idx = np.nonzero(acc)[0]
            ncor = np.bincount(t[a_idx][correct[a_idx]], minlength=NFILE)
            votes = {}
            for j, s_, ok_ in zip(t[a_idx], src[a_idx], correct[a_idx]):
                if not ok_:
                    votes.setdefault(int(j), {}).setdefault(int(s_), 0)
                    votes[int(j)][int(s_)] += 1
            content = tbytes.copy()
            era = np.zeros(NFILE, bool)
            for j in range(NFILE):
                mw, ws = 0, -1
                for s_, v in votes.get(j, {}).items():
                    if v > mw:
                        mw, ws = v, s_
                if ncor[j] == mw:
                    era[j] = True
                elif mw > ncor[j]:
                    content[j] = tbytes[ws] if ws >= 0 else (tbytes[j] ^ 1)   # ws = -1 cannot occur here
            for b in range(8):
                for r in range(0, 25, 5):
                    sl = slice(b * NB, (b + 1) * NB)
                    nsym = 50
                    cw = rs.encode([int(v) for v in crng.integers(256, size=NB - nsym)], nsym)
                    e = content[sl, r] ^ tbytes[sl, r]
                    recv = [ci ^ int(ei) for ci, ei in zip(cw, e)]
                    er = [i for i in range(NB) if era[b * NB + i]]
                    out = rs.decode(recv, nsym, er)
                    ok = out == cw
                    ne = int(((e != 0) & ~era[sl]).sum())
                    agree += ok == (2 * ne + len(er) <= nsym); tot += 1; succ += ok
    print(f'RS check on real A2 patterns (rate 0.8): decoder outcome matches 2e+x<=n-k in {agree}/{tot}; decoded {succ}/{tot}')
    print('done rscheck')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--budget', type=float, default=240)
    ap.add_argument('--summary', action='store_true')
    ap.add_argument('--rscheck', action='store_true')
    ap.add_argument('--index', action='store_true')
    a = ap.parse_args()
    if a.summary:
        summary()
    elif a.rscheck:
        rscheck()
    elif a.index:
        index_metrics()
    else:
        run(a.budget)
