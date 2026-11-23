"""Engine diagnostic: relate index-region channel events to the rank of the true index (M=10^4, r=3)."""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trellis_index_sim as ab  # noqa: E402

M, r, NREAD = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
k = math.ceil(math.log(M, 4) - 1e-12)
A = ab.parity_columns(k, r)
n = k + r
C = ab.codewords(M, k, A)
rng = np.random.default_rng(7)


def channel_events(x, rng):
    out, ev = [], []
    for pos, s in enumerate(x):
        m = 0
        while m < 2 and rng.random() < ab.P_I:
            out.append(int(rng.integers(4))); m += 1; ev.append(('I', pos))
        if rng.random() < ab.P_D:
            ev.append(('D', pos)); continue
        if rng.random() < ab.P_S:
            out.append(int((s + rng.integers(1, 4)) % 4)); ev.append(('S', pos))
        else:
            out.append(int(s))
    return out, ev


true = rng.integers(M, size=NREAD)
reads, evs = [], []
for b in range(NREAD):
    x = ab.strand(C[true[b]], rng)
    y, ev = channel_events(x, rng)
    reads.append(y); evs.append(ev)
app = ab.index_app(reads, n)
n = ab.INFO_POS[n - 1] + 1          # event window in strand positions (index incl. interleaved markers)
nsym = k + r
sym_ok = (app.argmax(axis=2) == C[true]).mean(axis=0)
print('per-position symbol-APP argmax accuracy:', np.round(sym_ok, 3))
logp = np.log(np.maximum(app, 1e-300))
rows = []
for b in range(NREAD):
    s = sum(logp[b, t][C[:, t]] for t in range(nsym))
    rank = int((s > s[true[b]]).sum())
    e_idx = [e for e in evs[b] if e[1] < n]
    e_near = [e for e in evs[b] if n <= e[1] < n + 3]
    rows.append((rank, len(e_idx), ''.join(t for t, _ in e_idx), len(e_near)))
rows = np.array(rows, dtype=object)
for ne in range(4):
    sel = [x for x in rows if x[1] == ne]
    if sel:
        rk = np.array([x[0] for x in sel])
        print(f'events_in_index={ne}: reads={len(sel)} P(rank>=1)={np.mean(rk >= 1):.3f} P(rank>=16)={np.mean(rk >= 16):.3f}')
sel = [x for x in rows if x[1] == 0 and x[0] >= 1]
print('rank>=1 with no index-region event: n=', len(sel), ' events just after index among them:', sum(1 for x in sel if x[3] > 0))
types = {}
for x in rows:
    if x[1] == 1:
        types.setdefault(x[2], []).append(x[0])
for t, v in types.items():
    v = np.array(v)
    print(f'single event {t}: n={len(v)} P(rank>=1)={np.mean(v >= 1):.3f} median rank={np.median(v)}')
