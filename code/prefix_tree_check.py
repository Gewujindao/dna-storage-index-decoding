"""Check: prefix-tree joint posterior equals direct enumeration over the index set (small subset of reads)."""
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import channel_model as ch  # noqa: E402
import evaluate_dataset1 as ev  # noqa: E402
import real_reads_dataset1 as rr  # noqa: E402

centers, clusters = rr.load()
mods, sims = ev.models()
C = np.array([[ev.B2I[c] for c in s[:ev.NC]] for s in centers], dtype=np.int64)
trie = ch.Trie(C)
print('trie nodes', trie.nodes, 'vs', C.size, flush=True)
items = ev.reads_for('real_design', 6, centers, clusters, sims)
for mname in ('iid', 'hmm2x'):
    m = mods[mname]
    reads = [y for _, y in items]
    Y, lens = ch.pad(reads)
    beta = ch.backward(m, Y, lens, t_stop=ev.NC)[:, ev.NC]
    errs, tt, te = [], 0.0, 0.0
    for b in range(len(reads)):
        t0 = time.perf_counter(); a = ch.joint_logpost_trie(m, Y[b:b + 1], lens[b:b + 1], beta[b], trie); tt += time.perf_counter() - t0
        t0 = time.perf_counter(); e = ch.joint_logpost(m, reads[b], C); te += time.perf_counter() - t0
        errs.append(np.abs(np.exp(a) - np.exp(e)).max())
    print(f'{mname}: max |p_trie - p_enum| = {max(errs):.2e}; ms/read trie {tt / len(reads) * 1e3:.1f}, enum {te / len(reads) * 1e3:.1f}', flush=True)
