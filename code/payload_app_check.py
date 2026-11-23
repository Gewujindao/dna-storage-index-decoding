"""Diagnostic of payload symbol APPs on design-half reads already used by the earlier evaluations.
  python -X utf8 payload_app_check.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import real_reads_dataset1 as rr  # noqa: E402
import channel_model as ch  # noqa: E402
import evaluate_dataset1 as ev  # noqa: E402

centers, clusters = rr.load()
mods, _ = ev.models()
B = ev.B2I
fresh = set(np.load(HERE.parent / 'results' / 'end_to_end' / 'fresh_clusters.npz')['fresh'].tolist())
items = [(ci, 0) for ci in range(0, 10000, 2) if ci not in fresh and clusters[ci]
         and abs(len(clusters[ci][0]) - 110) <= 10][:40]
reads = [[B[c] for c in clusters[ci][ri]] for ci, ri in items]
truth = np.array([[B[c] for c in centers[ci]] for ci, _ in items])
for nm in ('iid', 'hmm2x'):
    lp = ch.marginal_logapp(mods[nm], reads, 110)
    err = (lp.argmax(2) != truth).mean(0)
    print(nm, 'overall', round(float(err.mean()), 4), 'by 10:', ' '.join(f'{err[i:i + 10].mean():.2f}' for i in range(0, 110, 10)))
# naive: read itself as estimate (no alignment)
naive = np.array([[r[t] if t < len(r) else 0 for t in range(110)] for r in reads])
print('read-as-is', ' '.join(f'{(naive != truth).mean(0)[i:i + 10].mean():.2f}' for i in range(0, 110, 10)))
print('done appcheck')
