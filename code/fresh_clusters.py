"""Microsoft clusters not drawn by any earlier read sampling.
Replays every earlier sampling call conservatively (whole pick lists, several n values) and prints the fresh set.
  python -X utf8 fresh_clusters.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import real_reads_dataset1 as rr  # noqa: E402

OUT = HERE.parent / 'results' / 'end_to_end'


def used_clusters(clusters):
    used = set()
    pool_all = [(ci, ri) for ci, c in enumerate(clusters) for ri in range(len(c))]
    pick = np.random.default_rng(2026).choice(len(pool_all), size=3000, replace=False)       # real_reads_dataset1
    used |= {pool_all[p][0] for p in pick}
    pool_even = [(ci, ri) for ci in range(0, 10000, 2) for ri in range(len(clusters[ci]))]
    pick = np.random.default_rng(303).choice(len(pool_even), size=4000, replace=False)       # channel_fit_dataset1
    used |= {pool_even[p][0] for p in pick}
    for half in (0, 1):                                                                        # evaluate_dataset1
        pool = [(ci, ri) for ci in range(half, 10000, 2) for ri in range(len(clusters[ci]))]
        for n in (500, 1000, 1500, 2000):
            pick = np.random.default_rng([919, half]).choice(len(pool), size=3 * n, replace=False)
            used |= {pool[p][0] for p in pick}
    return used


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    centers, clusters = rr.load()
    used = used_clusters(clusters)
    fresh = np.array([ci for ci in range(10000) if ci not in used and len(clusters[ci]) > 0])
    empty_fresh = [ci for ci in range(10000) if ci not in used and len(clusters[ci]) == 0]
    sizes = np.array([len(clusters[ci]) for ci in fresh])
    print(f'used clusters={len(used)} fresh non-empty={len(fresh)} fresh empty={len(empty_fresh)}')
    print(f'fresh reads={sizes.sum()} size median={np.median(sizes):.0f} mean={sizes.mean():.1f} '
          f'p10={np.percentile(sizes, 10):.0f} p90={np.percentile(sizes, 90):.0f}')
    np.savez(OUT / 'fresh_clusters.npz', fresh=fresh, empty_fresh=np.array(empty_fresh, dtype=int), sizes=sizes)
    print('done fresh')


if __name__ == '__main__':
    main()
