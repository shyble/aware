#!/usr/bin/env python3
"""Are caps domain-specific, and does having more of them help?

The grid's failure has one measured cause: every cap serves both corpora,
so a per-cap accept/reject decision cannot separate helpful learning from
harmful. The pilot named the redirect — "bigger K" — and it was never
tested. This tests it.

Method: take ONE model (A-trained, before any B exposure) and run it over
both corpora, recording which cap wins at each position. For every cap:

    skew = firings_on_A / (firings_on_A + firings_on_B)

Equal sample sizes from both corpora, so 0.5 means perfectly shared, 1.0
means the cap is A's alone, 0.0 means B's alone.

What the shape means:

  BIMODAL (mass near 0 and 1)  caps ARE domain-specific. The mechanism
                              has something to protect, and more caps
                              should give more separation.
  UNIMODAL at 0.5             every cap is shared. No accept/reject rule
                              can work at ANY K, because the property it
                              needs does not exist.

The headline number is PROTECTABLE MASS: the fraction of A's activity
that happens in caps B barely touches. That is the ceiling on what any
per-cap protection could ever preserve.

Running this on both cap layers of a hierarchical model gives the
K-scaling evidence directly — same model, same corpora, K=330 vs K=64.

Usage:
    python3 scripts/cap_domain_skew.py A_WINNERS.bin B_WINNERS.bin N_CAPS [label]
"""

import sys
import numpy as np

# A cap is "owned" by a domain when that domain drives at least this much
# of its traffic. 0.9 is deliberately demanding: protection is only
# meaningful if the other domain almost never routes there.
OWNED = 0.9


def counts(path, k):
    w = np.fromfile(path, dtype=np.uint32).astype(np.int64)
    return np.bincount(w, minlength=k)[:k].astype(np.float64)


def main():
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    a_path, b_path, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
    label = sys.argv[4] if len(sys.argv) > 4 else f"K={k}"

    na, nb = counts(a_path, k), counts(b_path, k)
    total = na + nb
    live = total > 0

    print(f"\n  ══ {label} ══")
    print(f"  caps          {k}  ({int(live.sum())} fire at all)")
    print(f"  A positions   {int(na.sum())}    B positions {int(nb.sum())}")
    print(f"  fired by A    {int((na > 0).sum())}/{k}"
          f"      fired by B {int((nb > 0).sum())}/{k}")

    skew = np.full(k, np.nan)
    skew[live] = na[live] / total[live]

    # Histogram weighted by A's traffic, not by cap count: a cap holding
    # 2 of A's positions and one holding 20000 must not count equally.
    print("\n  skew distribution (share of each cap's traffic coming from A)")
    print(f"  {'bin':>10} {'caps':>6} {'% of A traffic':>16}")
    edges = [0.0, 0.1, 0.3, 0.45, 0.55, 0.7, 0.9, 1.0001]
    names = ["B only", "B-heavy", "B-lean", "SHARED", "A-lean", "A-heavy", "A only"]
    for lo, hi, name in zip(edges[:-1], edges[1:], names):
        sel = live & (skew >= lo) & (skew < hi)
        share = 100.0 * na[sel].sum() / max(na.sum(), 1)
        bar = "#" * int(round(share / 2))
        print(f"  {name:>10} {int(sel.sum()):>6} {share:>15.1f}%  {bar}")

    a_owned = live & (skew >= OWNED)
    protectable = 100.0 * na[a_owned].sum() / max(na.sum(), 1)
    b_owned = live & (skew <= 1 - OWNED)
    b_protect = 100.0 * nb[b_owned].sum() / max(nb.sum(), 1)

    print(f"\n  PROTECTABLE MASS   {protectable:.1f}% of A's activity is in caps that")
    print(f"                     B drives <{100*(1-OWNED):.0f}% of ({int(a_owned.sum())} caps)")
    print(f"  (mirror)           {b_protect:.1f}% of B's activity is in B-owned caps"
          f" ({int(b_owned.sum())} caps)")

    # Concentration of the shared region — where the damage came from.
    shared = live & (skew > 1 - OWNED) & (skew < OWNED)
    print(f"  CONTESTED          {int(shared.sum())} caps carry"
          f" {100.0 * na[shared].sum() / max(na.sum(), 1):.1f}% of A's activity")

    return protectable


if __name__ == "__main__":
    main()
