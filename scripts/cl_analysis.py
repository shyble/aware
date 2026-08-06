#!/usr/bin/env python3
"""Offline analysis for the continual-learning instrumentation.

Two subcommands, matching the two mechanism-level questions:

  slabs   Which caps' parameter slabs changed between two checkpoints?
          Per-cap, per-component byte comparison. Under top-1 routing a
          cap that never fired during the second training cycle should be
          BYTE-IDENTICAL — any change in an unfired cap is a bug in the
          dispatch, not a research finding.

  winners Agreement between two winner dumps (same eval seed, so the
          same token stream): what fraction of tokens still route to the
          same cap? This is firing stability — the identity property
          every cap-structured architecture inherits.

Zero dependencies: the safetensors container is parsed directly
(8-byte little-endian header length, JSON header, raw data), so this
runs on any Python without an environment.

Usage:
  python3 scripts/cl_analysis.py slabs ckpt_A.safetensors ckpt_B.safetensors
  python3 scripts/cl_analysis.py winners winA.l1.bin winB.l1.bin
"""
import json
import struct
import sys
from collections import defaultdict

CKPT_PREFIX = "__capnative__."


def read_safetensors(path):
    """name -> (dtype, shape, raw bytes)."""
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        blob = f.read()
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        a, b = info["data_offsets"]
        out[name] = (info["dtype"], info["shape"], blob[a:b])
    return out


def elem_size(dtype):
    return {"F32": 4, "F16": 2, "BF16": 2, "F64": 8, "U32": 4, "I64": 8, "U8": 1}[dtype]


def cmd_slabs(path_a, path_b):
    a, b = read_safetensors(path_a), read_safetensors(path_b)
    if set(a) != set(b):
        only_a = sorted(set(a) - set(b))
        only_b = sorted(set(b) - set(a))
        sys.exit(f"tensor sets differ; only in A: {only_a}, only in B: {only_b}")

    # The routing cap count: layer 1 if hierarchical, else layer 0.
    keys_name = (
        CKPT_PREFIX + "l1.keys"
        if CKPT_PREFIX + "l1.keys" in a
        else CKPT_PREFIX + "l0.keys"
    )
    n_caps = a[keys_name][1][0]
    print(f"routing caps: {n_caps}  (from {keys_name})")

    # Cap-keyed tensors are the ones whose leading dimension is the
    # routing cap count. Everything else (embeddings, cap-layer
    # projections, the cap keys themselves) is reported separately as
    # the shared/leak-path group.
    changed_caps = defaultdict(list)  # cap index -> [tensor names]
    shared_changed = []
    shared_identical = []

    for name in sorted(a):
        (dt_a, shape_a, raw_a) = a[name]
        (dt_b, shape_b, raw_b) = b[name]
        if shape_a != shape_b or dt_a != dt_b:
            sys.exit(f"{name}: shape/dtype mismatch {shape_a}/{dt_a} vs {shape_b}/{dt_b}")
        if name.startswith(CKPT_PREFIX):
            tag = "IDENTICAL" if raw_a == raw_b else "CHANGED"
            print(f"  [caps-meta] {name:<40} {tag}")
            continue
        if shape_a and shape_a[0] == n_caps and len(shape_a) >= 2:
            per = len(raw_a) // n_caps
            for k in range(n_caps):
                if raw_a[k * per : (k + 1) * per] != raw_b[k * per : (k + 1) * per]:
                    changed_caps[k].append(name)
        else:
            (shared_changed if raw_a != raw_b else shared_identical).append(name)

    untouched = n_caps - len(changed_caps)
    print(f"\ncap slabs: {untouched}/{n_caps} caps byte-identical across ALL components")
    if changed_caps:
        by_count = sorted(changed_caps.items(), key=lambda kv: -len(kv[1]))
        print(f"changed caps ({len(changed_caps)}):")
        for k, names in by_count[:10]:
            comps = sorted({n.split(".")[0] for n in names})
            print(f"  cap {k:>4}: {len(names)} tensors touched  ({', '.join(comps)})")
        if len(by_count) > 10:
            print(f"  ... and {len(by_count) - 10} more")

    print(f"\nshared (non-cap-keyed) tensors — the leak paths:")
    for name in shared_changed:
        print(f"  CHANGED   {name}")
    for name in shared_identical:
        print(f"  identical {name}")


def cmd_winners(path_a, path_b):
    wa = open(path_a, "rb").read()
    wb = open(path_b, "rb").read()
    if len(wa) != len(wb):
        sys.exit(f"length mismatch: {len(wa)} vs {len(wb)} bytes (different eval seed or batch count?)")
    n = len(wa) // 4
    a = struct.unpack(f"<{n}I", wa)
    b = struct.unpack(f"<{n}I", wb)
    same = sum(1 for x, y in zip(a, b) if x == y)
    print(f"tokens: {n}")
    print(f"agreement: {same}/{n} = {100.0 * same / n:.2f}%")

    moved = defaultdict(int)
    for x, y in zip(a, b):
        if x != y:
            moved[(x, y)] += 1
    if moved:
        top = sorted(moved.items(), key=lambda kv: -kv[1])[:8]
        print("largest reroutes (from -> to: tokens):")
        for (x, y), c in top:
            print(f"  cap {x:>4} -> cap {y:>4}: {c}")


def main():
    if len(sys.argv) != 4 or sys.argv[1] not in ("slabs", "winners"):
        sys.exit(__doc__)
    if sys.argv[1] == "slabs":
        cmd_slabs(sys.argv[2], sys.argv[3])
    else:
        cmd_winners(sys.argv[2], sys.argv[3])


if __name__ == "__main__":
    main()
