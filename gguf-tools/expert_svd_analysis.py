#!/usr/bin/env python3
"""Estimate low-rank (SVD) compressibility of DS4 routed-expert matrices.

DeepSeek V4 Flash stores every routed expert as a quantized matrix.  This tool
measures, per MoE layer and per projection (`ffn_gate_exps`, `ffn_up_exps`,
`ffn_down_exps`), what happens if each expert matrix is replaced by its rank-k
singular-value decomposition.

Model
-----
Every routed expert is a 2-D matrix ``W`` of shape ``m x n``:

    ffn_gate_exps / ffn_up_exps : (m, n) = (2048, 4096)   [d_ff, d_emb]
    ffn_down_exps               : (m, n) = (4096, 2048)   [d_emb, d_ff]

The rank-k truncation is ``W ~= U_k S_k V_k^T`` with ``U_k`` (m x k) and
``V_k`` (n x k).  A forward pass becomes two skinny GEMMs:

    W x  ~=  (U_k S_k) (V_k^T x)        # (m x k) . (k x n)

so the stored parameters are ``k*(m+n)`` (plus ``k`` singular values).  The
user-requested accounting ``k*min(m,n)`` is reported as well, but note that the
factorised base genuinely costs ``k*(m+n)`` params.

Metrics
-------
For a grid of ranks ``k`` this tool reports

  memory saved        : 1 - stored_bytes / quantized_expert_bytes
  decode speedup      : quantized_bytes / stored_bytes            (bandwidth bound)
  prefill speedup     : m*n / (k*(m+n))                           (compute bound)
  precision lost      : ||W - W_k||_F / ||W||_F  (relative Frobenius error)
  parameter ratio     : k*min(m,n) / (m*n)  and  k*(m+n) / (m*n)

Decode (batch 1) is memory-bandwidth bound: time is proportional to the bytes
moved per expert, so the speedup is the byte ratio.  Prefill (large batch) is
compute bound: time is proportional to multiply-accumulates, so the speedup is
the MAC ratio.  These are roofline *upper bounds for the expert projections
only*; attention, routing, shared experts and KV traffic are unaffected and
dilute the end-to-end gain.

Precision is measured against the IQ2_XXS/Q2_K *dequantized* weights, i.e. the
reference the engine actually computes with.  The original float weights are
not in the GGUF, so this is the truncation error added on top of the existing
quantization error, not the total error against the float model.

The full singular spectrum is obtained from the smaller Gram matrix
(``W W^T`` when ``m <= n`` else ``W^T W``), which is ``min(m,n) x min(m,n)``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from gguf import GGUFReader
from gguf.quants import dequantize

BASE_DTYPE_BYTES = {"f16": 2, "bf16": 2, "f32": 4}
PROJECTIONS = ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")
PROJ_SALT = {"ffn_gate_exps": 11, "ffn_up_exps": 29, "ffn_down_exps": 47}

# Worker globals, populated by _init_worker.
_TENSORS: dict[str, object] = {}


def _init_worker(path: str) -> None:
    """Open the GGUF once per process; the reader is mmap-backed and shared."""
    global _TENSORS
    reader = GGUFReader(path, mode="r")
    _TENSORS = {t.name: t for t in reader.tensors}


def _expert_count(tensor) -> int:
    """Number of experts in a routed-expert tensor (expert axis is GGUF dim -1)."""
    return int(tensor.shape[-1])


def _sample_experts(n_experts: int, count: int, seed: int) -> list[int]:
    if count <= 0 or count >= n_experts:
        return list(range(n_experts))
    rng = np.random.default_rng(seed & 0x7FFFFFFF)
    return sorted(int(x) for x in rng.choice(n_experts, size=count, replace=False))


def _analyze(task):
    name, expert, ranks, energy_levels, num_tols, base_bytes, sv_bytes, residual_bpw = task
    t = _TENSORS[name]
    n_experts = _expert_count(t)
    if expert >= n_experts:
        return None
    orig_bytes = t.n_bytes / n_experts
    W = dequantize(t.data[expert], t.tensor_type).astype(np.float32, copy=False)
    m, n = W.shape
    r = min(m, n)

    # Full spectrum from the smaller Gram matrix.  The discarded half of the
    # SVD carries no extra singular values, so this is exact for the spectrum.
    gram = W @ W.T if m <= n else W.T @ W
    ev = np.linalg.eigvalsh(gram)[::-1]          # ascending -> descending
    np.clip(ev, 0.0, None, out=ev)
    sigma = np.sqrt(ev)
    total = float(ev.sum())
    if total <= 0.0:
        total = 1e-30
    cum = np.cumsum(ev) / total                  # fraction of ||W||_F^2

    def eff_rank(level: float) -> int:
        idx = int(np.searchsorted(cum, level, side="left"))
        return min(idx + 1, r)

    eff = {level: eff_rank(level) for level in energy_levels}
    num = {tol: int(np.count_nonzero(sigma > tol * sigma[0])) for tol in num_tols}

    # Rank-sweep metrics.  The SVD base is optionally followed by a quantized
    # residual; a dense residual adds both bytes and a full-size GEMM.
    sweep = {}
    for k in ranks:
        if k <= 0 or k > r:
            continue
        kept = float(cum[k - 1])
        base_param = k * (m + n)
        base_bytes_k = base_param * base_bytes + k * sv_bytes
        residual_bytes = m * n * residual_bpw / 8.0 if residual_bpw > 0 else 0.0
        stored_bytes = base_bytes_k + residual_bytes
        mac = k * (m + n) + (m * n if residual_bpw > 0 else 0)
        sweep[k] = {
            "energy_kept": kept,
            "rel_err": math.sqrt(max(0.0, 1.0 - kept)),
            "base_param": base_param,
            "stored_bytes": stored_bytes,
            "mem_saved": 1.0 - stored_bytes / orig_bytes,
            "decode_speedup": orig_bytes / stored_bytes,
            "prefill_speedup": (m * n) / mac if mac > 0 else 0.0,
            "mac": mac,
            "param_ratio_user": (k * min(m, n)) / (m * n),
            "param_ratio_actual": base_param / (m * n),
        }

    # Rank at which the base stops being smaller than the quantized expert:
    # k * (m+n) * base_bytes = orig_bytes.
    crossover = orig_bytes / ((m + n) * base_bytes + sv_bytes)

    return {
        "quant": t.tensor_type.name,
        "m": m, "n": n, "rank_max": r, "orig_bytes": orig_bytes,
        "frobenius2": total, "sigma_max": float(sigma[0]),
        "effective_rank": eff, "numerical_rank": num,
        "crossover_rank": crossover, "sweep": sweep,
    }


def _mean(values):
    return float(np.mean(values)) if values else float("nan")


def _parse_layers(text: str, available: list[int]) -> list[int]:
    if text in ("all", ""):
        return available
    chosen = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            chosen.extend(range(int(lo), int(hi) + 1))
        elif part:
            chosen.append(int(part))
    return [l for l in chosen if l in available]


def collect(args):
    reader = GGUFReader(args.gguf, mode="r")
    tensors = {t.name: t for t in reader.tensors}
    layers = sorted({int(n.split(".")[1]) for n in tensors
                     if n.startswith("blk.") and "ffn_gate_exps" in n})
    layers = _parse_layers(args.layers, layers)

    ranks = [int(x) for x in args.ranks.split(",") if x.strip()]
    energy_levels = [float(x) for x in args.energy_levels.split(",") if x.strip()]
    num_tols = [float(x) for x in args.num_tols.split(",") if x.strip()]
    base_bytes = BASE_DTYPE_BYTES[args.base_dtype]

    tasks = []
    for layer in layers:
        for proj in PROJECTIONS:
            name = f"blk.{layer}.{proj}.weight"
            if name not in tensors:
                continue
            n_experts = _expert_count(tensors[name])
            seed = layer * 1000 + PROJ_SALT[proj]
            for e in _sample_experts(n_experts, args.experts, seed):
                tasks.append((name, e, ranks, energy_levels, num_tols,
                              base_bytes, args.sv_bytes, args.residual_bpw))
    del reader, tensors

    results: dict[tuple[int, str], list[dict]] = {}
    if args.workers <= 1:
        _init_worker(args.gguf)
        for task, out in zip(tasks, map(_analyze, tasks)):
            if out is not None:
                results.setdefault((int(task[0].split(".")[1]),
                                    task[0].split(".")[2]), []).append(out)
    else:
        with ProcessPoolExecutor(max_workers=args.workers,
                                 initializer=_init_worker,
                                 initargs=(args.gguf,)) as pool:
            for task, out in zip(tasks, pool.map(_analyze, tasks,
                                                 chunksize=args.chunksize)):
                if out is not None:
                    results.setdefault((int(task[0].split(".")[1]),
                                        task[0].split(".")[2]), []).append(out)
    return results


def aggregate(results):
    """Average per-expert metrics into per-layer stats."""
    per_layer = []
    for (layer, proj), experts in sorted(results.items()):
        m, n = experts[0]["m"], experts[0]["n"]
        orig = experts[0]["orig_bytes"]
        entry = {
            "layer": layer, "projection": proj, "quant": experts[0]["quant"],
            "m": m, "n": n, "rank_max": experts[0]["rank_max"],
            "orig_bytes": orig,
            "crossover_rank": _mean([x["crossover_rank"] for x in experts]),
            "effective_rank": {
                lvl: _mean([x["effective_rank"][lvl] for x in experts])
                for lvl in experts[0]["effective_rank"]
            },
            "numerical_rank": {
                tol: _mean([x["numerical_rank"][tol] for x in experts])
                for tol in experts[0]["numerical_rank"]
            },
        }
        entry["sweep"] = {}
        for k in sorted(experts[0]["sweep"]):
            e_kept = _mean([x["sweep"][k]["energy_kept"] for x in experts])
            stored = _mean([x["sweep"][k]["stored_bytes"] for x in experts])
            mac = _mean([x["sweep"][k]["mac"] for x in experts])
            base_param = k * (m + n)
            entry["sweep"][k] = {
                "energy_kept": e_kept,
                "rel_err": math.sqrt(max(0.0, 1.0 - e_kept)),
                "stored_bytes": stored,
                "mem_saved": 1.0 - stored / orig,
                "decode_speedup": orig / stored,
                "prefill_speedup": (m * n) / mac if mac > 0 else 0.0,
                "mac": mac,
                "param_ratio_user": (k * min(m, n)) / (m * n),
                "param_ratio_actual": base_param / (m * n),
            }
        per_layer.append(entry)
    return per_layer


def _op_row(e, energy):
    """Pick the operating rank: smallest sampled sweep rank reaching the target."""
    target = e["effective_rank"][energy]
    candidates = sorted(e["sweep"])
    k = next((kk for kk in candidates if kk >= target), candidates[-1])
    return k, e["sweep"][k]


def _nearest_key(d, target):
    return min(d, key=lambda k: abs(k - target))


ROW_HEADER = (f"{'lyr':>3} {'proj':<13} {'quant':<8} {'shape':>10} "
              f"{'origMB':>7} {'R99':>5} {'R999':>5} {'nR1e3':>5} {'k_op':>5} "
              f"{'baseMB':>7} {'mem%':>6} {'dec_x':>6} {'pre_x':>6} {'err%':>6} "
              f"{'k*min/mn':>9} {'k(m+n)/mn':>10}")


def _fmt_row(tag, e, sw, k):
    er = e["effective_rank"]
    nr = e["numerical_rank"]
    return (f"{tag:>3} {e['projection']:<13} {e['quant']:<8} "
            f"{e['m']}x{e['n']:<6} {e['orig_bytes']/1e6:>7.3f} "
            f"{er[_nearest_key(er, 0.99)]:>5.0f} {er[_nearest_key(er, 0.999)]:>5.0f} "
            f"{nr[_nearest_key(nr, 1e-3)]:>5.0f} {k:>5} "
            f"{sw['stored_bytes']/1e6:>7.3f} {100*sw['mem_saved']:>6.1f} "
            f"{sw['decode_speedup']:>6.2f} {sw['prefill_speedup']:>6.2f} "
            f"{100*sw['rel_err']:>6.1f} {sw['param_ratio_user']:>9.3f} "
            f"{sw['param_ratio_actual']:>10.3f}")


def print_report(per_layer, args):
    energy = float(args.energy_levels.split(",")[0])
    print()
    print(f"SVD compressibility of routed experts in {os.path.basename(args.gguf)}")
    print(f"operating energy = {energy:.4f}   base = {args.base_dtype}   "
          f"residual = {'dropped' if args.residual_bpw <= 0 else f'{args.residual_bpw:g} bpw'}")
    print("R99/R999 = numerical rank for 99%/99.9% spectral energy; "
          "nR1e3 = rank at sigma > 1e-3*sigma_max")
    print()
    print(ROW_HEADER)
    print("-" * len(ROW_HEADER))
    for e in per_layer:
        k, sw = _op_row(e, energy)
        print(_fmt_row(e["layer"], e, sw, k))

    print()
    print("Average over all sampled layers (equal weight per layer/projection)")
    print(ROW_HEADER)
    print("-" * len(ROW_HEADER))
    for proj in PROJECTIONS:
        rows = [e for e in per_layer if e["projection"] == proj]
        if not rows:
            continue
        chosen = [_op_row(r, energy) for r in rows]
        sws = [c[1] for c in chosen]
        agg = {
            "projection": proj, "quant": rows[0]["quant"],
            "m": rows[0]["m"], "n": rows[0]["n"],
            "orig_bytes": _mean([r["orig_bytes"] for r in rows]),
            "effective_rank": {lvl: _mean([r["effective_rank"][lvl] for r in rows])
                               for lvl in rows[0]["effective_rank"]},
            "numerical_rank": {tol: _mean([r["numerical_rank"][tol] for r in rows])
                               for tol in rows[0]["numerical_rank"]},
        }
        avg_sw = {key: _mean([s[key] for s in sws])
                  for key in ("stored_bytes", "mem_saved", "decode_speedup",
                              "prefill_speedup", "rel_err", "param_ratio_user",
                              "param_ratio_actual")}
        print(_fmt_row("avg", agg, avg_sw, _mean([c[0] for c in chosen])))

    print()
    print("All-layer average, split by energy target")
    print(f"{'energy':>8} {'rank':>6} {'mem%':>7} {'dec_x':>7} {'pre_x':>7} "
          f"{'err%':>7} {'k*min':>7} {'k(m+n)':>8}")
    print("-" * 60)
    for lvl in [float(x) for x in args.energy_levels.split(",")]:
        chosen = [_op_row(e, lvl) for e in per_layer]
        sws = [c[1] for c in chosen]
        print(f"{lvl:>8.4f} {_mean([c[0] for c in chosen]):>6.0f} "
              f"{100*_mean([s['mem_saved'] for s in sws]):>7.1f} "
              f"{_mean([s['decode_speedup'] for s in sws]):>7.2f} "
              f"{_mean([s['prefill_speedup'] for s in sws]):>7.2f} "
              f"{100*_mean([s['rel_err'] for s in sws]):>7.2f} "
              f"{_mean([s['param_ratio_user'] for s in sws]):>7.3f} "
              f"{_mean([s['param_ratio_actual'] for s in sws]):>8.3f}")

    print()
    print("Rank sweep (averaged over all sampled expert projections)")
    print(f"{'k':>6} {'energy%':>8} {'err%':>7} {'mem%':>7} "
          f"{'dec_x':>7} {'pre_x':>7} {'k*min':>7} {'k(m+n)':>8}")
    print("-" * 60)
    allk = sorted({k for e in per_layer for k in e["sweep"]})
    for k in allk:
        rows = [e["sweep"][k] for e in per_layer if k in e["sweep"]]
        print(f"{k:>6} {100*_mean([s['energy_kept'] for s in rows]):>8.2f} "
              f"{100*_mean([s['rel_err'] for s in rows]):>7.2f} "
              f"{100*_mean([s['mem_saved'] for s in rows]):>7.1f} "
              f"{_mean([s['decode_speedup'] for s in rows]):>7.2f} "
              f"{_mean([s['prefill_speedup'] for s in rows]):>7.2f} "
              f"{_mean([s['param_ratio_user'] for s in rows]):>7.3f} "
              f"{_mean([s['param_ratio_actual'] for s in rows]):>8.3f}")

    print()
    print("decode = bandwidth bound (byte ratio), prefill = compute bound (MAC ratio).")
    print("err is truncation error added on top of the existing 2-bit quantization,")
    print("measured against the dequantized weights the engine actually computes with.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-m", "--gguf", default="ds4flash.gguf")
    ap.add_argument("--layers", default="all", help="'all', '0,1,2' or '0-5'")
    ap.add_argument("--experts", type=int, default=8,
                    help="experts sampled per projection per layer (0 = all)")
    ap.add_argument("--ranks", default="64,128,192,256,384,512,768,1024",
                    help="rank grid for the sweep table")
    ap.add_argument("--energy-levels", default="0.99,0.999,0.9999")
    ap.add_argument("--num-tols", default="1e-2,1e-3,1e-4")
    ap.add_argument("--base-dtype", choices=sorted(BASE_DTYPE_BYTES), default="f16")
    ap.add_argument("--sv-bytes", type=int, default=2)
    ap.add_argument("--residual-bpw", type=float, default=0.0,
                    help="bits/weight kept for the residual (0 = pure truncation)")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--chunksize", type=int, default=1)
    ap.add_argument("--json", help="write full per-layer results as JSON")
    args = ap.parse_args()

    t0 = time.time()
    results = collect(args)
    per_layer = aggregate(results)
    print_report(per_layer, args)
    print(f"\nanalyzed {sum(len(v) for v in results.values())} expert matrices "
          f"in {time.time() - t0:.1f}s")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"per_layer": per_layer, "config": vars(args)}, fh, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
