# NVIDIA CUDA GPUs

[README](../README.md) | [Getting started](../README.md#start-here)

**Ada Lovelace is supported, including 48 GB L40S cards.** This is an important
use case: a model whose other inference implementations require newer GPUs
can still run here when DwarfStar supports its GGUF layout. The Flash kernels
do not require Blackwell's native FP4 instructions; Ada uses an appropriate
CUDA path. Support remains model-specific, not a promise to run arbitrary GGUFs.

## Build

Install the NVIDIA driver and CUDA toolkit, including `nvcc` and cuBLAS.
Use the local GPU architecture, or select Ada explicitly:

```sh
make cuda-generic
# For an Ada build:
make cuda CUDA_ARCH=sm_89
```

For a GB10 machine use the [DGX Spark](DGX_SPARK.md) target instead.

## Normal layer placement

Without tensor parallelism, DwarfStar places complete layers on the selected
GPUs. `--gpu-vram auto` uses reported free VRAM, reserving space for the graph
and context. Explicit budgets are comma-separated GiB values, one per device.

```sh
./download_model.sh ds4f-q2
./ds4 --cuda --gpu-devices 0,1,2,3 --gpu-vram auto --ctx 32768
```

This is also the multi-GPU placement mode for GLM. Startup refuses a layout
that would require unsupported CPU execution; reduce context or choose a
smaller model if necessary.

## Device scoring and role assignment

When several devices are configured, startup probes each one (PCIe link
generation/width, measured pinned host-to-device bandwidth, free/total VRAM,
memory bandwidth, FP16 throughput, SM count, tensor generation, peer access)
and normalizes those metrics across the set. Each device is scored for
prefill, decode, and capacity:

```text
S_prefill  = 0.5*norm(pcie_bw) + 0.3*norm(fp16) + 0.2*norm(vram_bw)
S_decode   = 0.4*norm(vram_bw) + 0.3*norm(fp16) + 0.2*norm(vram_free) + 0.1*norm(sm)
S_capacity = 0.8*norm(vram_free) + 0.1*norm(vram_bw) + 0.1*norm(pcie_bw)
```

The decode winner becomes the primary device, the capacity winner the expert
cache, and the prefill winner the secondary. `--gpu-devices` is reordered so
the layer packer fills primary, then cache, then secondary. Device 0 is the
one the engine treats as the dense-weight home, so the ordering is what makes
the roles take effect without a second placement path. With one device (or
`--disable-multi-gpu`) the plan degrades to single-GPU mode and the caller's
device order is left untouched, so nothing changes for existing setups.

The plan is emitted as TSON on stderr so it can be logged or diffed:

```text
ds4.roles.n_gpus = 2
ds4.roles.primary = 0
ds4.roles.cache = 1
ds4.roles.dev[0] = {id:0 name:... role:primary s_prefill:... s_decode:... ...}
```

Overrides:

```sh
./ds4 --cuda --gpu-devices 0,1 --gpu-vram auto \
  --force-primary-gpu 0 --force-cache-gpu 1 \
  --kv-cache-location vram --chunk-size 4096
```

* `--force-primary-gpu N` / `--force-cache-gpu N` pin a role by physical id.
* `--disable-multi-gpu` forces single-GPU mode.
* `--kv-cache-location vram|ram|gpu:N` records the requested KV home. Host
  (`ram`) residency is not implemented yet; the engine warns and keeps the KV
  cache in device VRAM.
* `--chunk-size N` sets the prefill chunk target (alias for
  `--prefill-chunk`); when unset the plan reports a recommendation from the
  primary device's free VRAM (512/2048/4096/8192).

`--cuda-tensor-parallel` fixes its own device order (homes first, then
partners), so scoring is skipped there.

## SSD streaming with multiple GPUs

`--ssd-streaming` can be combined with `--gpu-devices`/`--gpu-vram`. In that
mode one device (tier 0) computes, and every other configured device is used
as a pinned expert-cache tier, so a run draws on **SSD** (cold experts), **host
RAM** (the `--ssd-streaming-cache-experts` page cache), and **every GPU's
VRAM** (the Least-Stale resident pool) at once:

```sh
./ds4 --cuda --ssd-streaming --gpu-devices 0,1 --gpu-vram auto \
  --ssd-streaming-cache-experts 50GB --ctx 32768
```

The pool partitions one global slot namespace by tier (`slot_base`/`slot_cap`)
and records each slot's owning tier. On decode a hit on a secondary tier is
served with a peer copy of that expert's gate/up/down into the tier-0 per-call
table; a miss is uploaded into the table and promoted into a slot on whichever
tier has room. Compute stays on tier 0, so the proven single-device streaming
kernels are unchanged. Secondary tiers are only admitted when the peer path to
tier 0 was validated at startup; if peer access is unavailable the pool simply
stays on tier 0 and behavior is exactly the single-GPU streaming path. Startup
logs the split, e.g. `expert pool: N slots across 2 GPUs (tiers 0:153 1:846)`.

Role scoring still runs first when `--gpu-devices` is given, so the primary
device is the decode winner and the cache device is the capacity winner; the
pool fills the primary tier before the cache tier.

### Pinned host expert arena

The page cache is only a hint: the kernel may evict any page without telling the
process, and a long prefill sweep is exactly the kind of read that evicts a hot
decode set. `--expert-arena` replaces that hint with a guarantee — the hottest
routed experts are held in **non-pageable** host memory, so a resident expert is
uploaded with a single DMA and no file read, and the kernel cannot take it back.

```sh
./ds4 --cuda --ssd-streaming --gpu-devices 0,1 \
  --expert-arena 40GB --ctx 32768
```

Omitted, the arena defaults to 7/8 of the host RAM left after reserving a kernel/process headroom of a
few GB (1/16 of physical memory, floored at 2 GB, capped at 8 GB) and subtracting the graph/KV estimate.
Pinned pages are not reclaimable, so that headroom is a requirement, not a preference: the engine must
never take the RAM the kernel needs. `--expert-arena 0` disables it. When the arena is active the
streaming read path drops to O_DIRECT automatically, because a page cache beside a pin would compete for
the same RAM and hand back the prefill pollution the arena exists to remove.

The pin is allocated once at start-up and never grows; it is filled from two
sources. Start-up reads a bounded seed (`--expert-arena-preload`, default half
the arena with an 8 GB floor), then every fetch that misses both the GPU expert
pool and the arena claims a slot and is filed there by the read worker that was
already fetching it, so admission costs no extra read and no extra seek. While
free slots remain every admission is additive with no eviction; once the pin is
full, admission needs a repeat fetch before it can displace the least-used
resident. Filling from the session's own routing beats an offline hotlist: on a
306-token session this reached 11.2 tokens/s with a 6.5 ms host read, against
8.6 tokens/s and 52 ms for a full 40 GB preload — and a 4.7 s start instead of
24 s. `--expert-arena-preload NGB` forces an eager fill when that is what you
want.

"Reserved" and "filled" are different numbers and both are reported. The free
rest is not a fill that stalled: an expert only reaches the arena if it is
missing everywhere else, so on a session the preload and the pool already cover,
the free slots stay free and the pin simply has headroom for the misses that
never came. Size `--expert-arena` for the worst case, not for what a short run
touches.

The dashboard's storage panel reports the arena's residency and split fetches
(`arena_hits` / `arena_misses`), and, when runtime admission is on,
`arena_admissions`, `arena_evictions` and `arena_evicted_fetches` — so eviction
is something you can watch happen rather than assume.

## Flash tensor parallelism

`--cuda-tensor-parallel` pairs GPUs and divides routed-expert work inside
each pair. Pairs also own successive layer ranges. This is an in-process
configuration, not the network `--role coordinator` / `--role worker` mode.

The device order matters. List all layer-home GPUs first, then their partners.
For physical pairs `(0,1)`, `(2,3)`, `(4,5)`, `(6,7)`, use:

```text
0,2,4,6,1,3,5,7
```

Check your machine's topology with `nvidia-smi topo -m`; do not assume this
ordering gives the best pairs on another server. Each pair stores one half
of the routed experts per GPU. Dense attention, routers, and shared experts
are replicated within the pair; the output head is vocabulary-sharded.

For the tested eight-L40S setup:

```sh
./download_model.sh ds4f-q4
./ds4-agent --cuda --cuda-tensor-parallel \
  --gpu-devices 0,2,4,6,1,3,5,7 --gpu-vram auto --ctx 100000
```

Q4 has native grouped routed kernels for multi-user throughput. Q2 needs less
memory, but unsupported grouped shapes use a slower, correct fallback.
Four 48 GB cards with Q2 and eight with Q4 are tested configurations. An even
device count alone does not guarantee memory fit.

## Serve multiple users

```sh
./ds4-server --cuda --cuda-tensor-parallel \
  --gpu-devices 0,2,4,6,1,3,5,7 --gpu-vram auto \
  --ctx 100000 --batched-session 16 --host 0.0.0.0
```

The supported eight-L40S Flash configuration has reached roughly 126 aggregate
generation tokens/s with 16 decode rows. This is total throughput, not the
speed of one user. The recorded conditions and regression requirements are in
[the QA guide](../QA_BEFORE_RELEASES.md).

CUDA TP defaults to 2048-token prefill chunks. Keep that default for the
16-session, 100k-context setup; larger chunks need more scratch memory.
Reduce session count or context if all KV states do not fit.

The repository's [server launcher](../run-nvidia-tp-server.sh) is an example
from the L40S deployment, not a portable default: it uses host-specific model
and cache paths and defaults to MXFP4. The commands above need no launcher or
environment tuning. See [serving](SERVER.md) for disk caches and API access.
