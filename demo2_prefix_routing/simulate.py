#!/usr/bin/env python3
"""Prefix-cache routing SIMULATION (no GPU, no network, standard library only).

What it models
  * R model-server replicas, each with a fixed-size, block-based prefix cache
    evicted LRU (the same idea as vLLM automatic prefix caching: a cached block
    is only reusable if every block before it also matches).
  * A workload of prompts = <shared static prefix for a tenant> + <unique suffix>.
  * Routing policies: random, round_robin, least_loaded, prefix_affinity,
    prefix_and_load (cache match score, with a load guard).

What it reports (all are COUNTS produced by this simulation, not latency):
  * token-level prefix-cache hit rate
  * prefill tokens that had to be recomputed
  * load imbalance (busiest replica / mean)

It deliberately reports no milliseconds: real latency depends on your model,
GPU, batch size and server version. Measure that on real hardware.
"""
from __future__ import annotations
import argparse, hashlib, random
from collections import OrderedDict

BLOCK = 16  # tokens per cache block (vLLM hashes fixed-size token blocks)


class Replica:
    def __init__(self, rid: int, capacity_blocks: int):
        self.rid = rid
        self.capacity = capacity_blocks
        self.cache: OrderedDict[str, None] = OrderedDict()  # block-hash -> LRU order
        self.inflight = 0.0   # decaying load estimate
        self.served = 0

    def match_blocks(self, hashes: list[str]) -> int:
        n = 0
        for h in hashes:
            if h in self.cache:
                n += 1
            else:
                break            # prefix property: stop at first miss
        return n

    def serve(self, hashes: list[str]) -> int:
        hit = self.match_blocks(hashes)
        for h in hashes:         # touch / insert every block of this prompt
            if h in self.cache:
                self.cache.move_to_end(h)
            else:
                self.cache[h] = None
                if len(self.cache) > self.capacity:
                    self.cache.popitem(last=False)
        self.served += 1
        self.inflight += 1.0
        return hit


def block_hashes(prefix_id: int, prefix_tokens: int, req_id: int, suffix_tokens: int) -> list[str]:
    """Chained hashes: block i's hash depends on all earlier blocks."""
    out, h = [], hashlib.sha1(b"root")
    for i in range(prefix_tokens // BLOCK):
        h = hashlib.sha1(h.digest() + f"p{prefix_id}:{i}".encode())
        out.append(h.hexdigest())
    for i in range(suffix_tokens // BLOCK):
        h = hashlib.sha1(h.digest() + f"r{req_id}:{i}".encode())
        out.append(h.hexdigest())
    return out


def route(policy: str, replicas: list[Replica], hashes: list[str], prefix_id: int,
          rr_state: list[int], rng: random.Random, load_guard: float) -> Replica:
    if policy == "random":
        return rng.choice(replicas)
    if policy == "round_robin":
        rr_state[0] = (rr_state[0] + 1) % len(replicas)
        return replicas[rr_state[0]]
    if policy == "least_loaded":
        return min(replicas, key=lambda r: (r.inflight, r.rid))
    if policy == "prefix_affinity":          # hash the first block -> sticky replica
        return replicas[int(hashes[0][:8], 16) % len(replicas)]
    if policy == "prefix_and_load":          # longest cached prefix, unless overloaded
        mean = sum(r.inflight for r in replicas) / len(replicas)
        ok = [r for r in replicas if r.inflight <= mean * load_guard + 1.0]
        return max(ok, key=lambda r: (r.match_blocks(hashes), -r.inflight, -r.rid))
    raise ValueError(policy)


def run(policy: str, a) -> dict:
    rng = random.Random(a.seed)
    replicas = [Replica(i, a.cache_blocks) for i in range(a.replicas)]
    weights = [1.0 / (k + 1) ** a.zipf for k in range(a.prefixes)]  # skewed tenant mix
    rr = [-1]
    total_blocks = hit_blocks = 0
    for req in range(a.requests):
        pid = rng.choices(range(a.prefixes), weights)[0]
        hashes = block_hashes(pid, a.prefix_tokens, req, a.suffix_tokens)
        r = route(policy, replicas, hashes, pid, rr, rng, a.load_guard)
        hit_blocks += r.serve(hashes)
        total_blocks += len(hashes)
        for x in replicas:                   # requests finish over time
            x.inflight *= a.decay
    served = [r.served for r in replicas]
    return {
        "policy": policy,
        "hit_rate": hit_blocks / total_blocks,
        "recomputed_tokens": (total_blocks - hit_blocks) * BLOCK,
        "imbalance": max(served) / (sum(served) / len(served)),
        "served": served,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--replicas", type=int, default=3)
    p.add_argument("--prefixes", type=int, default=12, help="distinct shared prefixes (tenants/agents)")
    p.add_argument("--prefix-tokens", type=int, default=4000)
    p.add_argument("--suffix-tokens", type=int, default=208)
    p.add_argument("--requests", type=int, default=6000)
    p.add_argument("--cache-blocks", type=int, default=1250,
                   help="per-replica cache size in blocks (default fits ~5 prefixes)")
    p.add_argument("--zipf", type=float, default=0.8, help="tenant popularity skew (0 = uniform)")
    p.add_argument("--load-guard", type=float, default=1.25)
    p.add_argument("--decay", type=float, default=0.90)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()

    print("SIMULATION - counts only, not a latency benchmark")
    print(f"replicas={a.replicas} prefixes={a.prefixes} prefix_tokens={a.prefix_tokens} "
          f"suffix_tokens={a.suffix_tokens} requests={a.requests} "
          f"cache_blocks/replica={a.cache_blocks} zipf={a.zipf} seed={a.seed}\n")
    print(f"{'policy':<17}{'cache hit rate':>15}{'recomputed prefill tokens':>28}{'load imbalance':>16}   per-replica requests")
    for pol in ["random", "round_robin", "least_loaded", "prefix_affinity", "prefix_and_load"]:
        r = run(pol, a)
        print(f"{r['policy']:<17}{r['hit_rate']*100:>14.1f}%{r['recomputed_tokens']:>28,}{r['imbalance']:>15.2f}x   {r['served']}")
    print("\nhit rate   = cached prefix tokens / all prompt tokens")
    print("imbalance  = busiest replica's request count / mean (1.00x = perfectly even)")


if __name__ == "__main__":
    main()
