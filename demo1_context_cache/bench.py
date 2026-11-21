#!/usr/bin/env python3
"""Demo 1 - Gemini context caching: baseline vs explicit cache, measured.

Asks the same questions about one large document two ways:
  baseline : document + system instruction sent on every request
  explicit : document + system instruction stored once in a context cache,
             each request sends only the question and references the cache

Every number printed is measured from the API response (usage_metadata) or a
local clock. Nothing is assumed: note that the *baseline* can also report
cached tokens, because implicit caching is on by default.

Auth assumptions (Application Default Credentials):
  gcloud auth application-default login
  export GOOGLE_CLOUD_PROJECT=<your-project-id>
  export GOOGLE_CLOUD_LOCATION=global
  export GOOGLE_GENAI_USE_ENTERPRISE=True   # older SDKs: GOOGLE_GENAI_USE_VERTEXAI=True

Usage:
  python bench.py                       # both modes, 5 questions
  python bench.py --mode explicit --ttl 600s
  python bench.py --doc-uri gs://my-bucket/my-doc.pdf
"""
from __future__ import annotations
import argparse, statistics, sys, time

from google import genai
from google.genai import errors
from google.genai.types import (
    Content, CreateCachedContentConfig, GenerateContentConfig, HttpOptions, Part,
)

# ---- Pricing: USD per 1M tokens, Gemini 3.5 Flash, global endpoint, standard tier.
# Source: https://cloud.google.com/vertex-ai/generative-ai/pricing (checked 2026-11-14).
# RE-CHECK BEFORE THE TALK and edit here if you change --model.
PRICE = {"input": 1.50, "cached_input": 0.15, "output": 9.00, "storage_per_hour": 1.00}

DEFAULT_MODEL = "gemini-3.5-flash"
DEFAULT_DOC = "gs://cloud-samples-data/generative-ai/pdf/2312.11805v3.pdf"  # Google's public sample PDF

SYSTEM_INSTRUCTION = (
    "You are a precise technical analyst. Answer only from the attached document. "
    "If the document does not contain the answer, say so. Keep answers under 120 words."
)

QUESTIONS = [
    "What problem does this document set out to solve?",
    "List the three most important technical claims.",
    "Which evaluation results does the document report, and on what tasks?",
    "What limitations do the authors acknowledge?",
    "Summarize the document for a backend engineer in four sentences.",
]


def ask(client, model, contents, config):
    """Streams one request. Returns measured timings and the usage metadata."""
    t0 = time.perf_counter()
    ttft, usage = None, None
    for chunk in client.models.generate_content_stream(model=model, contents=contents, config=config):
        if ttft is None and chunk.text:
            ttft = time.perf_counter() - t0
        if chunk.usage_metadata:
            usage = chunk.usage_metadata          # final chunk carries the totals
    total = time.perf_counter() - t0
    return {
        "ttft_s": ttft, "total_s": total,
        "prompt": (usage.prompt_token_count or 0) if usage else 0,
        "cached": (usage.cached_content_token_count or 0) if usage else 0,
        "output": ((usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)) if usage else 0,
    }


def cost(row):
    fresh = row["prompt"] - row["cached"]        # prompt_token_count INCLUDES cached tokens
    return (fresh * PRICE["input"] + row["cached"] * PRICE["cached_input"]
            + row["output"] * PRICE["output"]) / 1e6


def report(label, rows):
    print(f"\n== {label} ==")
    print(f"{'#':<3}{'prompt tok':>11}{'cached tok':>11}{'output tok':>11}{'TTFT s':>9}{'total s':>9}{'est. USD':>11}")
    for i, r in enumerate(rows, 1):
        ttft = f"{r['ttft_s']:.2f}" if r["ttft_s"] is not None else "n/a"
        print(f"{i:<3}{r['prompt']:>11,}{r['cached']:>11,}{r['output']:>11,}{ttft:>9}{r['total_s']:>9.2f}{cost(r):>11.5f}")
    ttfts = [r["ttft_s"] for r in rows if r["ttft_s"] is not None]
    print(f"median TTFT {statistics.median(ttfts):.2f}s | median total "
          f"{statistics.median(r['total_s'] for r in rows):.2f}s | "
          f"cached share {sum(r['cached'] for r in rows) / max(1, sum(r['prompt'] for r in rows)):.0%} | "
          f"est. total ${sum(cost(r) for r in rows):.5f}")


def run_baseline(client, a, doc):
    cfg = GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION)
    return [ask(client, a.model, [doc, q], cfg) for q in QUESTIONS[: a.n]]


def run_explicit(client, a, doc):
    cache = None
    try:
        cache = client.caches.create(
            model=a.model,
            config=CreateCachedContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,          # static -> cached
                contents=[Content(role="user", parts=[doc])],   # static -> cached
                display_name="devfest-demo-cache",
                ttl=a.ttl,
            ),
        )
        tokens = cache.usage_metadata.total_token_count
        print(f"\ncache created: {cache.name}")
        print(f"cached tokens: {tokens:,} | expires: {cache.expire_time}")
        print(f"one-time creation est. ${tokens * PRICE['input'] / 1e6:.5f} (standard input rate) | "
              f"storage est. ${tokens * PRICE['storage_per_hour'] / 1e6:.5f} per hour")
        # The request carries ONLY the dynamic part. System instruction and
        # tools must not be repeated when a cache is referenced.
        cfg = GenerateContentConfig(cached_content=cache.name)
        return [ask(client, a.model, q, cfg) for q in QUESTIONS[: a.n]]
    finally:
        if cache is not None and not a.keep_cache:
            client.caches.delete(name=cache.name)
            print(f"cache deleted: {cache.name}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["baseline", "explicit", "both"], default="both")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--doc-uri", default=DEFAULT_DOC)
    p.add_argument("--mime-type", default="application/pdf")
    p.add_argument("--ttl", default="900s", help="cache lifetime, e.g. 900s (minimum 60s)")
    p.add_argument("-n", type=int, default=5, help="number of questions (max %d)" % len(QUESTIONS))
    p.add_argument("--keep-cache", action="store_true", help="do not delete the cache at exit")
    a = p.parse_args()

    client = genai.Client(http_options=HttpOptions(api_version="v1"))
    doc = Part.from_uri(file_uri=a.doc_uri, mime_type=a.mime_type)
    print(f"model={a.model} doc={a.doc_uri} questions={a.n}")
    try:
        if a.mode in ("baseline", "both"):
            report("BASELINE (document sent every request)", run_baseline(client, a, doc))
        if a.mode in ("explicit", "both"):
            report("EXPLICIT CONTEXT CACHE", run_explicit(client, a, doc))
    except errors.APIError as e:
        print(f"\nAPI error {e.code}: {e.message}", file=sys.stderr)
        print("Check: project/location env vars, API enabled, model supports explicit caching, "
              "document meets the model's minimum cacheable token count.", file=sys.stderr)
        return 1
    print("\nNotes: est. USD excludes cache storage and creation (printed above). "
          "A non-zero 'cached tok' in BASELINE is implicit caching at work.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
