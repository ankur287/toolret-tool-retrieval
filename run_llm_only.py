"""
Condition 1: LLM-only tool selection, no retrieval.

The entire tool catalogue for a slice is placed in the model's context and
the model is asked to rank the 10 most relevant tools for a query. This is
the baseline an agent framework implements when it simply dumps every
available tool into the prompt -- and the condition that breaks first as
the catalogue grows.

Three details matter for the numbers to mean anything:

1. TOOL ORDER IS SHUFFLED. The slice files list ground-truth tools first
   (that is how they are constructed). Presenting them in that order would
   cluster every correct answer at the top of the catalogue and the model
   would score well by position bias alone. We shuffle with a fixed seed
   derived from the slice size, so the order is scrambled but reproducible.

2. LATENCY IS MEASURED IN TWO REGIMES, because they answer different
   questions and differ by more than an order of magnitude.

     raw  -- every request pays the full prefill over the whole catalogue.
             This is what a single agent call actually costs, and it is the
             number to quote when asking "can I afford to do this per
             request?"
     warm -- the catalogue is byte-identical across queries, so it sits in
             the prefix and Ollama serves it from the KV cache. Prefill is
             paid once per slice. This is the number for a service with a
             stable tool catalogue and steady traffic.

   Raw mode is forced with a unique nonce at the very front of the prompt,
   which invalidates the cache from the first tokens on. The catalogue,
   the query position and every other byte stay identical, so accuracy is
   directly comparable between the two regimes -- only the cache behaviour
   changes. (Moving the query to the front would also defeat the cache, but
   it would change the task: the model would know what to look for while
   reading the catalogue.)

   THE QUERY STILL GOES LAST in both modes, so warm mode can cache at all.

3. OVERFLOW IS DETECTED, NOT ASSUMED. Ollama truncates prompts that exceed
   num_ctx by dropping tokens, without raising an error. A slice that does
   not fit would therefore produce plausible-looking but meaningless
   results. We read the true prompt token count back from the server on the
   first call and abort the slice if the catalogue did not survive intact.

Usage
-----
    python run_llm_only.py --limit 3 --slices 1000    # smoke test
    python run_llm_only.py                           # raw for 20/slice, warm rest
    python run_llm_only.py --latency-mode raw        # raw for ALL (very slow)
    python run_llm_only.py --latency-mode warm       # warm only (fastest)

Runtime, 200 queries x 4 runnable slices, measured prefill rate ~265 tok/s:
    warm only            ~2.5 h
    raw sample of 20     ~6 h      (default)
    raw for all          ~44 h
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import re
import statistics
import sys
import time
import uuid
import urllib.error
import urllib.request

from metrics import evaluate

ROOT = pathlib.Path(__file__).parent
OLLAMA = "http://localhost:11434"

TOP_K = 10
# The model's hard context window. This is what decides whether a slice is
# runnable at all, and it is the wall the experiment is measuring.
MODEL_CONTEXT_LIMIT = 131_072
# Ollama silently truncates at num_ctx; keep a margin so generation has room.
SAFETY_MARGIN_TOKENS = 400
# Runtime context is sized to the slice rather than pinned at the model
# maximum: asking for 131k on a 50k prompt makes Ollama allocate and manage
# a KV cache several times larger than needed, which inflates latency for no
# benefit. The headroom below absorbs error in the token estimate; the
# overflow guard in run_slice catches the case where it does not.
CTX_HEADROOM = 1.25
CTX_FLOOR = 4_096

# Room for the query, the response schema and the generated ranking.
PROMPT_OVERHEAD = 600

RANKING_SCHEMA = {
    "type": "object",
    "properties": {
        "tools": {"type": "array", "items": {"type": "integer"}, "maxItems": TOP_K}
    },
    "required": ["tools"],
}


# --------------------------------------------------------------------------
# prompt construction
# --------------------------------------------------------------------------
PREAMBLE = (
    "You are a tool-selection engine. Below is a numbered catalogue of the "
    "tools available to you.\n\nTOOL CATALOGUE:\n"
)

INSTRUCTIONS = (
    "\n\nSelect the {k} tools from the catalogue above that are most relevant "
    "to the task below, ordered from most to least relevant.\n"
    'Respond with JSON only, in the form {{"tools": [<index>, <index>, ...]}}, '
    "using the catalogue index numbers shown in brackets.\n\n"
)


def build_catalogue(tool_ids: list[str], text_by_id: dict[str, str]) -> str:
    """The stable prefix: one numbered line per tool."""
    return "\n".join(f"[{i}] {text_by_id[tid]}" for i, tid in enumerate(tool_ids))


def build_prompt(
    catalogue: str, query: str, instruction: str | None, nonce: str | None = None
) -> str:
    """Catalogue first (cacheable), query last (varies per request).

    `nonce`, when given, is prepended to defeat Ollama's prefix cache so the
    request pays a full prefill. It is a comment line the model ignores; the
    rest of the prompt is unchanged, keeping raw and warm runs comparable.
    """
    head = f"<!-- {nonce} -->\n" if nonce else ""
    task = f"TASK: {query}"
    if instruction:
        task += f"\nWHAT TO LOOK FOR: {instruction}"
    return head + PREAMBLE + catalogue + INSTRUCTIONS.format(k=TOP_K) + task + "\n\nJSON:"


# --------------------------------------------------------------------------
# Ollama
# --------------------------------------------------------------------------
def size_context(est_prompt_tokens: float) -> int:
    """Smallest power-of-two-ish context that comfortably holds the prompt."""
    want = int(est_prompt_tokens * CTX_HEADROOM) + CTX_FLOOR
    return max(CTX_FLOOR, min(MODEL_CONTEXT_LIMIT, want))


def ollama_generate(
    model: str, prompt: str, num_ctx: int, *, timeout: int = 3600
) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": RANKING_SCHEMA,
        "keep_alive": "30m",
        "options": {
            "num_ctx": num_ctx,
            "num_predict": 256,
            "temperature": 0.0,
            "seed": 13,
        },
    }
    req = urllib.request.Request(
        f"{OLLAMA}/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def check_server(model: str) -> None:
    try:
        with urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=10) as r:
            tags = json.load(r)
    except urllib.error.URLError:
        sys.exit(f"cannot reach Ollama at {OLLAMA} -- is `ollama serve` running?")
    names = {m["name"] for m in tags.get("models", [])}
    if model not in names:
        sys.exit(f"model '{model}' not found. Available: {sorted(names)}\n"
                 f"Pull it with:  ollama pull {model}")


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------
def parse_indices(raw: str, n_tools: int) -> list[int]:
    """Extract ranked catalogue indices from the model's reply.

    The schema-constrained decode should always give valid JSON, but a
    regex fallback keeps one malformed reply from costing us a whole slice.
    Out-of-range and duplicate indices are dropped rather than repaired --
    a hallucinated index is a wrong answer, not a parsing problem.
    """
    idxs: list[int] = []
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            idxs = [int(i) for i in obj.get("tools", []) if isinstance(i, (int, float))]
    except (json.JSONDecodeError, TypeError, ValueError):
        idxs = [int(m) for m in re.findall(r"-?\d+", raw)]

    seen, out = set(), []
    for i in idxs:
        if 0 <= i < n_tools and i not in seen:
            seen.add(i)
            out.append(i)
    return out[:TOP_K]


# --------------------------------------------------------------------------
# run one slice
# --------------------------------------------------------------------------
def run_slice(
    model: str,
    size: int,
    tool_ids: list[str],
    text_by_id: dict[str, str],
    queries: list[dict],
    out_path: pathlib.Path,
    use_instruction: bool,
    latency_mode: str,
    raw_sample: int,
    num_ctx: int,
) -> tuple[list[dict], dict | None]:
    """Evaluate every query against one corpus slice.

    Returns (per-query records, None) on success, or ([], reason) if the
    slice could not be run.
    """
    # Shuffle so ground-truth tools are not clustered at the front (see
    # module docstring). Seeded by size: same slice, same order, every run.
    presented = list(tool_ids)
    random.Random(1000 + size).shuffle(presented)

    catalogue = build_catalogue(presented, text_by_id)

    # Resume: skip queries already recorded for this slice.
    done: dict[str, dict] = {}
    if out_path.exists():
        for line in out_path.open():
            rec = json.loads(line)
            if rec["slice"] == size:
                done[rec["query_id"]] = rec
    if done:
        print(f"    resuming: {len(done)} queries already recorded")

    records = list(done.values())
    pending = [q for q in queries if q["id"] not in done]
    if not pending:
        return records, None

    fh = out_path.open("a")
    checked_overflow = False

    for n, q in enumerate(pending):
        # In "both" mode the first `raw_sample` queries are measured raw and
        # the rest warm: accuracy still comes from every query, while the
        # expensive uncached timing is sampled rather than paid 200 times.
        raw = latency_mode == "raw" or (latency_mode == "both" and n < raw_sample)
        nonce = uuid.uuid4().hex if raw else None

        prompt = build_prompt(
            catalogue, q["query"], q["instruction"] if use_instruction else None, nonce
        )
        t0 = time.perf_counter()
        try:
            resp = ollama_generate(model, prompt, num_ctx)
        except Exception as exc:  # noqa: BLE001 - surface and keep going
            print(f"    [{n+1}/{len(pending)}] {q['id']}: request failed: {exc}")
            continue
        elapsed = time.perf_counter() - t0

        prompt_tokens = resp.get("prompt_eval_count") or 0

        # Overflow guard. Only meaningful on a full prefill, where Ollama
        # reports the true prompt length; a cache hit reports far fewer.
        if not checked_overflow and (raw or n == 0):
            checked_overflow = True
            if prompt_tokens >= num_ctx - SAFETY_MARGIN_TOKENS:
                fh.close()
                over_model = prompt_tokens >= MODEL_CONTEXT_LIMIT - SAFETY_MARGIN_TOKENS
                return [], {
                    "reason": "context_overflow" if over_model else "undersized_ctx",
                    "prompt_tokens": prompt_tokens,
                    "num_ctx": num_ctx,
                    "detail": (
                        f"catalogue of {size} tools needs >= {prompt_tokens:,} tokens "
                        f"against the model's {MODEL_CONTEXT_LIMIT:,}-token window; "
                        "Ollama would truncate silently, so this slice is not "
                        "evaluated"
                    ) if over_model else (
                        f"token estimate was too low: prompt is {prompt_tokens:,} "
                        f"tokens but num_ctx was sized to {num_ctx:,}. Raise "
                        "CTX_HEADROOM and re-run this slice."
                    ),
                }
            print(f"    prompt = {prompt_tokens:,} tokens "
                  f"(num_ctx {num_ctx:,}, "
                  f"{prompt_tokens / MODEL_CONTEXT_LIMIT:.0%} of the model window), "
                  f"first call {elapsed:.1f}s")

        idxs = parse_indices(resp.get("response", ""), len(presented))
        ranked_ids = [presented[i] for i in idxs]

        # Ollama reports nanoseconds. prompt_eval_duration is the prefill and
        # collapses to ~0 on a cache hit; eval_duration is generation.
        prefill_s = (resp.get("prompt_eval_duration") or 0) / 1e9
        decode_s = (resp.get("eval_duration") or 0) / 1e9

        rec = {
            "slice": size,
            "query_id": q["id"],
            "source": q["source"],
            "mode": "raw" if raw else "warm",
            "ranked_ids": ranked_ids,
            "n_returned": len(ranked_ids),
            "latency_s": round(elapsed, 3),
            "prefill_s": round(prefill_s, 3),
            "decode_s": round(decode_s, 3),
            "prompt_tokens": prompt_tokens,
            "eval_tokens": resp.get("eval_count") or 0,
        }
        records.append(rec)
        fh.write(json.dumps(rec) + "\n")
        fh.flush()

        if (n + 1) % 10 == 0 or n + 1 == len(pending):
            print(f"    [{n+1}/{len(pending)}] last={elapsed:.1f}s ({rec['mode']})")

    fh.close()
    return records, None


def latency_stats(records: list[dict], mode: str) -> dict | None:
    """p50/p90 total latency and the prefill/decode split for one regime."""
    rs = [r for r in records if r.get("mode") == mode]
    if not rs:
        return None
    lat = sorted(r["latency_s"] for r in rs)
    return {
        "n": len(rs),
        "p50_s": round(lat[len(lat) // 2], 2),
        "p90_s": round(lat[min(int(0.9 * len(lat)), len(lat) - 1)], 2),
        "mean_prefill_s": round(statistics.mean(r["prefill_s"] for r in rs), 2),
        "mean_decode_s": round(statistics.mean(r["decode_s"] for r in rs), 2),
    }


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--category", default="code")
    ap.add_argument("--model", default="gemma3:12b")
    ap.add_argument("--slices", type=int, nargs="+", default=None,
                    help="slice sizes to run (default: all in the manifest)")
    ap.add_argument("--limit", type=int, default=None,
                    help="only run the first N queries (smoke test)")
    ap.add_argument("--no-instruction", action="store_true",
                    help="omit ToolRet's per-query instruction field")
    ap.add_argument("--latency-mode", choices=["warm", "raw", "both"], default="both",
                    help="warm: cache the catalogue prefix (amortized cost). "
                         "raw: full prefill on every query (true single-call "
                         "cost, ~44h for the full grid). both: raw for the "
                         "first --raw-sample queries, warm for the rest.")
    ap.add_argument("--raw-sample", type=int, default=20,
                    help="queries per slice measured raw when --latency-mode both")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = ROOT / "data" / "eval" / args.category
    if not data.exists():
        sys.exit(f"no dataset at {data} -- run build_dataset.py first")
    check_server(args.model)

    manifest = json.loads((data / "manifest.json").read_text())
    queries = [json.loads(l) for l in (data / "queries.jsonl").open()]
    corpus = [json.loads(l) for l in (data / "corpus.jsonl").open()]
    text_by_id = {c["id"]: c["compact"] for c in corpus}
    qrels = json.loads((data / "qrels.json").read_text())

    if args.limit:
        queries = queries[: args.limit]
        qrels = {q["id"]: qrels[q["id"]] for q in queries}

    out_dir = pathlib.Path(args.out) if args.out else ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    tag = f"llm_only_{args.category}_{args.model.replace(':', '-')}"
    raw_path = out_dir / f"{tag}.jsonl"

    sizes = args.slices or [s["size"] for s in manifest["slices"]]
    print(f"LLM-only | model={args.model} category={args.category} "
          f"queries={len(queries)} slices={sizes}")
    print(f"instruction field: {'omitted' if args.no_instruction else 'included'}\n")

    summary = []
    for size in sizes:
        meta = next((s for s in manifest["slices"] if s["size"] == size), None)
        if meta is None:
            print(f"  slice {size}: not in manifest, skipping")
            continue

        print(f"  slice {size:,} ({meta['n_distractors']:,} distractors, "
              f"est. {meta['compact_tokens']:,} tokens)")

        # Cheap pre-check from the manifest; the authoritative check is the
        # token count Ollama reports on the first call inside run_slice.
        if not meta["compact_fits_context"]:
            print(f"    SKIPPED: estimated {meta['compact_tokens']:,} tokens "
                  f"exceeds the {MODEL_CONTEXT_LIMIT:,}-token window\n")
            summary.append({
                "slice": size, "status": "exceeds_context",
                "est_tokens": meta["compact_tokens"], **{f"ndcg@{TOP_K}": None},
            })
            continue

        tool_ids = json.loads((data / "slices" / f"slice_{size}.json").read_text())
        num_ctx = size_context(meta["compact_tokens"] + PROMPT_OVERHEAD)
        print(f"    num_ctx sized to {num_ctx:,} "
              f"(model max {MODEL_CONTEXT_LIMIT:,})")
        records, failure = run_slice(
            args.model, size, tool_ids, text_by_id, queries, raw_path,
            use_instruction=not args.no_instruction,
            latency_mode=args.latency_mode, raw_sample=args.raw_sample,
            num_ctx=num_ctx,
        )

        if failure:
            print(f"    SKIPPED: {failure['detail']}\n")
            summary.append({"slice": size, "status": "exceeds_context",
                            "prompt_tokens": failure["prompt_tokens"]})
            continue

        run = {r["query_id"]: r["ranked_ids"] for r in records}
        scores = evaluate(run, qrels, k=TOP_K)

        raw_lat = latency_stats(records, "raw")
        warm_lat = latency_stats(records, "warm")
        row = {
            "slice": size,
            "status": "ok",
            "n_distractors": meta["n_distractors"],
            "num_ctx": num_ctx,
            "prompt_tokens": max((r["prompt_tokens"] for r in records), default=None),
            **{k: round(v, 4) if isinstance(v, float) else v
               for k, v in scores.items()},
            "latency_raw": raw_lat,
            "latency_warm": warm_lat,
        }
        summary.append(row)
        lat_txt = " ".join(
            f"{name}p50={st['p50_s']}s(n={st['n']})"
            for name, st in (("raw ", raw_lat), ("warm ", warm_lat)) if st
        )
        print(f"    nDCG@{TOP_K}={row[f'ndcg@{TOP_K}']:.4f}  "
              f"Recall@{TOP_K}={row[f'recall@{TOP_K}']:.4f}  "
              f"Complete@{TOP_K}={row[f'completeness@{TOP_K}']:.4f}  "
              f"{lat_txt}\n")

    # ----------------------------------------------------------------------
    summary_path = out_dir / f"{tag}_summary.json"
    summary_path.write_text(json.dumps(
        {"condition": "llm_only", "model": args.model, "category": args.category,
         "n_queries": len(queries), "top_k": TOP_K,
         "use_instruction": not args.no_instruction,
         "latency_mode": args.latency_mode, "raw_sample": args.raw_sample,
         "results": summary},
        indent=2))

    w = 96
    print("=" * w)
    print(f"{'slice':>7} {'tokens':>9} {'nDCG@10':>9} {'Recall@10':>10} "
          f"{'Compl@10':>9} | {'raw p50':>9} {'raw p90':>9} | "
          f"{'warm p50':>9} {'warm p90':>9}")
    print("-" * w)
    for r in summary:
        if r.get("status") != "ok":
            print(f"{r['slice']:>7,} {'--':>9}   exceeds context window "
                  f"-- not evaluated")
            continue
        rl, wl = r["latency_raw"], r["latency_warm"]
        fmt = lambda st, k: f"{st[k]:>9.2f}" if st else f"{'--':>9}"  # noqa: E731
        print(f"{r['slice']:>7,} {r['prompt_tokens']:>9,} "
              f"{r['ndcg@10']:>9.4f} {r['recall@10']:>10.4f} "
              f"{r['completeness@10']:>9.4f} | "
              f"{fmt(rl,'p50_s')} {fmt(rl,'p90_s')} | "
              f"{fmt(wl,'p50_s')} {fmt(wl,'p90_s')}")
    print("=" * w)
    print("raw  = full prefill every query (true cost of one agent call)")
    print("warm = catalogue served from KV cache (cost under steady traffic)")
    print(f"\nper-query records -> {raw_path}")
    print(f"summary           -> {summary_path}")


if __name__ == "__main__":
    main()
