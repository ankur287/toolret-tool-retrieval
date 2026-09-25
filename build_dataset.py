"""
Build a corpus-scaling evaluation set from ToolRet.

Design
------
One ToolRet category (default: `code`) provides a self-contained corpus --
every ground-truth tool for that category's queries lives inside that
category, so the retrieval task is well-posed at any corpus size.

We sample N queries (stratified across the category's source datasets) and
build nested corpus slices of increasing size:

    slice_k  =  ALL ground-truth tools  +  first (k - |GT|) distractors

Two invariants make the size sweep interpretable:
  1. Every slice contains every ground-truth tool, so a query is always
     answerable and differences across slices reflect difficulty, not
     missing labels.
  2. Slices nest (slice_1000 subset-of slice_2000 subset-of ...), because
     distractors are drawn as a prefix of one fixed shuffled order. The
     only variable across slices is how many distractors were added.

Usage
-----
    python build_dataset.py                       # defaults
    python build_dataset.py --category customized --n-queries 300
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import random
import statistics

ROOT = pathlib.Path(__file__).parent
RAW = ROOT / "data" / "raw"

# gemma3:12b. chars/token measured empirically against the Ollama tokenizer
# on real ToolRet tool lists (see README); context is the model's hard limit.
CHARS_PER_TOKEN = 3.79
CONTEXT_LIMIT = 131_072
# Reserve for system prompt, the query itself, and room to generate a ranking.
PROMPT_OVERHEAD_TOKENS = 600
INDEX_PREFIX_TOKENS = 4  # the "[i] " numbering we add to each line

DESC_TRUNC = 180  # chars; how much tool description survives into compact form


# --------------------------------------------------------------------------
# tool rendering
# --------------------------------------------------------------------------
def parse_doc(documentation: str) -> dict:
    try:
        obj = json.loads(documentation)
        return obj if isinstance(obj, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def tool_name(doc: dict, fallback: str) -> str:
    for key in ("name", "tool_name", "api_name", "function_name"):
        if doc.get(key):
            return str(doc[key])
    return fallback


def tool_description(doc: dict) -> str:
    for key in ("description", "desc", "functionality"):
        if doc.get(key):
            return " ".join(str(doc[key]).split())
    return ""


def compact_text(tool: dict) -> str:
    """`name: description` -- what the LLM and the retrievers both see.

    Parameter schemas are dropped: they matter for *calling* a tool, not for
    *picking* one, and they dominate the token budget.
    """
    doc = parse_doc(tool["documentation"])
    name = tool_name(doc, tool["id"])
    desc = tool_description(doc)[:DESC_TRUNC]
    return f"{name}: {desc}" if desc else name


def full_text(tool: dict) -> str:
    """The complete tool document, as ToolRet ships it."""
    return tool["documentation"]


def est_tokens(text: str, *, indexed: bool = True) -> float:
    return len(text) / CHARS_PER_TOKEN + (INDEX_PREFIX_TOKENS if indexed else 0)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load_jsonl(path: pathlib.Path) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh]


def load_category(category: str) -> tuple[list[dict], list[dict]]:
    tools = [t for t in load_jsonl(RAW / "tools.jsonl") if t["corpus"] == category]
    queries = [q for q in load_jsonl(RAW / "queries.jsonl") if q["category"] == category]
    for q in queries:
        q["gt_ids"] = sorted({lb["id"] for lb in json.loads(q["labels"])})
    return tools, queries


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------
def stratified_sample(queries: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Sample ~n queries proportionally across source datasets.

    Proportional allocation with a floor of 1 keeps small sources (gta has 14
    queries) represented while letting large ones dominate as they do in the
    benchmark.
    """
    by_source = collections.defaultdict(list)
    for q in queries:
        by_source[q["source"]].append(q)

    total = len(queries)
    quota = {s: max(1, round(n * len(v) / total)) for s, v in by_source.items()}

    # Correct rounding drift so we land on n.
    while (drift := sum(quota.values()) - n) != 0:
        step = -1 if drift > 0 else 1
        # Adjust the largest sources first; never drop a source below 1.
        for s in sorted(quota, key=lambda s: -quota[s]):
            if step < 0 and quota[s] <= 1:
                continue
            if quota[s] + step <= len(by_source[s]):
                quota[s] += step
                break
        else:
            break

    sample = []
    for source, qs in sorted(by_source.items()):
        sample.extend(rng.sample(qs, min(quota[source], len(qs))))
    rng.shuffle(sample)
    return sample


def build_slices(
    gt_ids: set[str], all_ids: list[str], sizes: list[int], rng: random.Random
) -> dict[int, list[str]]:
    """Nested slices, each containing every ground-truth tool.

    Distractors come from one fixed shuffled order, so slice_k is always a
    prefix-extension of the smaller slices.
    """
    distractors = [i for i in all_ids if i not in gt_ids]
    rng.shuffle(distractors)
    gt_sorted = sorted(gt_ids)

    slices = {}
    for size in sizes:
        if size < len(gt_sorted):
            raise ValueError(
                f"slice size {size} < {len(gt_sorted)} ground-truth tools; "
                "raise the slice size or lower --n-queries"
            )
        slices[size] = gt_sorted + distractors[: size - len(gt_sorted)]
    return slices


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--category", default="code", choices=["code", "customized", "web"])
    ap.add_argument("--n-queries", type=int, default=200)
    ap.add_argument("--slices", type=int, nargs="+",
                    default=[1000, 1500, 2000, 2500, 0],
                    help="slice sizes; 0 means the full category corpus")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = pathlib.Path(args.out) if args.out else ROOT / "data" / "eval" / args.category
    out.mkdir(parents=True, exist_ok=True)

    tools, queries = load_category(args.category)
    tool_by_id = {t["id"]: t for t in tools}
    print(f"category '{args.category}': {len(tools):,} tools, {len(queries):,} queries")

    # --- sanity: the category must be self-contained -----------------------
    orphans = {g for q in queries for g in q["gt_ids"] if g not in tool_by_id}
    if orphans:
        raise SystemExit(f"{len(orphans)} ground-truth ids missing from corpus")
    print("  ground-truth containment: OK (all labels resolve in-category)")

    # --- sample queries ----------------------------------------------------
    sample = stratified_sample(queries, args.n_queries, rng)
    gt_ids = {g for q in sample for g in q["gt_ids"]}
    per_query = [len(q["gt_ids"]) for q in sample]
    print(f"  sampled {len(sample)} queries -> {len(gt_ids):,} unique ground-truth tools")
    print(f"  labels/query: mean {statistics.mean(per_query):.2f}, max {max(per_query)}")

    # --- slices ------------------------------------------------------------
    sizes = sorted({len(tools) if s == 0 else s for s in args.slices})
    slices = build_slices(gt_ids, [t["id"] for t in tools], sizes, rng)

    # --- token budget per slice -------------------------------------------
    compact_tok = {t["id"]: est_tokens(compact_text(t)) for t in tools}
    full_tok = {t["id"]: est_tokens(full_text(t)) for t in tools}
    budget = CONTEXT_LIMIT - PROMPT_OVERHEAD_TOKENS

    manifest = {
        "category": args.category,
        "seed": args.seed,
        "n_queries": len(sample),
        "n_gt_tools": len(gt_ids),
        "corpus_size": len(tools),
        "context_limit": CONTEXT_LIMIT,
        "prompt_overhead_tokens": PROMPT_OVERHEAD_TOKENS,
        "chars_per_token": CHARS_PER_TOKEN,
        "slices": [],
    }

    print(f"\n  {'slice':>7} {'GT':>5} {'distr':>7} {'GT%':>6} "
          f"{'compact tok':>12} {'fits':>5} {'full tok':>10} {'fits':>5}")
    for size in sizes:
        ids = slices[size]
        c_tok = sum(compact_tok[i] for i in ids)
        f_tok = sum(full_tok[i] for i in ids)
        c_fits, f_fits = c_tok <= budget, f_tok <= budget
        n_dist = size - len(gt_ids)
        print(f"  {size:>7,} {len(gt_ids):>5,} {n_dist:>7,} {len(gt_ids)/size:>5.1%} "
              f"{c_tok:>12,.0f} {'yes' if c_fits else 'NO':>5} "
              f"{f_tok:>10,.0f} {'yes' if f_fits else 'NO':>5}")

        (out / "slices").mkdir(exist_ok=True)
        (out / "slices" / f"slice_{size}.json").write_text(json.dumps(ids))
        manifest["slices"].append({
            "size": size, "n_gt": len(gt_ids), "n_distractors": n_dist,
            "gt_density": round(len(gt_ids) / size, 4),
            "compact_tokens": round(c_tok), "compact_fits_context": c_fits,
            "full_tokens": round(f_tok), "full_fits_context": f_fits,
            "path": f"slices/slice_{size}.json",
        })

    # verify nesting
    for a, b in zip(sizes, sizes[1:]):
        assert set(slices[a]) <= set(slices[b]), f"slice {a} not nested in {b}"
    print("  nesting verified: every slice is a subset of the next")

    # --- write -------------------------------------------------------------
    with open(out / "queries.jsonl", "w") as fh:
        for q in sample:
            fh.write(json.dumps({
                "id": q["id"], "query": q["query"], "instruction": q["instruction"],
                "source": q["source"], "gt_ids": q["gt_ids"],
            }, ensure_ascii=False) + "\n")

    with open(out / "corpus.jsonl", "w") as fh:
        for t in tools:
            fh.write(json.dumps({
                "id": t["id"],
                "compact": compact_text(t),
                "full": full_text(t),
            }, ensure_ascii=False) + "\n")

    # qrels: binary relevance, the format the nDCG@10 scorer consumes
    qrels = {q["id"]: {g: 1 for g in q["gt_ids"]} for q in sample}
    (out / "qrels.json").write_text(json.dumps(qrels, indent=2))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nwrote -> {out}")
    for f in ("queries.jsonl", "corpus.jsonl", "qrels.json", "manifest.json"):
        print(f"  {f}")
    print(f"  slices/ ({len(sizes)} files)")


if __name__ == "__main__":
    main()
