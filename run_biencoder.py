"""
Condition 2: one-stage dense retrieval with a bi-encoder.

Tools and queries are embedded independently into the same vector space and
ranked by cosine similarity. Unlike the LLM condition, nothing has to fit in
a context window -- the corpus lives in a matrix, so corpus size is bounded
by memory rather than by attention.

Three details matter for the comparison to be fair:

1. THE CORPUS IS ENCODED ONCE, NOT PER SLICE. A tool's embedding does not
   depend on which other tools are in the index, so we embed all 3,794 tools
   and then restrict the similarity search to each slice's rows. Re-encoding
   per slice would produce identical vectors and inflate the reported index
   build time fivefold.

2. QUERIES ARE ENCODED ONE AT A TIME. Batching is far faster in throughput
   but it is not what a live agent does -- it has one query and needs one
   answer. Per-query latency here is directly comparable to the LLM
   condition's per-call latency.

3. THE SAME TEXT THE LLM SAW. By default tools are represented by the same
   compact `name: description` string the LLM condition ranks over, so any
   difference in score comes from the method and not from one side getting
   richer input. `--doc-field full` indexes the complete tool document
   instead, which is retrieval's structural advantage: it has no context
   budget to blow.

Embedding models are prompt-sensitive. BGE and E5 were both trained with
asymmetric prefixes, and omitting them costs several points of nDCG, so the
right prefix is applied per model family rather than left to the caller.

Usage
-----
    python run_biencoder.py                                  # bge-large
    python run_biencoder.py --model intfloat/e5-large-v2
    python run_biencoder.py --doc-field full
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from metrics import evaluate

ROOT = pathlib.Path(__file__).parent
TOP_K = 10

# Embedding models are trained with asymmetric query/document formatting and
# silently lose several points of nDCG when it is omitted or mismatched.
# Three families matter here:
#
#   bge_en   BGE v1.5 English -- an instruction on the query only.
#   e5       E5 -- both sides tagged.
#   instruct LLM-based embedders (Qwen3-Embedding, gte-Qwen2, e5-mistral,
#            NV-Embed, SFR) -- queries carry a natural-language task
#            description, documents stay raw.
#
# Matching is ordered and first-match-wins, because the names overlap:
# "e5-mistral-7b-instruct" contains "e5" but must NOT get E5's prefixes.
TASK_DESCRIPTION = (
    "Given a task that requires calling a tool, retrieve the documentation "
    "of the tool that can accomplish it."
)

PROMPT_STYLES = [
    ("e5-mistral", "instruct"),
    ("qwen3-embedding", "instruct"),
    ("gte-qwen", "instruct"),
    ("bge-en-icl", "instruct"),
    ("nv-embed", "instruct"),
    ("sfr-embedding", "instruct"),
    ("linq-embed", "instruct"),
    ("bge-m3", "none"),          # m3 takes no instruction, unlike bge v1.5
    ("bge", "bge_en"),
    ("e5", "e5"),
    ("default", "none"),
]


def style_for(model_name: str) -> str:
    lowered = model_name.lower()
    for pattern, style in PROMPT_STYLES:
        if pattern != "default" and pattern in lowered:
            return style
    return "none"


def format_query(text: str, style: str) -> str:
    if style == "instruct":
        return f"Instruct: {TASK_DESCRIPTION}\nQuery: {text}"
    if style == "bge_en":
        return "Represent this sentence for searching relevant passages: " + text
    if style == "e5":
        return "query: " + text
    return text


def format_doc(text: str, style: str) -> str:
    return "passage: " + text if style == "e5" else text


DTYPES = {"float32": torch.float32, "float16": torch.float16,
          "bfloat16": torch.bfloat16}


def pick_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------
# encoding
# --------------------------------------------------------------------------
def encode_corpus(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
    cache_path: pathlib.Path,
) -> tuple[np.ndarray, float]:
    """Embed every tool once, with an on-disk cache.

    Returns (matrix, build_seconds). Build time is only meaningful on a cache
    miss; a cached load reports the load time and is flagged by the caller.
    """
    if cache_path.exists():
        t0 = time.perf_counter()
        emb = np.load(cache_path)
        return emb, time.perf_counter() - t0

    t0 = time.perf_counter()
    emb = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,   # cosine similarity becomes a dot product
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)
    build_s = time.perf_counter() - t0

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, emb)
    return emb, build_s


def search(
    query_vec: np.ndarray, corpus_mat: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Exact top-k by cosine similarity.

    Both sides are L2-normalised, so the dot product is the cosine. This is a
    brute-force scan: at 43k tools it is a single small matmul, and an
    approximate index would only add error for no measurable speedup.
    """
    scores = corpus_mat @ query_vec
    if k >= len(scores):
        order = np.argsort(-scores)
    else:
        top = np.argpartition(-scores, k)[:k]
        order = top[np.argsort(-scores[top])]
    return order, scores[order]


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--category", default="code")
    ap.add_argument("--model", default="BAAI/bge-large-en-v1.5")
    ap.add_argument("--doc-field", choices=["compact", "full"], default="compact",
                    help="tool text to index; compact matches what the LLM saw")
    ap.add_argument("--slices", type=int, nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-instruction", action="store_true",
                    help="omit ToolRet's per-query instruction field")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default=None, help="mps / cuda / cpu")
    ap.add_argument("--dtype", choices=list(DTYPES), default=None,
                    help="load weights in this precision. 7B+ models default "
                         "to float32 (~28GB); float16 halves that.")
    ap.add_argument("--trust-remote-code", action="store_true",
                    help="required by some LLM-based embedders (gte-Qwen2)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = ROOT / "data" / "eval" / args.category
    if not data.exists():
        sys.exit(f"no dataset at {data} -- run build_dataset.py first")

    manifest = json.loads((data / "manifest.json").read_text())
    queries = [json.loads(l) for l in (data / "queries.jsonl").open()]
    corpus = [json.loads(l) for l in (data / "corpus.jsonl").open()]
    qrels = json.loads((data / "qrels.json").read_text())

    if args.limit:
        queries = queries[: args.limit]
        qrels = {q["id"]: qrels[q["id"]] for q in queries}

    device = pick_device(args.device)
    style = style_for(args.model)
    print(f"bi-encoder | model={args.model} device={device} "
          f"doc_field={args.doc_field} queries={len(queries)}")
    print(f"prompt style: {style}")
    print(f"  query -> {format_query('<QUERY>', style)!r}")
    print(f"  doc   -> {format_doc('<DOC>', style)!r}\n")

    model_kwargs = {}
    if args.dtype:
        model_kwargs["torch_dtype"] = DTYPES[args.dtype]
    if args.trust_remote_code:
        model_kwargs["trust_remote_code"] = True

    t0 = time.perf_counter()
    model = SentenceTransformer(
        args.model, device=device,
        model_kwargs=model_kwargs or None,
        trust_remote_code=args.trust_remote_code,
    )
    print(f"  model loaded in {time.perf_counter() - t0:.1f}s"
          + (f" (dtype={args.dtype})" if args.dtype else ""))

    # --- index the whole corpus once --------------------------------------
    tool_ids = [c["id"] for c in corpus]
    row_of = {tid: i for i, tid in enumerate(tool_ids)}
    texts = [format_doc(c[args.doc_field], style) for c in corpus]

    slug = args.model.replace("/", "-")
    cache = (ROOT / "results" / "embeddings"
             / f"{slug}_{args.category}_{args.doc_field}.npy")
    cached = cache.exists()
    emb, build_s = encode_corpus(model, texts, args.batch_size, cache)
    print(f"  corpus: {emb.shape[0]:,} tools x {emb.shape[1]} dims "
          f"({'loaded from cache in' if cached else 'encoded in'} {build_s:.1f}s)")
    if not cached:
        print(f"  index build rate: {emb.shape[0] / build_s:,.0f} tools/s")

    out_dir = pathlib.Path(args.out) if args.out else ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    tag = f"biencoder_{args.category}_{slug}_{args.doc_field}"
    raw_path = out_dir / f"{tag}.jsonl"
    raw_fh = raw_path.open("w")

    # Queries are encoded once, not once per slice: the vector does not
    # depend on which tools are indexed. Each encode is timed individually
    # (no batching) so the per-query latency still reflects what a live
    # agent pays, while a 7B encoder does not redo the same work 5 times.
    t0 = time.perf_counter()
    q_vecs, q_encode_s = {}, {}
    for q in queries:
        text = q["query"]
        if not args.no_instruction:
            text = f"{text} {q['instruction']}"
        t1 = time.perf_counter()
        q_vecs[q["id"]] = model.encode(
            format_query(text, style), normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        q_encode_s[q["id"]] = time.perf_counter() - t1
    print(f"  {len(queries)} queries encoded in {time.perf_counter() - t0:.1f}s "
          f"(mean {1000 * np.mean(list(q_encode_s.values())):.1f}ms each)")

    sizes = args.slices or [s["size"] for s in manifest["slices"]]
    summary = []

    for size in sizes:
        meta = next((s for s in manifest["slices"] if s["size"] == size), None)
        if meta is None:
            print(f"  slice {size}: not in manifest, skipping")
            continue

        slice_ids = json.loads((data / "slices" / f"slice_{size}.json").read_text())
        rows = np.array([row_of[t] for t in slice_ids], dtype=np.int64)
        sub = np.ascontiguousarray(emb[rows])          # slice's vectors
        sub_ids = [tool_ids[r] for r in rows]

        print(f"\n  slice {size:,} ({meta['n_distractors']:,} distractors)")

        run: dict[str, list[str]] = {}
        latencies, encode_ts, search_ts = [], [], []

        for q in queries:
            t1 = time.perf_counter()
            order, scores = search(q_vecs[q["id"]], sub, TOP_K)
            search_s = time.perf_counter() - t1
            encode_s = q_encode_s[q["id"]]

            ranked = [sub_ids[i] for i in order[:TOP_K]]
            run[q["id"]] = ranked
            latencies.append(encode_s + search_s)
            encode_ts.append(encode_s)
            search_ts.append(search_s)

            raw_fh.write(json.dumps({
                "slice": size, "query_id": q["id"], "source": q["source"],
                "ranked_ids": ranked,
                "scores": [round(float(s), 5) for s in scores[:TOP_K]],
                "latency_s": round(encode_s + search_s, 5),
                "encode_s": round(encode_s, 5),
                "search_s": round(search_s, 5),
            }) + "\n")

        scores_agg = evaluate(run, qrels, k=TOP_K)
        lat = sorted(latencies)
        row = {
            "slice": size,
            "status": "ok",
            "n_distractors": meta["n_distractors"],
            **{k: round(v, 4) if isinstance(v, float) else v
               for k, v in scores_agg.items()},
            "latency_p50_ms": round(1000 * lat[len(lat) // 2], 2),
            "latency_p90_ms": round(1000 * lat[min(int(0.9 * len(lat)),
                                                   len(lat) - 1)], 2),
            "encode_mean_ms": round(1000 * float(np.mean(encode_ts)), 2),
            "search_mean_ms": round(1000 * float(np.mean(search_ts)), 3),
        }
        summary.append(row)
        print(f"    nDCG@{TOP_K}={row[f'ndcg@{TOP_K}']:.4f}  "
              f"Recall@{TOP_K}={row[f'recall@{TOP_K}']:.4f}  "
              f"Complete@{TOP_K}={row[f'completeness@{TOP_K}']:.4f}  "
              f"p50={row['latency_p50_ms']}ms")

    raw_fh.close()

    summary_path = out_dir / f"{tag}_summary.json"
    summary_path.write_text(json.dumps({
        "condition": "biencoder", "model": args.model, "category": args.category,
        "doc_field": args.doc_field, "device": device,
        "prompt_style": style, "dtype": args.dtype or "float32",
        "n_queries": len(queries), "top_k": TOP_K,
        "use_instruction": not args.no_instruction,
        "index_build_s": round(build_s, 2), "index_cached": cached,
        "results": summary,
    }, indent=2))

    w = 86
    print("\n" + "=" * w)
    print(f"{'slice':>7} {'nDCG@10':>9} {'Recall@10':>10} {'Compl@10':>9} "
          f"{'p50 ms':>9} {'p90 ms':>9} {'encode':>9} {'search':>9}")
    print("-" * w)
    for r in summary:
        print(f"{r['slice']:>7,} {r['ndcg@10']:>9.4f} {r['recall@10']:>10.4f} "
              f"{r['completeness@10']:>9.4f} {r['latency_p50_ms']:>9.2f} "
              f"{r['latency_p90_ms']:>9.2f} {r['encode_mean_ms']:>7.2f}ms "
              f"{r['search_mean_ms']:>7.3f}ms")
    print("=" * w)
    print("latency is per query: encode the query, then scan the slice.")
    print(f"index build (all {emb.shape[0]:,} tools) is a one-time "
          f"{build_s:.1f}s, amortised over every query.")
    print(f"\nper-query records -> {raw_path}")
    print(f"summary           -> {summary_path}")


if __name__ == "__main__":
    main()
