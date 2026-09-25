"""
Lexical baseline: Okapi BM25 over the same slices.

No model, no GPU, no embeddings -- term frequencies and a formula from the
1990s. It is here because ToolRet's own results put BM25 within two points
of BGE-large on the full benchmark, which makes "does dense retrieval earn
its cost for tool selection?" an open question rather than a rhetorical one.

Two implementation choices carry most of the quality:

IDENTIFIER TOKENISATION. Tool names are code: `get_factors`,
`imagenet_mobilenet_v2_100_224_feature_vector`, `QueryHistoryToday`. A plain
whitespace tokeniser sees one opaque token and matches nothing, while a
query says "factors" or "history". We split on non-alphanumerics, split
camelCase boundaries, and keep the original compound alongside its parts --
so `get_factors` indexes as {get_factors, get, factors} and matches both a
literal name reference and a natural-language description of what it does.

PER-SLICE INDEXING. IDF depends on how many documents contain a term, so it
changes with corpus composition. Each slice is indexed independently,
which is what a system actually holding N tools would compute. Reusing the
full-corpus IDF across slices would leak information about tools that are
not in the smaller indexes.

Usage
-----
    python run_bm25.py
    python run_bm25.py --doc-field full --k1 1.5 --b 0.6
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import re
import sys
import time

from metrics import evaluate

ROOT = pathlib.Path(__file__).parent
TOP_K = 10

# Okapi defaults. k1 controls term-frequency saturation, b the strength of
# length normalisation.
DEFAULT_K1 = 1.2
DEFAULT_B = 0.75

_SPLIT = re.compile(r"[^A-Za-z0-9]+")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

STOPWORDS = frozenset("""
a an the of for to in on at by with and or is are be as from this that it its
""".split())


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, with code identifiers split into parts.

    The compound is kept as well as its pieces: dropping it would lose exact
    name matches, and keeping only it would lose every descriptive match.
    """
    out: list[str] = []
    for chunk in _SPLIT.split(text):
        if not chunk:
            continue
        parts = _CAMEL.sub(" ", chunk).split()
        lowered = [p.lower() for p in parts]
        if len(lowered) > 1:
            out.append(chunk.lower())      # the compound itself
        out.extend(lowered)
    return [t for t in out if t not in STOPWORDS and len(t) > 1]


class BM25:
    """Okapi BM25 over an in-memory inverted index."""

    def __init__(self, docs: list[list[str]], k1: float, b: float) -> None:
        self.k1, self.b = k1, b
        self.n_docs = len(docs)
        self.doc_len = [len(d) for d in docs]
        self.avg_len = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0

        # term -> list of (doc index, term frequency)
        self.postings: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
        for i, doc in enumerate(docs):
            for term, tf in collections.Counter(doc).items():
                self.postings[term].append((i, tf))

        # Robertson/Sparck-Jones IDF, floored at zero so a term appearing in
        # more than half the corpus cannot push scores negative.
        self.idf = {
            term: max(0.0, math.log(
                1.0 + (self.n_docs - len(post) + 0.5) / (len(post) + 0.5)
            ))
            for term, post in self.postings.items()
        }

    def top_k(self, query: list[str], k: int) -> list[tuple[int, float]]:
        scores: dict[int, float] = collections.defaultdict(float)
        for term in query:
            post = self.postings.get(term)
            if not post:
                continue
            idf = self.idf[term]
            for doc_i, tf in post:
                norm = 1.0 - self.b + self.b * self.doc_len[doc_i] / self.avg_len
                scores[doc_i] += idf * tf * (self.k1 + 1.0) / (tf + self.k1 * norm)
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        return ranked[:k]


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--category", default="code")
    ap.add_argument("--doc-field", choices=["compact", "full"], default="compact")
    ap.add_argument("--slices", type=int, nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-instruction", action="store_true")
    ap.add_argument("--k1", type=float, default=DEFAULT_K1)
    ap.add_argument("--b", type=float, default=DEFAULT_B)
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

    print(f"BM25 | category={args.category} doc_field={args.doc_field} "
          f"queries={len(queries)} k1={args.k1} b={args.b}\n")

    text_by_id = {c["id"]: c[args.doc_field] for c in corpus}

    # Tokenise once; the tokens are slice-independent even though IDF is not.
    t0 = time.perf_counter()
    tokens_by_id = {tid: tokenize(t) for tid, t in text_by_id.items()}
    q_tokens = {}
    for q in queries:
        text = q["query"]
        if not args.no_instruction:
            text = f"{text} {q['instruction']}"
        q_tokens[q["id"]] = tokenize(text)
    print(f"  tokenised {len(tokens_by_id):,} tools + {len(queries)} queries "
          f"in {time.perf_counter() - t0:.2f}s")
    mean_len = sum(len(v) for v in tokens_by_id.values()) / len(tokens_by_id)
    print(f"  mean tool length {mean_len:.1f} tokens, "
          f"mean query length "
          f"{sum(len(v) for v in q_tokens.values()) / len(queries):.1f} tokens")

    out_dir = pathlib.Path(args.out) if args.out else ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    tag = f"bm25_{args.category}_{args.doc_field}"
    raw_fh = (out_dir / f"{tag}.jsonl").open("w")

    sizes = args.slices or [s["size"] for s in manifest["slices"]]
    summary = []

    for size in sizes:
        meta = next((s for s in manifest["slices"] if s["size"] == size), None)
        if meta is None:
            print(f"  slice {size}: not in manifest, skipping")
            continue

        slice_ids = json.loads((data / "slices" / f"slice_{size}.json").read_text())

        t0 = time.perf_counter()
        index = BM25([tokens_by_id[t] for t in slice_ids], args.k1, args.b)
        build_s = time.perf_counter() - t0

        print(f"\n  slice {size:,} ({meta['n_distractors']:,} distractors) "
              f"-- indexed in {build_s * 1000:.0f}ms, "
              f"{len(index.postings):,} unique terms")

        run, latencies = {}, []
        for q in queries:
            t1 = time.perf_counter()
            hits = index.top_k(q_tokens[q["id"]], TOP_K)
            latencies.append(time.perf_counter() - t1)

            ranked = [slice_ids[i] for i, _ in hits]
            run[q["id"]] = ranked
            raw_fh.write(json.dumps({
                "slice": size, "query_id": q["id"], "source": q["source"],
                "ranked_ids": ranked,
                "scores": [round(s, 4) for _, s in hits],
                "latency_s": round(latencies[-1], 6),
            }) + "\n")

        scores_agg = evaluate(run, qrels, k=TOP_K)
        lat = sorted(latencies)
        row = {
            "slice": size, "status": "ok",
            "n_distractors": meta["n_distractors"],
            **{k: round(v, 4) if isinstance(v, float) else v
               for k, v in scores_agg.items()},
            "index_build_ms": round(build_s * 1000, 1),
            "latency_p50_ms": round(1000 * lat[len(lat) // 2], 3),
            "latency_p90_ms": round(1000 * lat[min(int(0.9 * len(lat)),
                                                   len(lat) - 1)], 3),
        }
        summary.append(row)
        print(f"    nDCG@{TOP_K}={row[f'ndcg@{TOP_K}']:.4f}  "
              f"Recall@{TOP_K}={row[f'recall@{TOP_K}']:.4f}  "
              f"Complete@{TOP_K}={row[f'completeness@{TOP_K}']:.4f}  "
              f"p50={row['latency_p50_ms']}ms")

    raw_fh.close()
    summary_path = out_dir / f"{tag}_summary.json"
    summary_path.write_text(json.dumps({
        "condition": "bm25", "category": args.category,
        "doc_field": args.doc_field, "k1": args.k1, "b": args.b,
        "n_queries": len(queries), "top_k": TOP_K,
        "use_instruction": not args.no_instruction,
        "results": summary,
    }, indent=2))

    w = 80
    print("\n" + "=" * w)
    print(f"{'slice':>7} {'nDCG@10':>9} {'Recall@10':>10} {'Compl@10':>9} "
          f"{'index ms':>10} {'p50 ms':>9} {'p90 ms':>9}")
    print("-" * w)
    for r in summary:
        print(f"{r['slice']:>7,} {r['ndcg@10']:>9.4f} {r['recall@10']:>10.4f} "
              f"{r['completeness@10']:>9.4f} {r['index_build_ms']:>10.1f} "
              f"{r['latency_p50_ms']:>9.3f} {r['latency_p90_ms']:>9.3f}")
    print("=" * w)
    print(f"\nper-query records -> {out_dir / (tag + '.jsonl')}")
    print(f"summary           -> {summary_path}")


if __name__ == "__main__":
    main()
