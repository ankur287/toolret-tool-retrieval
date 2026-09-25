"""
Condition 4: two-stage retrieval -- retrieve, then rerank.

Stage 1 scans the whole corpus cheaply and returns its best `--depth`
candidates. Stage 2 re-scores only those candidates with a cross-encoder,
which reads the query and the tool *together* in one forward pass instead of
embedding them separately. Joint encoding is what makes it more accurate; it
is also why it cannot scan the corpus -- one forward pass per candidate,
versus one matmul for all of them.

What the numbers mean
---------------------
Stage 2 can only reorder what stage 1 handed it, so this run reports three
recalls per slice rather than one:

    stage1_recall@10   what single-stage retrieval already achieved
    stage1_recall@K    the CEILING -- the reranker cannot exceed this
    final_recall@10    where the two-stage pipeline actually landed

The gap between the first two is the available headroom. Where the third
sits inside that gap says whether the cross-encoder earned its K extra
forward passes. Note that improvement is not guaranteed: reranking reorders
the candidate set, so a cross-encoder that disagrees with the retriever can
demote a correct tool out of the top 10 as easily as promote one into it.

Usage
-----
    python run_rerank.py                          # BGE-large -> bge-reranker-v2-m3
    python run_rerank.py --first-stage bm25       # lexical stage 1
    python run_rerank.py --depth 100
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
from sentence_transformers import CrossEncoder

import run_bm25
from metrics import evaluate, recall_at_k
from run_biencoder import (
    encode_corpus, format_doc, format_query, pick_device, search, style_for,
)
from sentence_transformers import SentenceTransformer

ROOT = pathlib.Path(__file__).parent
TOP_K = 10


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--category", default="code")
    ap.add_argument("--first-stage", choices=["biencoder", "bm25"],
                    default="biencoder")
    ap.add_argument("--retriever", default="BAAI/bge-large-en-v1.5",
                    help="stage-1 model when --first-stage biencoder")
    ap.add_argument("--reranker", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--depth", type=int, default=50,
                    help="candidates passed from stage 1 to stage 2")
    ap.add_argument("--doc-field", choices=["compact", "full"], default="compact")
    ap.add_argument("--slices", type=int, nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-instruction", action="store_true")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--device", default=None)
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
    text_by_id = {c["id"]: c[args.doc_field] for c in corpus}
    tool_ids = [c["id"] for c in corpus]

    print(f"two-stage | stage1={args.first_stage} depth={args.depth} "
          f"stage2={args.reranker}")
    print(f"device={device} doc_field={args.doc_field} queries={len(queries)}\n")

    # ---- query text (shared by both stages) ------------------------------
    q_text = {}
    for q in queries:
        t = q["query"]
        if not args.no_instruction:
            t = f"{t} {q['instruction']}"
        q_text[q["id"]] = t

    # ---- stage 1 setup ---------------------------------------------------
    if args.first_stage == "biencoder":
        style = style_for(args.retriever)
        t0 = time.perf_counter()
        st = SentenceTransformer(args.retriever, device=device)
        print(f"  stage-1 model loaded in {time.perf_counter() - t0:.1f}s")

        slug = args.retriever.replace("/", "-")
        cache = (ROOT / "results" / "embeddings"
                 / f"{slug}_{args.category}_{args.doc_field}.npy")
        cached = cache.exists()
        emb, build_s = encode_corpus(
            st, [format_doc(text_by_id[t], style) for t in tool_ids],
            args.batch_size, cache)
        print(f"  corpus embeddings {emb.shape} "
              f"({'cached' if cached else f'built in {build_s:.1f}s'})")

        q_vecs = {
            qid: st.encode(format_query(t, style), normalize_embeddings=True,
                           convert_to_numpy=True).astype(np.float32)
            for qid, t in q_text.items()
        }
        row_of = {t: i for i, t in enumerate(tool_ids)}
    else:
        tokens_by_id = {t: run_bm25.tokenize(text_by_id[t]) for t in tool_ids}
        q_tokens = {qid: run_bm25.tokenize(t) for qid, t in q_text.items()}
        print(f"  stage-1 BM25: tokenised {len(tokens_by_id):,} tools")

    # ---- stage 2 ---------------------------------------------------------
    t0 = time.perf_counter()
    ce = CrossEncoder(args.reranker, device=device, max_length=args.max_length)
    print(f"  reranker loaded in {time.perf_counter() - t0:.1f}s")

    out_dir = pathlib.Path(args.out) if args.out else ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    stage1_slug = (args.retriever.replace("/", "-")
                   if args.first_stage == "biencoder" else "bm25")
    tag = (f"rerank_{args.category}_{stage1_slug}_"
           f"{args.reranker.replace('/', '-')}_d{args.depth}_{args.doc_field}")
    raw_fh = (out_dir / f"{tag}.jsonl").open("w")

    sizes = args.slices or [s["size"] for s in manifest["slices"]]
    summary = []

    for size in sizes:
        meta = next((s for s in manifest["slices"] if s["size"] == size), None)
        if meta is None:
            print(f"  slice {size}: not in manifest, skipping")
            continue
        slice_ids = json.loads((data / "slices" / f"slice_{size}.json").read_text())
        depth = min(args.depth, len(slice_ids))

        if args.first_stage == "biencoder":
            rows = np.array([row_of[t] for t in slice_ids], dtype=np.int64)
            sub = np.ascontiguousarray(emb[rows])
            sub_ids = [tool_ids[r] for r in rows]
        else:
            bm = run_bm25.BM25([tokens_by_id[t] for t in slice_ids],
                               run_bm25.DEFAULT_K1, run_bm25.DEFAULT_B)

        print(f"\n  slice {size:,} ({meta['n_distractors']:,} distractors)")

        run, s1_run, s1_deep = {}, {}, {}
        s1_times, s2_times = [], []

        for q in queries:
            qid = q["id"]
            # -- stage 1 --
            t0 = time.perf_counter()
            if args.first_stage == "biencoder":
                order, _ = search(q_vecs[qid], sub, depth)
                cands = [sub_ids[i] for i in order[:depth]]
            else:
                cands = [slice_ids[i] for i, _ in bm.top_k(q_tokens[qid], depth)]
            s1_times.append(time.perf_counter() - t0)

            s1_run[qid] = cands[:TOP_K]     # what single-stage would have said
            s1_deep[qid] = cands            # the reranker's ceiling

            # -- stage 2 --
            t0 = time.perf_counter()
            if cands:
                scores = ce.predict([(q_text[qid], text_by_id[c]) for c in cands],
                                    batch_size=args.batch_size,
                                    show_progress_bar=False)
                ranked = [c for _, c in sorted(zip(scores, cands),
                                               key=lambda p: -float(p[0]))]
            else:
                scores, ranked = [], []
            s2_times.append(time.perf_counter() - t0)

            run[qid] = ranked[:TOP_K]
            raw_fh.write(json.dumps({
                "slice": size, "query_id": qid, "source": q["source"],
                "ranked_ids": run[qid],
                "stage1_ids": cands[:TOP_K],
                "scores": [round(float(s), 5) for s in sorted(
                    (float(x) for x in scores), reverse=True)[:TOP_K]],
                "stage1_s": round(s1_times[-1], 5),
                "stage2_s": round(s2_times[-1], 5),
            }) + "\n")

        # ---- the three recalls -------------------------------------------
        final = evaluate(run, qrels, k=TOP_K)
        s1 = evaluate(s1_run, qrels, k=TOP_K)
        ceiling = float(np.mean([
            recall_at_k(s1_deep[qid], {t for t, r in rels.items() if r > 0}, depth)
            for qid, rels in qrels.items()
        ]))

        gap = ceiling - s1[f"recall@{TOP_K}"]
        gained = final[f"recall@{TOP_K}"] - s1[f"recall@{TOP_K}"]
        row = {
            "slice": size, "status": "ok", "depth": depth,
            "n_distractors": meta["n_distractors"],
            **{f"final_{k}": round(v, 4) if isinstance(v, float) else v
               for k, v in final.items()},
            "stage1_ndcg@10": round(s1["ndcg@10"], 4),
            "stage1_recall@10": round(s1["recall@10"], 4),
            f"stage1_recall@{depth}_ceiling": round(ceiling, 4),
            "headroom": round(gap, 4),
            "recall_gained": round(gained, 4),
            "headroom_used": round(gained / gap, 4) if gap > 1e-9 else None,
            "stage1_mean_ms": round(1000 * float(np.mean(s1_times)), 2),
            "stage2_mean_ms": round(1000 * float(np.mean(s2_times)), 2),
        }
        summary.append(row)
        print(f"    stage1 recall@10={s1['recall@10']:.4f} -> "
              f"ceiling recall@{depth}={ceiling:.4f} (headroom {gap:+.4f})")
        print(f"    FINAL  nDCG@10={final['ndcg@10']:.4f} "
              f"(stage1 {s1['ndcg@10']:.4f}, {final['ndcg@10'] - s1['ndcg@10']:+.4f})  "
              f"Recall@10={final['recall@10']:.4f} ({gained:+.4f})  "
              f"Compl@10={final['completeness@10']:.4f}")
        print(f"    stage1 {row['stage1_mean_ms']:.1f}ms + "
              f"stage2 {row['stage2_mean_ms']:.1f}ms per query")

    raw_fh.close()
    summary_path = out_dir / f"{tag}_summary.json"
    summary_path.write_text(json.dumps({
        "condition": "rerank", "category": args.category,
        "first_stage": args.first_stage, "retriever": args.retriever,
        "reranker": args.reranker, "depth": args.depth,
        "doc_field": args.doc_field, "device": device,
        "n_queries": len(queries), "top_k": TOP_K,
        "use_instruction": not args.no_instruction, "results": summary,
    }, indent=2))

    w = 104
    print("\n" + "=" * w)
    print(f"{'slice':>7} | {'s1 nDCG':>8} {'s1 R@10':>8} {'ceiling':>8} | "
          f"{'nDCG@10':>8} {'R@10':>8} {'C@10':>8} | {'gain':>7} {'used':>6} | "
          f"{'s1 ms':>7} {'s2 ms':>8}")
    print("-" * w)
    for r in summary:
        used = f"{r['headroom_used']:.0%}" if r["headroom_used"] is not None else "--"
        ck = next(k for k in r if k.startswith("stage1_recall@") and k.endswith("ceiling"))
        print(f"{r['slice']:>7,} | {r['stage1_ndcg@10']:>8.4f} "
              f"{r['stage1_recall@10']:>8.4f} {r[ck]:>8.4f} | "
              f"{r['final_ndcg@10']:>8.4f} {r['final_recall@10']:>8.4f} "
              f"{r['final_completeness@10']:>8.4f} | "
              f"{r['recall_gained']:>+7.4f} {used:>6} | "
              f"{r['stage1_mean_ms']:>7.1f} {r['stage2_mean_ms']:>8.1f}")
    print("=" * w)
    print("ceiling = stage-1 recall at full depth; the reranker cannot exceed it.")
    print("used    = fraction of available headroom the reranker actually captured.")
    print(f"\nper-query records -> {out_dir / (tag + '.jsonl')}")
    print(f"summary           -> {summary_path}")


if __name__ == "__main__":
    main()
