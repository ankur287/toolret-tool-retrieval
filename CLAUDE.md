# ToolRet corpus-scaling evaluation

Evaluation harness behind a Towards Data Science article on **tool retrieval /
selection for LLM agents**: when an agent has more tools than fit in context,
how do you pick the right one?

The experiment is a **corpus-scaling sweep**. One fixed set of queries with
fixed ground truth is evaluated against nested tool corpora of increasing
size. Same queries, same labels, only the size of the haystack changes, so
the curve isolates how each method degrades as the tool pool grows.

## Framing constraint (important)

ToolRet **predates MCP and is not MCP-specific**. We apply it to the
MCP-shaped problem. Do not claim the benchmark is about MCP in any doc,
comment, or draft.

Paper: *Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for
Large Language Models*, arXiv 2503.01763 (ACL 2025 Findings).

---

## Environment

Python **3.14.4** works — torch 2.14.0 with MPS, sentence-transformers 6.1.0,
transformers 5.17.0. No fallback to an older Python was needed.

```bash
.venv-314/bin/python <script>.py      # NOT system python
```

Hardware: Apple M5 Pro, 20-core GPU, 48 GB unified memory. Everything runs
locally on Metal/MPS. No cloud, no paid APIs.

Ollama 0.23.2 with `gemma3:12b` (Q4_K_M, 131,072 context). Start it with
`ollama serve` if `curl localhost:11434/api/version` fails.

---

## Data

`data/raw/` — downloaded by `scripts_download.py` from HuggingFace:

| File | Rows | Size | Source |
|---|---|---|---|
| `tools.jsonl` | 44,453 | 34 MB | `mangopy/ToolRet-Tools` (code/customized/web) |
| `queries.jsonl` | 7,961 | 23 MB | `mangopy/ToolRet-Queries` (35 configs) |
| `train.jsonl` | 208,826 | **3.0 GB** | `mangopy/ToolRet-Training-20w` |

**Keep `train.jsonl` out of git.**

Verified facts about the raw data (re-checked, not assumed):
- 44,453 **unique** tool IDs, zero duplicate rows.
- All **14,106** ground-truth label references resolve into the corpus. None missing.
- Labels/query mean 1.77 (paper says 2.17 — they report on a filtered 7,615-query set).

### Schemas

**Queries**: `id`, `query`, `instruction`, `labels`, `category` (+ `source` added
at download). `labels` is a **JSON-encoded string** holding
`[{"id": "<tool_id>", "doc": {...}}, ...]` — binary relevance over those IDs.
`instruction` is ToolRet's GPT-4o-generated task instruction (the paper's
"w/ instruction" setting).

**Tools**: `id`, `documentation` (JSON-encoded string; fields are heterogeneous
across source datasets — `name`/`description`/`parameters`/`doc_arguments`/
`category`/`path` depending on origin).

**Train**: `query`, `id`, `prompt`, `positive` (list of tool-doc JSON strings),
`negative` (list). Already in bi-encoder triplet shape with hard negatives.

---

## Eval set

Built by `build_dataset.py` into `data/eval/code/`.

**Category `code`** was chosen over `customized` because its docs are verbose
enough (52.1 tok/tool) that the category **hits gemma3's context wall at
~2,500 of its 3,794 tools** — the wall stays inside the experiment.
`customized` (27.4 tok/tool) fits entirely and would lose that.

Both categories are **fully self-contained**: zero ground-truth leakage
outside their own corpus.

200 queries, stratified across all 7 source datasets:
`gorilla-huggingface` 57 · `toolink` 57 · `craft-math-algebra` 32 ·
`craft-vqa` 23 · `craft-tabmwp` 20 · `gorilla-tensor` 6 · `gorilla-pytorch` 5

**186 unique ground-truth tools** (1.30 labels/query, heavy sharing).

| Slice | Distractors | GT density | Compact tokens | Fits gemma3? |
|---|---|---|---|---|
| 1,000 | 814 | 18.6% | 52,828 | yes |
| 1,500 | 1,314 | 12.4% | 78,573 | yes |
| 2,000 | 1,814 | 9.3% | 104,544 | yes |
| 2,500 | 2,314 | 7.4% | 129,989 | marginal (483 tok headroom) |
| 3,794 | 3,608 | 4.9% | 197,750 | **NO — the wall** |

Two invariants, both asserted in the builder:
1. **Every slice contains all 186 GT tools** — a query is always answerable.
2. **Slices nest** — distractors are a prefix of one fixed shuffled order.

Files: `queries.jsonl`, `corpus.jsonl` (both `compact` and `full` text per
tool), `qrels.json`, `manifest.json`, `slices/slice_*.json`.

### compact vs full

`compact` = `name: description`, description truncated to 180 chars. Parameter
schemas are dropped — they matter for *calling* a tool, not *picking* one, and
they dominate the token budget.

**Full docs never fit, not even at 1,000 tools** (192,632 tokens vs a 131,072
window). Compact is not an optimization; it is the only reason the LLM
condition exists at all.

---

## Measured constants (do not re-derive)

| Quantity | Value | How |
|---|---|---|
| gemma3 tok/tool, mixed corpus, compact | 29.5 | Ollama `prompt_eval_count` |
| gemma3 tok/tool, **code** category, compact | 52.1 | same |
| chars/token, gemma3, on these tool lists | 3.79 | same |
| Context wall, mixed corpus | ~4,400 tools | 131,072 / 29.5 |
| **Context wall, code category** | **~2,500 tools** | 131,072 / 52.1 |
| Prefill rate, gemma3:12b on MPS | **247 tok/s** | 53,161 tok in 215.2 s |
| Decode rate | ~24 tok/s | 54 tok in 2.28 s |

The wall is **per query**, not per run — query count does not affect it.

The wall is a **knob, not a constant**: it moves with `DESC_TRUNC` and whether
lines are index-prefixed. 100-char descriptions and no index → ~20 tok/tool.

---

## Scripts

```bash
.venv-314/bin/python build_dataset.py                       # build eval set
.venv-314/bin/python run_llm_only.py --limit 1 --slices 1000 --latency-mode raw
.venv-314/bin/python run_biencoder.py --model BAAI/bge-large-en-v1.5
.venv-314/bin/python run_biencoder.py --model Qwen/Qwen3-Embedding-8B --dtype float16
.venv-314/bin/python run_bm25.py
```

`metrics.py` is shared by every condition: nDCG@10 (binary gains, ideal
capped at min(k, |relevant|)), Recall@10, Completeness@10 (1.0 only if *all*
targets are in the top k). Unanswered queries score zero rather than being
skipped, so nothing gains by declining to answer.

---

## Gotchas that cost real time — do not rediscover

**Slice files list GT tools first.** That is how they are built. Presenting
them in that order clusters every correct answer at the top of the catalogue
and the LLM scores well on position bias alone. `run_llm_only.py` shuffles
with a seed derived from slice size. Verified: GT spreads from index 1 to 998.

**Ollama silently truncates past `num_ctx`.** It drops tokens and returns a
plausible answer with no error. A "5,000 tool" run would quietly evaluate
~4,400 with the front deleted. The runner reads the true `prompt_eval_count`
back on the first call of each slice and aborts the slice if the catalogue
did not survive.

**Size `num_ctx` to the slice.** Asking for 131k on a 53k prompt makes Ollama
allocate a KV cache several times larger than needed and inflates latency.
`MODEL_CONTEXT_LIMIT` stays 131,072 (it decides what is runnable); runtime
`num_ctx` is `est_tokens * 1.25 + 4096`, capped.

**Query goes LAST in the LLM prompt.** The catalogue is byte-identical across
queries, so it sits in the prefix and Ollama reuses the KV cache. Prefill is
then paid once per slice instead of 200 times — ~20x wall-clock. Putting the
query first would silently destroy this.

**Raw latency is forced with a nonce, not by moving the query.** A unique
`<!-- uuid -->` at position 0 defeats the cache while leaving the prompt
byte-identical otherwise, so raw and warm accuracy stay comparable. Moving
the query to the front would change the task (model reads the catalogue
already knowing what to look for).

**Embedding prompt prefixes are model-specific and matter.** BGE v1.5 puts an
instruction on the query only; E5 tags both sides; LLM-based embedders
(Qwen3-Embedding, gte-Qwen2, e5-mistral, NV-Embed, SFR) want
`Instruct: {task}\nQuery: {text}` with raw documents. Matching in
`run_biencoder.py` is **ordered, first-match-wins** because the names overlap:
`e5-mistral-7b-instruct` contains `e5` but must NOT get E5's prefixes.
Getting this wrong costs several nDCG points invisibly.

**Encode the corpus once, not per slice.** A tool's embedding does not depend
on what else is indexed. Cached in `results/embeddings/`.

**Encode each query once, not once per slice.** Free at 20 ms, minutes of
waste at 7-8B. Still timed individually and unbatched so per-query latency
keeps its meaning.

**7B+ models default to fp32 (~28-32 GB).** Pass `--dtype float16`.

**BM25 needs identifier tokenization.** Tool names are code:
`get_factors`, `imagenet_mobilenet_v2_100_224_feature_vector`,
`QueryHistoryToday`. Whitespace tokenization sees one opaque token and matches
nothing — which looks like a result, not a bug. We split on non-alphanumerics,
split camelCase, and keep the compound alongside its parts. IDF is rebuilt
**per slice** since it depends on corpus composition.

**The paper's "E5" is `e5-mistral-7b-instruct` (7B), not `e5-large-v2` (335M).**
Do not compare our e5-large numbers to the paper's 38.97.

---

## Results so far (200 queries, code category, compact docs, w/ instruction)

### nDCG@10

| Slice | BM25 | BGE-large | e5-large-v2 | gemma3:12b |
|---|---|---|---|---|
| 1,000 | **0.6218** | 0.6108 | 0.5442 | pending |
| 1,500 | 0.5783 | **0.5802** | 0.4895 | pending |
| 2,000 | **0.5601** | 0.5538 | 0.4689 | pending |
| 2,500 | **0.5394** | 0.5309 | 0.4526 | pending |
| 3,794 | **0.4942** | 0.4931 | 0.4152 | **cannot run** |

### Recall@10 / Completeness@10 at the full corpus

| Method | Recall@10 | Completeness@10 |
|---|---|---|
| BM25 | 0.6217 | 0.5750 |
| BGE-large | 0.6042 | 0.5600 |
| e5-large-v2 | 0.5617 | 0.5350 |

### Cost

| Method | Index build | Query p50 | Model |
|---|---|---|---|
| BM25 | 3-9 **ms** | 0.12-0.47 **ms** | none, CPU |
| BGE-large | 25.7 s | ~19 ms | 1.3 GB, MPS |
| gemma3:12b raw | — | **215 s** | 8.1 GB, MPS |

### Reading of the results

- **BM25 ties BGE-large.** nDCG gaps of 0.001-0.011 on 200 queries are inside
  noise; the honest claim is "statistically indistinguishable," not "BM25
  wins." BM25 does take Recall and Completeness at all 5 slices, but by small
  margins on correlated metrics. This reproduces ToolRet's own finding
  (BM25 36.46 vs BGE-large 38.49).
- BM25's real advantage is cost: same accuracy, ~40-150x faster per query,
  ~3,000x faster to index, no model, no GPU.
- Caveat: `code` tools are identifier-heavy and tool names often literally
  describe the function, which is the regime most favourable to lexical
  matching. `web`/`customized` may differ.
- Everything degrades monotonically — retrieval **erodes**, the LLM hits a
  **wall**. That contrast is the article's spine.
- Absolute numbers are NOT comparable to the paper's table (3,794-tool
  single-category corpus vs their 43k). The **shape of the curve** is ours.

---

## Status

Done: dataset, `metrics.py`, LLM-only runner (1 query smoke-tested),
bi-encoder runner (BGE-large, e5-large-v2), BM25 runner.

In flight: `Qwen/Qwen3-Embedding-8B` — tests whether 8B/16 GB beats a 3 ms
keyword index.

Not started: two-stage rerank (bge-reranker-v2-m3), bi-encoder fine-tune on
ToolRet-train, full LLM-only sweep.

**The article itself is not to be written until real results exist for all
conditions.**

### Open decisions
- Full LLM-only raw sweep is ~76 h at 247 tok/s. `--latency-mode both
  --raw-sample 20` gets the same p50/p90 in ~7 h. User asked for raw-for-all;
  the flag is `--latency-mode raw` and the runner is resumable.
- Delete `results/llm_only_code_gemma3-12b.jsonl` before a full run or the
  smoke-test query is skipped on resume.
- `--no-instruction` untested. Mean query length is 42.6 tokens *because* the
  instruction is concatenated; BM25 likely benefits from that more than a
  bi-encoder does.
