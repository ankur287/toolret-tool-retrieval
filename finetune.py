"""Fine-tune a BGE bi-encoder on a sample of ToolRet's training set.

Rows are (query + instruction, positive tool doc, hard negative tool doc),
trained with MultipleNegativesRankingLoss so every other row in the batch is
an extra negative. Any training query whose text matches a ToolRet eval query
is dropped first, so the model never sees the test questions.

    python finetune.py --n 5000 --max-steps 30     # timing run
    python finetune.py --n 5000                    # full run
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.sentence_transformer.losses import (
    MultipleNegativesRankingLoss,
)
from transformers import TrainerCallback

ROOT = Path(__file__).parent
TRAIN = ROOT / "data/raw/train.jsonl"
EVAL_QUERIES = ROOT / "data/raw/queries.jsonl"
TRAIN_ROWS = 208_826
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def norm(text: str) -> str:
    return " ".join(text.lower().split())


def load_rows(n: int, seed: int) -> list[dict]:
    eval_queries = {norm(json.loads(l)["query"]) for l in open(EVAL_QUERIES)}

    # Sample line numbers up front so we only parse the rows we keep. Take
    # extra to cover rows dropped for leakage or missing negatives.
    rng = random.Random(seed)
    wanted = set(rng.sample(range(TRAIN_ROWS), min(TRAIN_ROWS, n * 2)))

    rows, leaked = [], 0
    with open(TRAIN) as f:
        for i, line in enumerate(f):
            if i not in wanted:
                continue
            r = json.loads(line)
            if not r["positive"] or not r["negative"]:
                continue
            if norm(r["query"]) in eval_queries:
                leaked += 1
                continue
            rows.append({
                "anchor": QUERY_PREFIX + f"{r['query']} {r['prompt']}",
                "positive": r["positive"][0],
                "negative": r["negative"][0],
            })
    rng.shuffle(rows)
    print(f"sampled {len(wanted):,} lines, dropped {leaked} leaked queries, "
          f"kept {min(n, len(rows)):,}")
    return rows[:n]


class StepTimer(TrainerCallback):
    def __init__(self):
        self.times: list[float] = []
        self._t = None

    def on_step_begin(self, args, state, control, **kw):
        self._t = time.perf_counter()

    def on_step_end(self, args, state, control, **kw):
        torch.mps.synchronize() if torch.backends.mps.is_available() else None
        self.times.append(time.perf_counter() - self._t)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="BAAI/bge-large-en-v1.5")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default="models/bge-large-toolret")
    args = ap.parse_args()

    t0 = time.perf_counter()
    rows = load_rows(args.n, args.seed)
    print(f"data loaded in {time.perf_counter() - t0:.0f}s")

    model = SentenceTransformer(args.model, device="mps")
    model.max_seq_length = args.max_seq_length

    timer = StepTimer()
    trainer = SentenceTransformerTrainer(
        model=model,
        train_dataset=Dataset.from_list(rows),
        loss=MultipleNegativesRankingLoss(model),
        args=SentenceTransformerTrainingArguments(
            output_dir=str(ROOT / args.out),
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            per_device_train_batch_size=args.batch_size,
            warmup_ratio=0.1,
            learning_rate=args.lr,
            bf16=args.bf16,
            logging_steps=10,
            save_strategy="no",
            report_to="none",
            seed=args.seed,
        ),
        callbacks=[timer],
    )
    t1 = time.perf_counter()
    trainer.train()
    train_s = time.perf_counter() - t1

    steady = timer.times[3:] or timer.times
    per_step = sum(steady) / len(steady)
    total_steps = -(-len(rows) // args.batch_size) * args.epochs
    print(f"\nsteps run: {len(timer.times)}  train time: {train_s:.0f}s")
    print(f"steady-state: {per_step:.2f}s/step")
    print(f"full run: {total_steps} steps -> ~{per_step * total_steps / 60:.0f} min")

    if args.max_steps < 0:
        model.save_pretrained(str(ROOT / args.out))
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
