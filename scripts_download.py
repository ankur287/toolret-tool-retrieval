"""Download ToolRet (queries, tools, train) from HuggingFace to local JSONL."""
import json, pathlib
from datasets import load_dataset, get_dataset_config_names

RAW = pathlib.Path(__file__).parent / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

def dump(rows, path):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {len(rows):,} rows -> {path.name}")

# 1. Tool corpus (3 configs: code / customized / web)
print("[1/3] ToolRet-Tools")
tools = []
for cfg in ["code", "customized", "web"]:
    ds = load_dataset("mangopy/ToolRet-Tools", cfg, split="tools")
    for r in ds:
        tools.append({"id": r["id"], "documentation": r["documentation"], "corpus": cfg})
    print(f"  {cfg}: {len(ds):,}")
dump(tools, RAW / "tools.jsonl")

# 2. Queries (one config per source dataset)
print("[2/3] ToolRet-Queries")
cfgs = get_dataset_config_names("mangopy/ToolRet-Queries")
queries = []
for c in cfgs:
    ds = load_dataset("mangopy/ToolRet-Queries", c, split="queries")
    for r in ds:
        queries.append({**r, "source": c})
print(f"  {len(cfgs)} source configs")
dump(queries, RAW / "queries.jsonl")

# 3. Training set for fine-tuning
print("[3/3] ToolRet-Training-20w")
tr = load_dataset("mangopy/ToolRet-Training-20w", split="train")
print(f"  features: {list(tr.features)}")
dump([dict(r) for r in tr], RAW / "train.jsonl")
