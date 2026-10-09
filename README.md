# GenJudge: HYpothetical Document Relevance based Assessment (HYDRA),

GenJudge evaluates information retrieval (IR) systems using Large Language Models (LLMs) across standard pooling and selective pooling configurations, comparing pseudo-qrels against official benchmark judgments.

---

## Directory Structure & Prerequisites

Set up the project environment and ensure datasets and runs are structured under a `data/` directory:

```text
GenJudge/
├── data/
│   ├── official_qrels/
│   │   ├── 2019qrels-pass.txt
│   │   ├── 2020qrels-pass.txt
│   │   ├── covidqrels-pass.txt
│   │   └── qrels.dev.small.tsv
│   └── TREC_runs/
│       ├── TREC19/ (and TREC28)
│       ├── TREC20/ (and TREC29)
│       ├── TRECCOVID/
│       └── TREC30/
├── baseline/
│   └── topic-adapters-as-judges/
└── scripts/
    └── train.py

```

Set UTF-8 encoding support when invoking Python modules (`python -X utf8 -m ...`).

---

## 1. Standard Pipeline Execution

The standard pipeline runs sequentially through four phases: **Document Pooling**, **Context Generation**, **LLM Judgment Generation**, and **Pooled Evaluation**.

### Step 1.1: Document Pooling

Pool the top-k ranked documents per topic across all system runs:

```bash
# General Syntax
python -m GenJudge.pooling --runs_loc <RUNS_DIRECTORY> --k <POOL_DEPTH>

# Example: TREC Deep Learning 2019 (k = 1 to 10)
python -m GenJudge.pooling --runs_loc "data/TREC_runs/TREC19" --k 1

# Example: TREC Deep Learning 2020
python -m GenJudge.pooling --runs_loc "data/TREC_runs/TREC20" --k 1

# Example: TREC-COVID
python -m GenJudge.pooling --runs_loc "data/TREC_runs/TRECCOVID" --k 1

```

### Step 1.2: Context Generation

Generate topic-document context representations using the pooled candidates and reference qrels:

```bash
# General Syntax
python -X utf8 -m GenJudge.gen_context --runs_loc <RUNS_DIRECTORY> --qrels_loc <QRELS_FILE> --k <POOL_DEPTH>

# Example: TREC DL 2019
python -X utf8 -m GenJudge.gen_context --runs_loc "data/TREC_runs/TREC19" --qrels_loc "data/official_qrels/2019qrels-pass.txt" --k 1

# Example: TREC DL 2020
python -X utf8 -m GenJudge.gen_context --runs_loc "data/TREC_runs/TREC20" --qrels_loc "data/official_qrels/2020qrels-pass.txt" --k 1

# Example: TREC-COVID
python -X utf8 -m GenJudge.gen_context --runs_loc "data/TREC_runs/TRECCOVID" --qrels_loc "data/official_qrels/covidqrels-pass.txt" --k 1

```

### Step 1.3: Few-Shot LLM Judgment Generation

Prompt the generative judge using few-shot in-context demonstrations to label relevance:

```bash
# General Syntax
python -X utf8 -m GenJudge.fs_gen_judgment --runs_loc <RUNS_DIRECTORY> --qrels_loc <QRELS_FILE> --k <POOL_DEPTH> --fewshot_count <NUM_SHOTS>

# Example: TREC DL 2019 (1-shot judgment generation)
python -X utf8 -m GenJudge.fs_gen_judgment --runs_loc "data/TREC_runs/TREC19" --qrels_loc "data/official_qrels/2019qrels-pass.txt" --k 1 --fewshot_count 1

```

### Step 1.4: Pooled Evaluation

Evaluate system rankings using the generated pseudo-labels against official judgments:

```bash
# General Syntax
python -X utf8 -m GenJudge.pooled_eval --runs_loc <RUNS_DIRECTORY> --qrels_loc <QRELS_FILE> --k <POOL_DEPTH>

# Example: TREC DL 2019
python -X utf8 -m GenJudge.pooled_eval --runs_loc "data/TREC_runs/TREC19" --qrels_loc "data/official_qrels/2019qrels-pass.txt" --k 1

# Example: TREC DL 2020
python -X utf8 -m GenJudge.pooled_eval --runs_loc "data/TREC_runs/TREC20" --qrels_loc "data/official_qrels/2020qrels-pass.txt" --k 1

# Example: TREC-COVID
python -X utf8 -m GenJudge.pooled_eval --runs_loc "data/TREC_runs/TRECCOVID" --qrels_loc "data/official_qrels/covidqrels-pass.txt" --k 1

```

---

## 2. Selective Pooling Pipeline (Testing & Ranker Expansion)

Designed for testing, this workflow fixes the pooling depth (k) while iteratively adding rankers to incrementally expand the document pool. This allows for controlled evaluation of how newly introduced systems contribute to the overall pool at a constant evaluation depth.

```text
selective_pooling  ──>  selective_gen_context  ──>  selective_[fs_]gen_judgment  ──>  selective_pooled_eval

```

### Step 2.1: Selective Document Pooling

```bash
python -m GenJudge.selective_pooling --runs_loc "data/TREC_runs/TREC19" --k 1
python -m GenJudge.selective_pooling --runs_loc "data/TREC_runs/TREC20" --k 1
python -m GenJudge.selective_pooling --runs_loc "data/TREC_runs/TRECCOVID" --k 1

```

### Step 2.2: Selective Context Generation

```bash
python -X utf8 -m GenJudge.selective_gen_context --runs_loc "data/TREC_runs/TREC19" --qrels_loc "data/official_qrels/2019qrels-pass.txt" --k 1
python -X utf8 -m GenJudge.selective_gen_context --runs_loc "data/TREC_runs/TREC20" --qrels_loc "data/official_qrels/2020qrels-pass.txt" --k 1
python -X utf8 -m GenJudge.selective_gen_context --runs_loc "data/TREC_runs/TRECCOVID" --qrels_loc "data/official_qrels/covidqrels-pass.txt" --k 1

```

### Step 2.3: Selective Judgment Generation

* **Few-Shot Selection:**
```bash
python -X utf8 -m GenJudge.selective_fs_gen_judgment --runs_loc "data/TREC_runs/TREC19" --qrels_loc "data/official_qrels/2019qrels-pass.txt" --k 1

```



* **Zero-Shot Selection:**
```bash
python -X utf8 -m GenJudge.selective_gen_judgment --runs_loc "data/TREC_runs/TREC19" --qrels_loc "data/official_qrels/2019qrels-pass.txt" --k 1

```



### Step 2.4: Selective Pooled Evaluation

```bash
python -X utf8 -m GenJudge.selective_pooled_eval --runs_loc "data/TREC_runs/TREC19" --qrels_loc "data/official_qrels/2019qrels-pass.txt" --k 1
python -X utf8 -m GenJudge.selective_pooled_eval --runs_loc "data/TREC_runs/TREC20" --qrels_loc "data/official_qrels/2020qrels-pass.txt" --k 1
python -X utf8 -m GenJudge.selective_pooled_eval --runs_loc "data/TREC_runs/TRECCOVID" --qrels_loc "data/official_qrels/covidqrels-pass.txt" --k 1

```

---

## 3. Depth-Specific Evaluation Benchmarks

To analyze system correlation at fixed evaluation depths, run `GenJudge.pooled_eval` with the `--eval_depth` parameter.

### Depth 10 Evaluation (`--eval_depth 10`)

```bash
python -X utf8 -m GenJudge.pooled_eval --runs_loc "data/TREC_runs/TREC19" --qrels_loc "data/official_qrels/2019qrels-pass.txt" --k 1 --eval_depth 10
python -X utf8 -m GenJudge.pooled_eval --runs_loc "data/TREC_runs/TREC20" --qrels_loc "data/official_qrels/2020qrels-pass.txt" --k 1 --eval_depth 10
python -X utf8 -m GenJudge.pooled_eval --runs_loc "data/TREC_runs/TRECCOVID" --qrels_loc "data/official_qrels/covidqrels-pass.txt" --k 1 --eval_depth 10

```

### Depth 100 Evaluation (`--eval_depth 100`)

```bash
python -X utf8 -m GenJudge.pooled_eval --runs_loc "data/TREC_runs/TREC19" --qrels_loc "data/official_qrels/2019qrels-pass.txt" --k 1 --eval_depth 100
python -X utf8 -m GenJudge.pooled_eval --runs_loc "data/TREC_runs/TREC20" --qrels_loc "data/official_qrels/2020qrels-pass.txt" --k 1 --eval_depth 100
python -X utf8 -m GenJudge.pooled_eval --runs_loc "data/TREC_runs/TRECCOVID" --qrels_loc "data/official_qrels/covidqrels-pass.txt" --k 1 --eval_depth 100

```

---

## 4. Baseline: Topic Adapters as Judges (TAJ)

### Step 4.1: LoRA Fine-Tuning MonoT5 Adapters

Train MonoT5 base adapters on pooled query-document texts:

```bash
python scripts/train.py \
    --encoder_type mono \
    --data_path "results/pooled_qrels_with_texts" \
    --lora True \
    --model_name "castorini/monot5-base-msmarco-10k" \
    --output_dir "baseline/topic-adapters-as-judges/data/output" \
    --predict True \
    --depth 2 \
    --dataset_name TREC28

```

### Step 4.2: Inference with TAJ

Generate relevance scores using the adapter-tuned judge:

```bash
python -X utf8 -m GenJudge.taj_judgment --runs_loc "data/TREC_runs/TREC19" --qrels_loc "data/official_qrels/2019qrels-pass.txt" --k 1
python -X utf8 -m GenJudge.taj_judgment --runs_loc "data/TREC_runs/TREC20" --qrels_loc "data/official_qrels/2020qrels-pass.txt" --k 1
python -X utf8 -m GenJudge.taj_judgment --runs_loc "data/TREC_runs/TRECCOVID" --qrels_loc "data/official_qrels/covidqrels-pass.txt" --k 1

```