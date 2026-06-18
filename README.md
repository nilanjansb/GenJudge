# GenJudge
Online Evaluation of Information Retrieval Models Using Generative Judgment with Pooled Context


# Usage

The evaluation pipeline consists of three steps:

## 1. Pool Top-*k* Documents from Official Runs

Generate a pooled set of documents by selecting the top-*k* documents from each retrieval run.

```bash
python -m GenJudge.pooling \
    --runs_loc <PATH_TO_RUNS_DIRECTORY> \
    --k <TOP_K>
```

**Arguments**

* `--runs_loc`: Directory containing the retrieval run files.
* `--k`: Number of top-ranked documents to pool from each run.

---

## 2. Generate Context from Pooled Documents

Create evaluation contexts using a subset of qrels filtered by the pooled top-*k* documents.

```bash
python -X utf8 -m GenJudge.gen_context \
    --runs_loc <PATH_TO_RUNS_DIRECTORY> \
    --qrels_loc <PATH_TO_QRELS_FILE> \
    --k <TOP_K>
```

**Arguments**

* `--runs_loc`: Directory containing the retrieval run files.
* `--qrels_loc`: Path to the official qrels file.
* `--k`: Number of top-ranked documents used during pooling.

---

## 3. Run Evaluation

Performs:

* Evaluation using the full qrels.
* Evaluation using the filtered qrels.
* Online evaluation using the proposed generative judge.

```bash
python -m GenJudge.pooled_eval \
    --runs_loc <PATH_TO_RUNS_DIRECTORY> \
    --qrels_loc <PATH_TO_QRELS_FILE> \
    --k <TOP_K>
```

**Arguments**

* `--runs_loc`: Directory containing the retrieval run files.
* `--qrels_loc`: Path to the official qrels file.
* `--k`: Number of pooled documents used for evaluation.

---

## Example

```bash
python -m GenJudge.pooling \
    --runs_loc ./runs/TREC29 \
    --k 1

python -X utf8 -m GenJudge.gen_context \
    --runs_loc ./runs/TREC29 \
    --qrels_loc ./data/qrels/2020qrels-pass.txt \
    --k 1

python -m GenJudge.pooled_eval \
    --runs_loc ./runs/TREC29 \
    --qrels_loc ./data/qrels/2020qrels-pass.txt \
    --k 1
```

