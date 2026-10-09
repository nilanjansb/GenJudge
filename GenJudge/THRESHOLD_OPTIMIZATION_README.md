# Threshold Optimization Guide

## Overview
The `optimize_thresholds.py` script finds optimal threshold values for F1 score to relevance grade conversion in the `ndcg_all_runs` function.

## What It Does

1. **Loads baseline nDCG scores** using official QRELS (PyTerrier)
2. **Computes BERT F1 scores** for all documents against references
3. **Tests multiple threshold configurations** using two strategies:
   - **Grid Search**: Tests all combinations within ranges (comprehensive but slower)
   - **Quantile Search**: Tests pre-defined quantile configurations (faster)
4. **Evaluates each configuration** by computing nDCG with those thresholds
5. **Compares against baseline** using Kendall's tau correlation
6. **Reports optimal thresholds** with best correlation to official results

## Usage

### Basic Usage
```bash
python optimize_thresholds.py \
    --runs_loc "GenJudge\data\TREC_runs\TREC28" \
    --qrels_loc "GenJudge\data\official_qrels\2019qrels-pass.txt" \
    --k 10
```

### Parameters

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `--runs_loc` | Yes | - | Path to runs directory |
| `--qrels_loc` | Yes | - | Path to official QRELS file |
| `--k` | Yes | - | Cutoff for nDCG calculation (e.g., 10) |
| `--num_runs` | No | 5 | Number of runs to test (for faster testing) |
| `--method` | No | 'both' | Optimization method: 'grid', 'quantile', or 'both' |

### Examples

**Quick Test with 5 runs (Quantile only):**
```bash
python optimize_thresholds.py \
    --runs_loc "GenJudge\data\TREC_runs\TREC28" \
    --qrels_loc "GenJudge\data\official_qrels\2019qrels-pass.txt" \
    --k 10 \
    --num_runs 5 \
    --method quantile
```

**Comprehensive Test (Grid + Quantile):**
```bash
python optimize_thresholds.py \
    --runs_loc "GenJudge\data\TREC_runs\TREC28" \
    --qrels_loc "GenJudge\data\official_qrels\2019qrels-pass.txt" \
    --k 10 \
    --num_runs 10 \
    --method both
```

**Fast Grid Search (higher granularity):**
```bash
python optimize_thresholds.py \
    --runs_loc "GenJudge\data\TREC_runs\TREC28" \
    --qrels_loc "GenJudge\data\official_qrels\2019qrels-pass.txt" \
    --k 10 \
    --num_runs 3 \
    --method grid
```

## Output

### Console Output
```
============================================================
THRESHOLD OPTIMIZATION FOR TREC28
============================================================

Configuration:
  Dataset: TREC28
  QRELS: 12345
  Runs to test: 5
  Cutoff (K): 10

============================================================
BASELINE EVALUATION (Official QRELS)
============================================================

Baseline nDCG@10 scores:
  run1.gz: 0.5234
  run2.gz: 0.4891
  run3.gz: 0.5672
  run4.gz: 0.4456
  run5.gz: 0.5123

F1 Score Statistics:
  Count: 45678
  Mean: 0.7234
  Std: 0.1456
  Min: 0.1234
  Max: 0.9876

============================================================
QUANTILE SEARCH OPTIMIZATION
============================================================

Testing 7 quantile configurations...
Quantile Search: 100%|████████| 7/7 [00:45<00:00]

✓ Quantile Search Best Tau: 0.8000
  Optimal Quantiles: [0.67, 0.5, 0.33]
  Resulting Thresholds: t3=0.8234, t2=0.7123, t1=0.5891

✓ Results saved to: GenJudge/results/optimization/threshold_optimization_TREC28_10.tsv
✓ Summary saved to: GenJudge/results/optimization/optimization_summary_TREC28_10.json

============================================================
OPTIMIZATION COMPLETE
============================================================
```

### Output Files

#### 1. **threshold_optimization_TREC28_10.tsv**
Detailed results for all tested configurations:

| threshold_3 | threshold_2 | threshold_1 | kendall_tau | p_value |
|-------------|-------------|-------------|------------|---------|
| 0.8500 | 0.7200 | 0.5800 | 0.7500 | 0.0800 |
| 0.8400 | 0.7100 | 0.5700 | 0.7800 | 0.0600 |
| 0.8234 | 0.7123 | 0.5891 | 0.8000 | 0.0400 |

#### 2. **optimization_summary_TREC28_10.json**
Best results summary:

```json
{
  "grid_search": {
    "thresholds": [0.8234, 0.7123, 0.5891],
    "tau": 0.8000,
    "log_count": 1200
  },
  "quantile_search": {
    "quantiles": [0.67, 0.5, 0.33],
    "thresholds": [0.8234, 0.7123, 0.5891],
    "tau": 0.8000,
    "log_count": 7
  }
}
```

## How to Use Results

### Option 1: Update ndcg_all_runs with Optimal Values
After optimization, update your `ndcg_all_runs` call with the optimal thresholds:

```python
# From optimization_summary_TREC28_10.json:
# Optimal thresholds: [0.8234, 0.7123, 0.5891]
# OR optimal quantiles: [0.67, 0.5, 0.33]

gen_scores, gen_scores_pq, execution_time, similarity_lookup = self.ndcg_all_runs(
    test_passage_runs, 
    relevant_docs_dict, 
    qrel_texts, 
    filtered_qrels, 
    10,
    use_dynamic_thresholds=True,
    quantiles=[0.67, 0.5, 0.33]  # Use optimal quantiles
)
```

### Option 2: Create a Configuration File
Save optimal thresholds to reuse across runs:

```python
# Save after optimization
import json
optimal_config = {
    "TREC28": {"quantiles": [0.67, 0.5, 0.33], "thresholds": [0.8234, 0.7123, 0.5891]},
    "TREC29": {"quantiles": [0.70, 0.5, 0.30], "thresholds": [0.8145, 0.7001, 0.5623]}
}
with open("optimal_thresholds.json", "w") as f:
    json.dump(optimal_config, f, indent=2)
```

Then load and reuse:
```python
with open("optimal_thresholds.json") as f:
    config = json.load(f)
    
optimal_quantiles = config[self.ds]["quantiles"]
gen_scores, ... = self.ndcg_all_runs(..., quantiles=optimal_quantiles)
```

## Performance Notes

- **Quantile Search**: ~1-2 minutes for 7 configurations with 5 runs
- **Grid Search**: ~30-60 minutes for full search with 5 runs (depending on ranges)
- **Both**: ~35-65 minutes total

For faster results:
- Reduce `--num_runs` to 3-5
- Use `--method quantile` instead of `both`
- Run on GPU (BERT F1 computation is CUDA-accelerated)

## Interpreting Results

### Kendall's Tau
- **Range**: -1.0 to 1.0
- **Interpretation**:
  - **0.8 - 1.0**: Excellent correlation (thresholds match baseline ranking)
  - **0.6 - 0.8**: Good correlation
  - **0.4 - 0.6**: Moderate correlation
  - **< 0.4**: Weak correlation

Higher tau means your thresholds produce run rankings very similar to official QRELS.

### P-value
- **< 0.05**: Statistically significant correlation
- **< 0.10**: Marginally significant

## Troubleshooting

### "No results generated"
- Ensure QRELS file path is correct
- Verify runs directory contains .gz files
- Check dataset name (TREC28 or TREC29)

### "Out of memory"
- Reduce `--num_runs`
- Process one dataset at a time
- Consider running on GPU with larger VRAM

### "Kendall tau is negative"
- Your thresholds produce opposite ranking vs. baseline
- Try broader ranges or different quantile values
- May indicate noisy F1 scores or misaligned references

## Advanced Usage

### Custom Grid Search Ranges
Modify the script to adjust ranges:

```python
# In optimize_grid_search call:
best_t, best_tau, grid_log, _ = self.optimize_grid_search(
    test_passage_runs, qrel_lookup, similarity_lookup_all, 
    filtered_qrels, official_results, self.k,
    t3_range=(0.85, 0.98),  # Higher range for t3
    t2_range=(0.70, 0.90),  # Adjusted range for t2
    t1_range=(0.50, 0.80),  # Adjusted range for t1
    step=0.02  # Finer granularity
)
```

### Per-Dataset Optimization
Run optimization for both TREC28 and TREC29:

```bash
# TREC28
python optimize_thresholds.py --runs_loc "...\TREC28" --qrels_loc "...2019qrels-pass.txt" --k 10
 python -X utf8 -m GenJudge.optimize_thresholds --runs_loc "GenJudge\\data\\TREC_runs\\TREC28" --qrels_loc "GenJudge\\data\\official_qrels\\2019qrels-pass.txt" --k 10

# TREC29
python optimize_thresholds.py --runs_loc "...\TREC29" --qrels_loc "...2020qrels-pass.txt" --k 10
```

Compare results in `optimization_summary_*.json` files.
