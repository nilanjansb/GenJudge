import gzip
import math
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import ir_datasets
import pyterrier as pt
from GenJudge.pooling import Pooling
from bert_score import score as bscore 
import json
import math
from collections import defaultdict
import numpy as np
from scipy.stats import kendalltau, spearmanr
import time
import torch
torch.backends.cuda.matmul.allow_tf32 = True
import os
import logging
import transformers
from transformers import logging as hf_logging

transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)
hf_logging.set_verbosity_error()


class ThresholdOptimizer:
    """Optimize thresholds for F1 score to relevance grade conversion"""
    
    def __init__(self, runs_loc, qrels_loc, k):
        self.runs_loc = runs_loc
        self.qrels_loc = qrels_loc
        self.k = k
        self.ds = self.runs_loc.replace('\\', '/').rstrip('/').split('/')[-1]
        
    def load_qrels(self, required_loc):
        """Load QRELS from file"""
        qrels = pd.read_csv(
            Path(required_loc),
            sep=r"\s+",
            names=["qid", "iter", "docno", "label"]
        )
        qrels["qid"] = qrels["qid"].astype(str)
        qrels["docno"] = qrels["docno"].astype(str)
        return qrels

    def load_pooled_runs(self, file_loc_name):
        """Load pooled runs"""
        pooled_runs = pd.read_csv(file_loc_name, sep='\t')
        pooled_runs["qid"] = pooled_runs["qid"].astype(str)
        pooled_runs["docno"] = pooled_runs["docno"].astype(str)
        run_name = str(file_loc_name).split('\\')[-1]
        print('Run Loaded: ', run_name)
        return pooled_runs

    def load_runs_to_df(self, run_path):
        """Load a single TREC .gz run into a DataFrame"""
        rows = []
        with gzip.open(run_path, 'rt') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue
                qid, _, docno, rank, score, _ = parts
                rows.append({
                    "qid": str(qid),
                    "docno": str(docno),
                    "score": float(score),
                    "rank": int(rank)
                })
        df = pd.DataFrame(rows)
        df = df.sort_values(["qid", "rank"])
        return df

    def filter_qrels(self, qrels, pooled_runs):
        """Filter QRELS to only those in pooled runs"""
        filtered_qrels = qrels.merge(
            pooled_runs[["qid", "docno"]],
            on=["qid", "docno"],
            how="inner"
        )
        return filtered_qrels

    def load_relevant_docs_dict(self, model_responses_loc):
        """Load generated documents for reference"""
        with open(model_responses_loc, 'r', encoding='utf-8') as file:
            data = json.load(file)
        relevant_docs_dict = {}
        for qid in data.keys():
            relevant_docs_dict[qid] = [data[qid]['generated_doc']]
        return relevant_docs_dict

    def build_qrel_texts(self, qrels):
        """Add document text to qrels"""
        if self.ds == 'TREC28':
            dataset_name = "msmarco-passage/trec-dl-2019/judged"
        elif self.ds == 'TREC29':
            dataset_name = "msmarco-passage/trec-dl-2020/judged"
        else:
            raise ValueError(f"Unknown dataset: {self.ds}")

        dataset = ir_datasets.load(dataset_name)
        needed_docnos = set(qrels["docno"].astype(str).unique())
        doc_lookup = {}

        for doc in tqdm(
            dataset.docs_iter(),
            total=len(needed_docnos),
            desc="Loading QREL documents"
        ):
            if doc.doc_id in needed_docnos:
                doc_lookup[doc.doc_id] = doc.text
                if len(doc_lookup) == len(needed_docnos):
                    break

        qrels_with_text = qrels.copy()
        qrels_with_text["text"] = (
            qrels_with_text["docno"]
            .astype(str)
            .map(doc_lookup)
        )
        qrels_with_text = qrels_with_text.dropna(subset=["text"])
        return qrels_with_text

    def compute_f1_scores(self, passage_runs_loc, relevant_docs_dict, qrel_texts, qrels, K):
        """Compute BERT F1 scores for all runs"""
        print("\n" + "="*60)
        print("COMPUTING BERT F1 SCORES")
        print("="*60)
        
        qrel_lookup = defaultdict(dict)
        for _, row in qrels.iterrows():
            qrel_lookup[row["qid"]][row["docno"]] = row["label"]

        all_f1_scores = []
        similarity_lookup_all = {}

        for file_loc in tqdm(
            passage_runs_loc,
            total=len(passage_runs_loc),
            desc="Computing F1 scores"
        ):
            run_df = self.load_runs_to_df(file_loc)
            effective_run_df = run_df[
                run_df["qid"].isin(qrels["qid"].unique())
            ]
            grouped = effective_run_df.groupby("qid")

            all_cands = []
            all_refs = []
            meta_qids = []
            meta_docnos = []
            meta_ref_type = []
            meta_ref_label = []

            for qid, group in grouped:
                ranked_docs = group["docno"].tolist()[:K]
                generated_ref = relevant_docs_dict.get(qid)
                qrel_docs = qrel_texts[qrel_texts["qid"] == qid].to_dict('records')

                for doc in ranked_docs:
                    if generated_ref is not None:
                        all_cands.append(doc)
                        all_refs.append(generated_ref[0])
                        meta_qids.append(qid)
                        meta_docnos.append(doc)
                        meta_ref_type.append("generated")
                        meta_ref_label.append(None)

                    for qrel_doc in qrel_docs:
                        all_cands.append(doc)
                        all_refs.append(qrel_doc["text"])
                        meta_qids.append(qid)
                        meta_docnos.append(doc)
                        meta_ref_type.append("qrel")
                        meta_ref_label.append(qrel_doc["label"])

            if len(all_cands) == 0:
                continue

            # BERTSCORE
            _, _, F1 = bscore(
                all_cands,
                all_refs,
                lang="en",
                device="cuda",
                batch_size=256,
                verbose=False,
            )

            gains = F1.tolist()
            all_f1_scores.extend(gains)

            # Store similarity lookup
            similarity_lookup = defaultdict(list)
            for (qid, docno, score, ref_type, ref_label) in zip(
                meta_qids, meta_docnos, gains, meta_ref_type, meta_ref_label
            ):
                similarity_lookup[(qid, docno)].append({
                    "score": score,
                    "type": ref_type,
                    "label": ref_label
                })

            file_name = str(file_loc).split("\\")[-1]
            similarity_lookup_all[file_name] = similarity_lookup

        return all_f1_scores, similarity_lookup_all

    def evaluate_thresholds(
        self,
        thresholds,
        passage_runs_loc,
        qrel_lookup,
        similarity_lookup_all,
        qrels,
        K,
        threshold_name="Test"
    ):
        """Evaluate a specific threshold configuration"""
        t3, t2, t1 = thresholds

        def bert_to_rel(score):
            if score >= t3:
                return 3
            elif score >= t2:
                return 2
            elif score >= t1:
                return 1
            return 0

        def dcg(scores):
            return sum(
                (2**g - 1) / math.log2(i + 2)
                for i, g in enumerate(scores)
            )

        all_results = {}

        for file_loc in passage_runs_loc:
            file_name = str(file_loc).split("\\")[-1]
            
            run_df = self.load_runs_to_df(file_loc)
            effective_run_df = run_df[
                run_df["qid"].isin(qrels["qid"].unique())
            ]
            grouped = effective_run_df.groupby("qid")

            similarity_lookup = similarity_lookup_all.get(file_name, {})
            grouped_scores = defaultdict(list)

            for qid, group in grouped:
                ranked_docs = group["docno"].tolist()[:K]
                qrels_for_q = qrel_lookup.get(qid, {})
                rels = []

                for docno in ranked_docs:
                    if docno in qrels_for_q:
                        rel = qrels_for_q[docno]
                    else:
                        matches = similarity_lookup.get((qid, docno), [])
                        best_qrel_sim = -1
                        best_qrel_grade = 0
                        generated_sim = 0

                        for m in matches:
                            if m["type"] == "generated":
                                generated_sim = max(generated_sim, m["score"])
                            elif m["type"] == "qrel":
                                if m["score"] > best_qrel_sim:
                                    best_qrel_sim = m["score"]
                                    best_qrel_grade = m["label"]

                        if best_qrel_sim >= t2:
                            rel = best_qrel_grade
                        else:
                            rel = bert_to_rel(
                                max(generated_sim, best_qrel_sim)
                            )

                    rels.append(rel)

                grouped_scores[qid] = rels

            # Calculate nDCG
            ndcgs = []
            for qid, scores in grouped_scores.items():
                dcg_val = dcg(scores)
                ideal_rels = sorted(
                    list(qrel_lookup.get(qid, {}).values()),
                    reverse=True,
                )[:len(scores)]

                if len(ideal_rels) < len(scores):
                    ideal_rels += [0] * (len(scores) - len(ideal_rels))

                idcg_val = dcg(ideal_rels)
                ndcg = (dcg_val / idcg_val if idcg_val > 0 else 0.0)
                ndcgs.append(ndcg)

            all_results[file_name] = round(
                sum(ndcgs) / len(ndcgs) if ndcgs else 0.0, 4
            )

        return all_results

    def optimize_grid_search(
        self,
        passage_runs_loc,
        qrel_lookup,
        similarity_lookup_all,
        qrels,
        official_results,
        K,
        t3_range=None,
        t2_range=None,
        t1_range=None,
        step=0.05
    ):
        """Grid search for optimal thresholds"""
        print("\n" + "="*60)
        print("GRID SEARCH OPTIMIZATION")
        print("="*60)
        
        if t3_range is None:
            t3_range = (0.80, 1.00)
        if t2_range is None:
            t2_range = (0.65, 0.95)
        if t1_range is None:
            t1_range = (0.50, 0.85)

        best_tau = -2.0
        best_thresholds = None
        best_results = None
        results_log = []

        total_combinations = (
            len(np.arange(t3_range[0], t3_range[1] + step, step)) *
            len(np.arange(t2_range[0], t2_range[1] + step, step)) *
            len(np.arange(t1_range[0], t1_range[1] + step, step))
        )

        print(f"Testing {total_combinations} threshold combinations...")

        with tqdm(total=total_combinations, desc="Grid Search") as pbar:
            for t3 in np.arange(t3_range[0], t3_range[1] + step, step):
                for t2 in np.arange(t2_range[0], t2_range[1] + step, step):
                    for t1 in np.arange(t1_range[0], t1_range[1] + step, step):
                        # Ensure t3 > t2 > t1
                        if not (t3 > t2 > t1):
                            pbar.update(1)
                            continue

                        # Evaluate thresholds
                        results = self.evaluate_thresholds(
                            (t3, t2, t1),
                            passage_runs_loc,
                            qrel_lookup,
                            similarity_lookup_all,
                            qrels,
                            K,
                            f"t3={t3:.2f}, t2={t2:.2f}, t1={t1:.2f}"
                        )

                        # Compute correlation with official results
                        runs = list(results.keys())
                        if len(runs) > 1:
                            test_scores = [results[r] for r in runs]
                            official_scores = [official_results.get(r, 0.0) for r in runs]
                            
                            tau, p_val = kendalltau(test_scores, official_scores)
                            
                            results_log.append({
                                "threshold_3": round(t3, 3),
                                "threshold_2": round(t2, 3),
                                "threshold_1": round(t1, 3),
                                "kendall_tau": round(tau, 4),
                                "p_value": round(p_val, 4)
                            })

                            if tau > best_tau:
                                best_tau = tau
                                best_thresholds = (t3, t2, t1)
                                best_results = results

                        pbar.update(1)

        return best_thresholds, best_tau, results_log, best_results

    def optimize_quantile_search(self, all_f1_scores, official_results, passage_runs_loc,
                                  qrel_lookup, similarity_lookup_all, qrels, K):
        """Search for optimal quantile values"""
        print("\n" + "="*60)
        print("QUANTILE SEARCH OPTIMIZATION")
        print("="*60)
        
        best_tau = -2.0
        best_quantiles = None
        best_thresholds = None
        results_log = []

        quantile_combinations = [
            [0.75, 0.50, 0.25],
            [0.70, 0.50, 0.30],
            [0.67, 0.50, 0.33],
            [0.65, 0.50, 0.35],
            [0.60, 0.45, 0.30],
            [0.80, 0.55, 0.30],
            [0.75, 0.55, 0.35],
        ]

        print(f"Testing {len(quantile_combinations)} quantile configurations...")

        all_f1_array = np.array(all_f1_scores)

        for quantiles in tqdm(quantile_combinations, desc="Quantile Search"):
            t3 = np.percentile(all_f1_array, quantiles[0] * 100)
            t2 = np.percentile(all_f1_array, quantiles[1] * 100)
            t1 = np.percentile(all_f1_array, quantiles[2] * 100)

            results = self.evaluate_thresholds(
                (t3, t2, t1),
                passage_runs_loc,
                qrel_lookup,
                similarity_lookup_all,
                qrels,
                K
            )

            runs = list(results.keys())
            if len(runs) > 1:
                test_scores = [results[r] for r in runs]
                official_scores = [official_results.get(r, 0.0) for r in runs]
                
                tau, p_val = kendalltau(test_scores, official_scores)

                results_log.append({
                    "quantiles": f"[{quantiles[0]}, {quantiles[1]}, {quantiles[2]}]",
                    "threshold_3": round(t3, 4),
                    "threshold_2": round(t2, 4),
                    "threshold_1": round(t1, 4),
                    "kendall_tau": round(tau, 4),
                    "p_value": round(p_val, 4)
                })

                if tau > best_tau:
                    best_tau = tau
                    best_quantiles = quantiles
                    best_thresholds = (t3, t2, t1)

        return best_quantiles, best_thresholds, best_tau, results_log

    def save_results(self, results_dict, output_path):
        """Save optimization results to file"""
        output_file = Path(output_path) / f"threshold_optimization_{self.ds}_{self.k}.tsv"
        
        if isinstance(results_dict, list):
            df = pd.DataFrame(results_dict)
        else:
            df = pd.DataFrame([results_dict])
        
        df.to_csv(output_file, index=False, sep='\t')
        print(f"\n✓ Results saved to: {output_file}")
        return output_file

    def run(self, num_runs, optimization_method='both'):
        """Run threshold optimization"""
        print(f"\n{'='*60}")
        print(f"THRESHOLD OPTIMIZATION FOR {self.ds}")
        print(f"{'='*60}")

        # Load data
        file_loc_name = Path(f'GenJudge/results/pooled_runs/pooled_{self.ds}_runs_{self.k}.tsv')
        pooled_runs = self.load_pooled_runs(file_loc_name)
        
        qrels = self.load_qrels(self.qrels_loc)
        filtered_qrels = self.filter_qrels(qrels, pooled_runs)

        pooling_obj = Pooling()
        passage_runs_name, test_passage_runs = pooling_obj.separate_runs()
        if num_runs is not None:
            test_passage_runs = test_passage_runs[:num_runs]

        print(f"\nConfiguration:")
        print(f"  Dataset: {self.ds}")
        print(f"  QRELS: {len(filtered_qrels)}")
        print(f"  Runs to test: {len(test_passage_runs)}")
        print(f"  Cutoff (K): {self.k}")

        # Baseline evaluation with official QRELS
        print("\n" + "="*60)
        print("BASELINE EVALUATION (Official QRELS)")
        print("="*60)
        
        official_results = {}
        for file_loc in tqdm(test_passage_runs, desc="Baseline"):
            run_df = self.load_runs_to_df(file_loc)
            eval_res = pt.Evaluate(
                run_df,
                filtered_qrels,
                metrics=[f"ndcg_cut_{self.k}"]
            )
            file_name = str(file_loc).split('\\')[-1]
            official_results[file_name] = round(eval_res[f"ndcg_cut_{self.k}"], 4)

        print("\nBaseline nDCG@{} scores:".format(self.k))
        for run, score in official_results.items():
            print(f"  {run}: {score:.4f}")

        # Compute F1 scores
        context_file_loc_name = Path(f'GenJudge/results/context_store/context_{self.ds}_runs_{self.k}.json')
        relevant_docs_dict = self.load_relevant_docs_dict(context_file_loc_name)
        qrel_texts = self.build_qrel_texts(filtered_qrels)

        all_f1_scores, similarity_lookup_all = self.compute_f1_scores(
            test_passage_runs, relevant_docs_dict, qrel_texts, filtered_qrels, self.k
        )

        print(f"\nF1 Score Statistics:")
        print(f"  Count: {len(all_f1_scores)}")
        print(f"  Mean: {np.mean(all_f1_scores):.4f}")
        print(f"  Std: {np.std(all_f1_scores):.4f}")
        print(f"  Min: {np.min(all_f1_scores):.4f}")
        print(f"  Max: {np.max(all_f1_scores):.4f}")

        # QREL lookup
        qrel_lookup = defaultdict(dict)
        for _, row in filtered_qrels.iterrows():
            qrel_lookup[row["qid"]][row["docno"]] = row["label"]

        # Optimization
        all_results_log = []
        best_results_summary = {}

        if optimization_method in ['grid', 'both']:
            best_t, best_tau, grid_log, _ = self.optimize_grid_search(
                test_passage_runs, qrel_lookup, similarity_lookup_all, 
                filtered_qrels, official_results, self.k
            )
            all_results_log.extend(grid_log)
            best_results_summary['grid_search'] = {
                'thresholds': best_t,
                'tau': best_tau,
                'log_count': len(grid_log)
            }
            print(f"\n✓ Grid Search Best Tau: {best_tau:.4f}")
            print(f"  Optimal Thresholds: t3={best_t[0]:.4f}, t2={best_t[1]:.4f}, t1={best_t[2]:.4f}")

        if optimization_method in ['quantile', 'both']:
            best_q, best_t, best_tau, quantile_log = self.optimize_quantile_search(
                all_f1_scores, official_results, test_passage_runs,
                qrel_lookup, similarity_lookup_all, filtered_qrels, self.k
            )
            all_results_log.extend(quantile_log)
            best_results_summary['quantile_search'] = {
                'quantiles': best_q,
                'thresholds': best_t,
                'tau': best_tau,
                'log_count': len(quantile_log)
            }
            print(f"\n✓ Quantile Search Best Tau: {best_tau:.4f}")
            print(f"  Optimal Quantiles: {best_q}")
            print(f"  Resulting Thresholds: t3={best_t[0]:.4f}, t2={best_t[1]:.4f}, t1={best_t[2]:.4f}")

        # Save results
        output_dir = Path('GenJudge/results/optimization')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_results(all_results_log, output_dir)
        
        # Save summary
        summary_file = output_dir / f"optimization_summary_{self.ds}_{self.k}.json"
        with open(summary_file, 'w') as f:
            json.dump(best_results_summary, f, indent=2)
        print(f"✓ Summary saved to: {summary_file}")

        return best_results_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize thresholds for nDCG calculation")
    parser.add_argument("--runs_loc", type=str, required=True, help="Path to runs directory")
    parser.add_argument("--qrels_loc", type=str, required=True, help="Path to QRELS file")
    parser.add_argument("--k", type=int, required=True, help="Cutoff for nDCG")
    parser.add_argument("--num_runs", type=int, default=None, help="Number of runs to test (default: All Runs)")
    parser.add_argument("--method", type=str, default='both', 
                        choices=['grid', 'quantile', 'both'],
                        help="Optimization method (default: both)")

    args = parser.parse_args()

    optimizer = ThresholdOptimizer(args.runs_loc, args.qrels_loc, args.k)
    results = optimizer.run(num_runs=args.num_runs, optimization_method=args.method)

    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    print(json.dumps(results, indent=2))
