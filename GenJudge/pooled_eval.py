import gzip
import math
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import ir_datasets
import pyterrier as pt
import pyterrier_alpha as pta
from pyterrier.measures import *
from GenJudge.pooling import Pooling
from GenJudge.modules.utility import Utility
from bert_score import score as bscore 
import json
import math
from collections import defaultdict
import numpy as np
from scipy.stats import kendalltau
import time
import torch
torch.backends.cuda.matmul.allow_tf32 = True
import os
import logging
import transformers
from transformers import logging as hf_logging
from scipy.stats import rankdata
from collections import defaultdict

transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)
hf_logging.set_verbosity_error()

class PooledEval():

    def __init__(self, **kwargs) -> None:
        # parse the command line 
        CLI=argparse.ArgumentParser()

        CLI.add_argument("--runs_loc", type=str)
        CLI.add_argument("--qrels_loc", type=str)
        CLI.add_argument("--k", type=int)
        CLI.add_argument("--eval_depth", type=int)

        args = CLI.parse_args()

        self.runs_loc = args.runs_loc
        self.qrels_loc = args.qrels_loc
        self.k = args.k
        self.eval_depth = args.eval_depth
        # self.eval_depth = 10 # Cutoff Depth for nDCG@k evaluation
        # self.map_eval_depth = 100 # Cutoff Depth for MAP evaluation

        self.ds = self.runs_loc.replace('\\', '/').rstrip('/').split('/')[-1]
        self.utility_obj = Utility()
        self.pooling_obj = Pooling()
        
        # Initialize cache directory
        self.cache_dir = Path('GenJudge/results/evaluation/cache')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache file naming: {model}_{dataset}_inference_cache.json
        self.inference_cache_path = None

    def set_cache_filename(self, model_name="Llama-3.1-8B-Instruct"):
        """Set the inference cache filename based on model and dataset"""
        self.inference_cache_path = self.cache_dir / f"{model_name}_{self.ds}_inference_cache.json"
        return self.inference_cache_path
    
    def load_inference_cache(self):
        """Load cached BERT F1 scores from JSON file"""
        if not self.inference_cache_path or not self.inference_cache_path.exists():
            return {}
        
        try:
            with open(self.inference_cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            print(f"Loaded inference cache from {self.inference_cache_path}")
            print(f"  Total cached (qid, docno) pairs: {len(cache_data.get('lookups', {}))}")
            return cache_data.get('lookups', {})
        except Exception as e:
            print(f"Warning: Could not load inference cache: {e}")
            return {}
    
    def save_inference_cache(self, lookups_dict):
        """Save BERT F1 scores to JSON cache file"""
        if not self.inference_cache_path:
            return
        
        try:
            cache_data = {
                "cache_version": 1,
                "model": "Llama-3.1-8B-Instruct",
                "dataset": self.ds,
                "lookups": lookups_dict
            }
            
            with open(self.inference_cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2)
            
            total_pairs = sum(len(v) if isinstance(v, dict) else len(list(v)) for v in lookups_dict.values())
            print(f"Saved inference cache to {self.inference_cache_path}")
            print(f"  Total cached (qid, docno) pairs: {total_pairs}")
        except Exception as e:
            print(f"Warning: Could not save inference cache: {e}")

    def load_qrels(self, required_loc):
        path = Path(required_loc)
        sep = r"\t" if path.suffix == ".tsv" else r"\s+"

        # Inspect the first line to check if it contains a header
        with open(path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip().split()
        
        # TREC qrel labels or iteration tokens are typically numeric integers.
        # If columns 1 or 3 cannot be cast to int, the first line is likely a header.
        has_header = False
        try:
            int(first_line[3])  # test relevance label
        except (ValueError, IndexError):
            has_header = True

        qrels = pd.read_csv(
            path,
            sep=sep,
            engine="python",
            header=0 if has_header else None,
            names=["qid", "iter", "docno", "label"],
        )

        qrels["qid"] = qrels["qid"].astype(str)
        qrels["docno"] = qrels["docno"].astype(str)
        qrels["label"] = pd.to_numeric(qrels["label"], errors="coerce").fillna(0).astype(int)

        return qrels
    
    def filter_qrels(self, qrels, pooled_runs):
        qrels = qrels.copy()
        pooled_runs = pooled_runs.copy()

        # Ensure qid and docno have compatible types
        qrels["qid"] = qrels["qid"].astype(str)
        qrels["docno"] = qrels["docno"].astype(str)

        pooled_runs["qid"] = pooled_runs["qid"].astype(str)
        pooled_runs["docno"] = pooled_runs["docno"].astype(str)

        filtered_qrels = qrels.merge(
            pooled_runs[["qid", "docno"]],
            on=["qid", "docno"],
            how="inner"
        )

        return filtered_qrels

    def combine_qrels(self, filtered_qrels, gen_qrels):
        """
        Combine filtered and generated qrels, keeping filtered judgments
        when the same (qid, docno) pair appears in both inputs.
        """
        combined_qrels = pd.concat(
            [filtered_qrels, gen_qrels],
            ignore_index=True,
        )

        combined_qrels = combined_qrels.drop_duplicates(
            subset=["qid", "docno"],
            keep="first",
        ).reset_index(drop=True)

        return combined_qrels
    
    def load_pooled_runs(self, file_loc_name):
        # try:
        pooled_runs = pd.read_csv(file_loc_name, sep='\t')
        pooled_runs["qid"] = pooled_runs["qid"].astype(str)
        pooled_runs["docno"] = pooled_runs["docno"].astype(str)
        run_name = str(file_loc_name).split('\\')[-1]
        print('Run Loaded: ',run_name)
        # except:
        #     print('Issue loading Run file.')

        return pooled_runs

    def load_runs_to_df(self, run_path):
        """
        Load a single TREC .gz run into a DataFrame
        """
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

        # Sort to ensure proper ranking
        df = df.sort_values(["qid", "rank"])

        return df

    def evaluate_map_pyterrier(self, desc_text, passage_runs_loc, qrels, K):
        """
        Evaluate MAP using PyTerrier
        """
        results = {}
        per_query_results = {}

        
        if self.ds.lower() == "covid":
            rel_label = 1
            metric_name = f"AP@{K}"
        else:
            rel_label = 2
            metric_name = f"AP(rel={rel_label})@{K}"
        
        for file_loc in tqdm(passage_runs_loc, total=len(passage_runs_loc), desc=f"Computing Pyterrier MAP with {desc_text}"):

            run_df = self.load_runs_to_df(file_loc)

            # Evaluate using PyTerrier
            eval_res = pt.Evaluate(
                run_df,
                qrels,
                metrics=[MAP(rel=rel_label)@K]
            )
            file_name = str(file_loc).split('\\')[-1]
            # print(eval_res)
            results[file_name] = round(eval_res[metric_name],4)

            per_query = pt.Evaluate(
                run_df,
                qrels,
                metrics=[MAP(rel=rel_label)@K],
                perquery=True
            )

            per_query_results[file_name] = [
                v[metric_name]
                for v in per_query.values()
            ]

        return results, per_query_results, metric_name
    
    def evaluate_ndcg_pyterrier(self, desc_text, passage_runs_loc, qrels, K):
        """
        Evaluate nDCG@k using PyTerrier
        """
        results = {}
        per_query_results = {}

        for file_loc in tqdm(passage_runs_loc, total=len(passage_runs_loc), desc=f"Computing Pyterrier nDCG with {desc_text}"):

            run_df = self.load_runs_to_df(file_loc)

            metric_name = f"ndcg_cut_{K}"

            # Evaluate using PyTerrier
            eval_res = pt.Evaluate(
                run_df,
                qrels,
                metrics=[metric_name]
            )
            file_name = str(file_loc).split('\\')[-1]
            results[file_name] = round(eval_res[metric_name],4)

            per_query = pt.Evaluate(
                run_df,
                qrels,
                metrics=[metric_name],
                perquery=True
            )

            per_query_results[file_name] = [
                v[metric_name]
                for v in per_query.values()
            ]

        return results, per_query_results, metric_name

    def ndcg_all_runs(
        self,
        passage_runs_loc,
        relevant_docs_dict,
        qrel_docs_dict,
        qrels,
        K,
        use_dynamic_thresholds=True,
        quantiles=[0.67, 0.50, 0.33],
        per_query_quantiles=True,
    ):
        """
        Calculate nDCG using dynamic thresholds for F1 score conversion.
        
        Args:
            passage_runs_loc: List of run file locations
            relevant_docs_dict: Dict mapping qid -> generated_doc
            qrel_docs_dict: Dict mapping qid -> list of qrel docs with text and label
            qrels: DataFrame with qrel judgments
            K: cutoff for nDCG calculation
            use_dynamic_thresholds: If True, compute thresholds from F1 distribution
            quantiles: Quantile levels for grades [grade3, grade2, grade1]
            per_query_quantiles: If True, compute quantiles per-query; else use global quantiles
        """

        # --------------------------------------------------
        # QREL LOOKUP
        # --------------------------------------------------
        qrel_lookup = defaultdict(dict)
        for _, row in qrels.iterrows():
            qrel_lookup[row["qid"]][row["docno"]] = row["label"]

        all_results = {}
        all_results_pq = {}
        execution_time = {}
        
        # Collect F1 scores (per-query if needed)
        all_f1_scores = []  # For global quantiles
        query_f1_scores = defaultdict(list)  # For per-query quantiles
        
        # Load inference cache
        inference_cache = self.load_inference_cache()
        cache_hits = 0
        cache_misses = 0

        for file_loc in tqdm(
            passage_runs_loc,
            total=len(passage_runs_loc),
            desc="Computing nDCG with BERT F1 scores Thresholds"
        ):

            start_time_per = time.perf_counter()
            run_df = self.load_runs_to_df(file_loc)
            effective_run_df = run_df[
                run_df["qid"].isin(qrels["qid"].unique())
            ]
            grouped = effective_run_df.groupby("qid")

            # --------------------------------------------------
            # BUILD BERTSCORE BATCH (with cache checking)
            # --------------------------------------------------
            all_cands = []
            all_refs = []
            meta_qids = []
            meta_docnos = []
            meta_ref_type = []
            meta_ref_label = []
            cached_results = []  # (qid, docno, score, ref_type, ref_label)

            file_loc_str = str(file_loc)
            file_cache = inference_cache.get(file_loc_str, {}) if file_loc_str in inference_cache else {}

            for qid, group in grouped:
                ranked_docs = group["docno"].tolist()[:K]
                generated_ref = relevant_docs_dict.get(qid)
                qrel_docs = qrel_docs_dict.get(qid, [])

                for doc in ranked_docs:
                    cache_key = f"({qid}, {doc})"
                    cached_matches = file_cache.get(cache_key, None)
                    
                    if cached_matches is not None:
                        # Use cached results
                        for cached_match in cached_matches:
                            cached_results.append((qid, doc, cached_match["score"], cached_match["type"], cached_match.get("label")))
                            all_f1_scores.append(cached_match["score"])
                            query_f1_scores[qid].append(cached_match["score"])
                        cache_hits += len(cached_matches)
                    else:
                        # Need to compute
                        if generated_ref is not None:
                            all_cands.append(doc)
                            all_refs.append(generated_ref)
                            meta_qids.append(qid)
                            meta_docnos.append(doc)
                            meta_ref_type.append("generated")
                            meta_ref_label.append(None)
                            cache_misses += 1

                        for qrel_doc in qrel_docs:
                            all_cands.append(doc)
                            all_refs.append(qrel_doc["text"])
                            meta_qids.append(qid)
                            meta_docnos.append(doc)
                            meta_ref_type.append("qrel")
                            meta_ref_label.append(qrel_doc["label"])
                            cache_misses += 1

            # --------------------------------------------------
            # BERTSCORE (only for cache misses)
            # --------------------------------------------------
            if len(all_cands) > 0:
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
            else:
                gains = []

            # --------------------------------------------------
            # STORE ALL MATCHES & COLLECT PER-QUERY SCORES
            # --------------------------------------------------
            similarity_lookup = defaultdict(list)
            
            # Add computed results
            for (qid, docno, score, ref_type, ref_label) in zip(
                meta_qids, meta_docnos, gains, meta_ref_type, meta_ref_label
            ):
                similarity_lookup[(qid, docno)].append({
                    "score": float(score),
                    "type": ref_type,
                    "label": ref_label
                })
                query_f1_scores[qid].append(float(score))
            
            # Add cached results
            for (qid, docno, score, ref_type, ref_label) in cached_results:
                similarity_lookup[(qid, docno)].append({
                    "score": float(score),
                    "type": ref_type,
                    "label": ref_label
                })

            # Store for later use
            if not hasattr(self, '_similarity_lookup_cache'):
                self._similarity_lookup_cache = {}
            self._similarity_lookup_cache[file_loc_str] = similarity_lookup
            
            # Update cache with newly computed results
            if file_loc_str not in inference_cache:
                inference_cache[file_loc_str] = {}
            for (qid, docno), matches in similarity_lookup.items():
                cache_key = f"({qid}, {docno})"
                if cache_key not in inference_cache[file_loc_str]:
                    inference_cache[file_loc_str][cache_key] = [
                        {"score": float(m["score"]), "type": m["type"], "label": m.get("label")}
                        for m in matches
                    ]

            execution_time[file_loc_str] = time.perf_counter() - start_time_per
        
        # Save updated cache
        self.save_inference_cache(inference_cache)
        if cache_hits > 0 or cache_misses > 0:
            print(f"Cache performance: {cache_hits} hits, {cache_misses} misses\n")

        # --------------------------------------------------
        # COMPUTE DYNAMIC THRESHOLDS (GLOBAL OR PER-QUERY)
        # --------------------------------------------------
        query_thresholds = {}  # Dict[qid] -> {threshold_3, threshold_2, threshold_1}
        
        if use_dynamic_thresholds:
            if per_query_quantiles:
                print(f"\n=== Dynamic Thresholds (Per-Query Quantile-based) ===")
                for qid in query_f1_scores:
                    qid_scores = np.array(query_f1_scores[qid])
                    if len(qid_scores) > 0:
                        t3 = np.percentile(qid_scores, quantiles[0] * 100)
                        t2 = np.percentile(qid_scores, quantiles[1] * 100)
                        t1 = np.percentile(qid_scores, quantiles[2] * 100)
                        query_thresholds[qid] = {
                            'threshold_3': t3,
                            'threshold_2': t2,
                            'threshold_1': t1
                        }
                
                # Print sample queries for verification
                sample_qids = list(query_thresholds.keys())[:3]
                for qid in sample_qids:
                    thresholds = query_thresholds[qid]
                    print(f"Query {qid}:")
                    print(f"  Grade 3 threshold: {thresholds['threshold_3']:.4f} (top {(1-quantiles[0])*100:.1f}%)")
                    print(f"  Grade 2 threshold: {thresholds['threshold_2']:.4f}")
                    print(f"  Grade 1 threshold: {thresholds['threshold_1']:.4f}")
                print(f"... ({len(query_thresholds)} queries total)\n")
            else:
                # Global quantiles
                all_f1_scores_array = np.array(all_f1_scores)
                threshold_3 = np.percentile(all_f1_scores_array, quantiles[0] * 100)
                threshold_2 = np.percentile(all_f1_scores_array, quantiles[1] * 100)
                threshold_1 = np.percentile(all_f1_scores_array, quantiles[2] * 100)
                
                print("\n=== Dynamic Thresholds (Global Quantile-based) ===")
                print(f"Grade 3 threshold: {threshold_3:.4f} (top {(1-quantiles[0])*100:.1f}%)")
                print(f"Grade 2 threshold: {threshold_2:.4f} ({(quantiles[1]-quantiles[2])*100:.1f}% band)")
                print(f"Grade 1 threshold: {threshold_1:.4f} ({(quantiles[2])*100:.1f}% band)")
                print(f"Grade 0 threshold: 0.0000 (bottom {quantiles[2]*100:.1f}%)\n")
        else:
            # Default fixed thresholds
            threshold_3 = 0.95
            threshold_2 = 0.90
            threshold_1 = 0.80

            ## From optimization
            # threshold_3 = 0.8
            # threshold_2 = 0.65
            # threshold_1 = 0.5
            print("\n=== Using Fixed Thresholds ===\n")

        def bert_to_rel_dynamic(score, qid=None):
            """Convert F1 score to relevance grade using dynamic thresholds"""
            # Get thresholds for this query
            if per_query_quantiles and qid in query_thresholds:
                t3 = query_thresholds[qid]['threshold_3']
                t2 = query_thresholds[qid]['threshold_2']
                t1 = query_thresholds[qid]['threshold_1']
            else:
                t3, t2, t1 = threshold_3, threshold_2, threshold_1
            
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

        # --------------------------------------------------
        # COMPUTE NDCG WITH DYNAMIC THRESHOLDS
        # --------------------------------------------------
        for file_loc in tqdm(
            passage_runs_loc,
            total=len(passage_runs_loc),
            desc="Computing nDCG with dynamic thresholds"
        ):
            start_time_per = time.perf_counter()
            run_df = self.load_runs_to_df(file_loc)
            effective_run_df = run_df[
                run_df["qid"].isin(qrels["qid"].unique())
            ]
            grouped = effective_run_df.groupby("qid")
            similarity_lookup = self._similarity_lookup_cache.get(str(file_loc), {})

            # --------------------------------------------------
            # BUILD RELEVANCE GRADES WITH DYNAMIC THRESHOLDS
            # --------------------------------------------------
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

                        # Get query-specific thresholds
                        if per_query_quantiles and qid in query_thresholds:
                            q_t2 = query_thresholds[qid]['threshold_2']
                        else:
                            q_t2 = threshold_2 if use_dynamic_thresholds else 0.90

                        # Relevance propagation from qrels
                        if best_qrel_sim >= q_t2:
                            rel = best_qrel_grade
                        else:
                            rel = bert_to_rel_dynamic(
                                max(generated_sim, best_qrel_sim), qid
                            )

                    rels.append(rel)

                grouped_scores[qid] = rels

            # --------------------------------------------------
            # NDCG CALCULATION
            # --------------------------------------------------
            ndcgs = []

            for qid, scores in grouped_scores.items():
                dcg_val = dcg(scores)
                ideal_rels = sorted(
                    list(qrel_lookup.get(qid, {}).values()),
                    reverse=True,
                )[: len(scores)]

                if len(ideal_rels) < len(scores):
                    ideal_rels += [0] * (len(scores) - len(ideal_rels))

                idcg_val = dcg(ideal_rels)
                ndcg = (dcg_val / idcg_val if idcg_val > 0 else 0.0)
                ndcgs.append(ndcg)

            file_name = str(file_loc).split("\\")[-1]
            all_results[file_name] = round(
                sum(ndcgs) / len(ndcgs) if ndcgs else 0.0, 4
            )
            all_results_pq[file_name] = ndcgs
            execution_time[file_name] = time.perf_counter() - start_time_per

        return (
            all_results,
            all_results_pq,
            execution_time,
            similarity_lookup,
        )

    def ndcg_all_runs_dist(
        self,
        passage_runs_loc,
        bert_f1_scores_dict,
        qrels,
        K,
        use_exponential_gain,
    ):
        """Calculate nDCG@K using a unified ground truth of QRELs and BERT F1 scores.

        - Ground truth priority: Binary/Graded QRELs > BERT F1 scores > 0.0.
        - True IDCG@K: Derived from the top-K highest gains in the entire ground
        truth pool per query.
        - Deterministic ranking: Sorts runs by 'score' descending.

        Args:
            passage_runs_loc: List of run file paths.
            bert_f1_scores_dict: Dict mapping (qid, docno) -> float F1 score [0, 1].
            qrels: DataFrame containing columns ['qid', 'docno', 'label'].
            K: Cutoff rank for evaluation.
            use_exponential_gain: If True, uses (2^gain - 1) / log2(rank + 1). If
            False, uses gain / log2(rank + 1).

        Returns:
            all_results: Dict mapping run_name -> mean nDCG@K score.
            all_results_pq: Dict mapping run_name -> list of per-query nDCG scores.
            execution_time: Dict mapping run_name -> runtime in seconds.
        """

        def dcg_vectorized(gain_array):
            """Vectorized DCG calculation."""
            if len(gain_array) == 0:
                return 0.0

            # Truncate to K
            gains = gain_array[:K]
            ranks = np.arange(1, len(gains) + 1)
            discounts = np.log2(ranks + 1)

            if use_exponential_gain:
                gains = np.power(2.0, gains) - 1.0

            return float(np.sum(gains / discounts))

        # ------------------------------------------------------------------
        # 1. Build Consolidated Ground Truth Lookup
        # ------------------------------------------------------------------
        # Priority: QRELs (binarized or normalized) > BERT F1
        ground_truth = defaultdict(dict)

        # Ingest QRELs (binarized threshold: label >= 1 -> 1.0, else 0.0)
        for _, row in qrels.iterrows():
            qid = str(row["qid"])
            docno = str(row["docno"])
            label = float(row["label"])
            if self.ds=='covid':
                ground_truth[qid][docno] = 1.0 if label >= 2.0 else 0.0 # COVID
            else:
                ground_truth[qid][docno] = 1.0 if label >= 1.0 else 0.0 # DL-2019 DL-2020

        # # Ingest QRELs (normalized 0-1 scale)
        # min_label = qrels["label"].min()
        # max_label = qrels["label"].max()

        # for _, row in qrels.iterrows():

        #     qid = str(row["qid"])
        #     docno = str(row["docno"])
        #     label = float(row["label"])

        #     normalized_label = (label - min_label) / (max_label - min_label)

        #     ground_truth[qid][docno] = normalized_label

        # Ingest BERT F1 scores for documents not judged in QRELs
        for (qid, docno), f1 in bert_f1_scores_dict.items():
            qid_str, docno_str = str(qid), str(docno)
            if docno_str not in ground_truth[qid_str]:
                ground_truth[qid_str][docno_str] = float(f1)

        # ------------------------------------------------------------------
        # 2. Precompute Theoretical IDCG@K per Query
        # ------------------------------------------------------------------
        ideal_dcg_per_query = {}
        for qid, docs_dict in ground_truth.items():
            # Retrieve all candidate gains for this query, sort descending
            all_gains = sorted(docs_dict.values(), reverse=True)
            top_k_gains = np.array(all_gains[:K], dtype=np.float32)
            ideal_dcg_per_query[qid] = dcg_vectorized(top_k_gains)

        all_results = {}
        all_results_pq = {}
        execution_time = {}

        # ------------------------------------------------------------------
        # 3. Evaluate Each Run File
        # ------------------------------------------------------------------
        for file_loc in tqdm(
            passage_runs_loc,
            total=len(passage_runs_loc),
            desc="Computing nDCG@K",
        ):
            start_time_per = time.perf_counter()
            run_df = self.load_runs_to_df(file_loc)

            # Ensure consistent types
            run_df["qid"] = run_df["qid"].astype(str)
            run_df["docno"] = run_df["docno"].astype(str)

            # Run diagnostics per file
            docs_from_qrels = 0
            docs_from_f1 = 0
            docs_zero = 0
            total_docs_evaluated = 0

            # Sort explicitly by score descending to guarantee ranking order
            run_df = run_df.sort_values(by=["qid", "score"], ascending=[True, False])
            grouped = run_df.groupby("qid")

            ndcgs = []

            for qid, group in grouped:
                # Sliced top-K documents retrieved by this system
                ranked_docs = group["docno"].tolist()[:K]
                q_truth = ground_truth.get(qid, {})

                # Assign gain scores
                retrieved_gains = []
                for docno in ranked_docs:
                    gain = q_truth.get(docno, 0.0)
                    retrieved_gains.append(gain)

                    # Diagnostic tracking
                    if docno in qrels.loc[qrels["qid"] == qid, "docno"].values:
                        docs_from_qrels += 1
                    elif gain > 0.0:
                        docs_from_f1 += 1
                    else:
                        docs_zero += 1

                total_docs_evaluated += len(ranked_docs)

                # Compute Actual DCG
                gains_arr = np.array(retrieved_gains, dtype=np.float32)
                dcg_val = dcg_vectorized(gains_arr)

                # Retrieve True Theoretical IDCG
                idcg_val = ideal_dcg_per_query.get(qid, 0.0)

                # Calculate nDCG
                ndcg = (dcg_val / idcg_val) if idcg_val > 1e-10 else 0.0
                ndcgs.append(float(ndcg))

            file_name = str(file_loc).replace("\\", "/").split("/")[-1]
            all_results[file_name] = round(
                float(np.mean(ndcgs)) if ndcgs else 0.0, 4
            )
            all_results_pq[file_name] = ndcgs
            execution_time[file_name] = time.perf_counter() - start_time_per

        return all_results, all_results_pq, execution_time

    def map_all_runs_dist(
            self,
            passage_runs_loc,
            bert_f1_scores_dict,
            qrels,
            K,
            bert_f1_threshold=0.8,
        ):
            """Calculate MAP@K using QRELs and thresholded BERT F1 scores.

            - QRELs have priority over BERT F1 scores.
            - QREL relevance:
                COVID: label >= 2 -> relevant
            - For documents not present in QRELs:
                BERT F1 >= bert_f1_threshold -> relevant
            - MAP@K is computed using binary relevance.
            - Deterministic ranking: Sorts runs by 'score' descending.

            Args:
                passage_runs_loc: List of run file paths.
                bert_f1_scores_dict: Dict mapping (qid, docno) -> float F1 score [0, 1].
                qrels: DataFrame containing columns ['qid', 'docno', 'label'].
                K: Cutoff rank for evaluation.
                use_exponential_gain: Kept for API compatibility; not used for MAP.
                bert_f1_threshold: Threshold above which a BERT F1 score is
                    considered relevant.

            Returns:
                all_results: Dict mapping run_name -> mean MAP@K score.
                all_results_pq: Dict mapping run_name -> list of per-query AP@K scores.
                execution_time: Dict mapping run_name -> runtime in seconds.
            """

            # ------------------------------------------------------------------
            # 1. Build Consolidated Ground Truth Lookup
            # ------------------------------------------------------------------
            ground_truth = defaultdict(dict)

            # Ingest QRELs
            for _, row in qrels.iterrows():
                qid = str(row["qid"])
                docno = str(row["docno"])
                label = float(row["label"])

                # COVID
                ground_truth[qid][docno] = 1.0 if label >= 2.0 else 0.0

                # For TREC DL:
                # ground_truth[qid][docno] = 1.0 if label >= 1.0 else 0.0

            # Ingest BERT F1 scores for documents not judged in QRELs
            for (qid, docno), f1 in bert_f1_scores_dict.items():
                qid_str = str(qid)
                docno_str = str(docno)

                if docno_str not in ground_truth[qid_str]:
                    ground_truth[qid_str][docno_str] = float(f1)

            # ------------------------------------------------------------------
            # 2. Precompute Number of Relevant Documents per Query
            # ------------------------------------------------------------------
            # QREL relevance is already encoded as 0/1.
            # BERT relevance is determined using bert_f1_threshold.
            relevant_docs_per_query = {}

            for qid, docs_dict in ground_truth.items():
                relevant_docs_per_query[qid] = sum(
                    1
                    for gain in docs_dict.values()
                    if gain >= bert_f1_threshold
                )

            # ------------------------------------------------------------------
            # 3. Evaluate Each Run File
            # ------------------------------------------------------------------
            all_results = {}
            all_results_pq = {}
            execution_time = {}

            for file_loc in tqdm(
                passage_runs_loc,
                total=len(passage_runs_loc),
                desc="Computing MAP@K",
            ):
                start_time_per = time.perf_counter()

                run_df = self.load_runs_to_df(file_loc)

                # Ensure consistent types
                run_df["qid"] = run_df["qid"].astype(str)
                run_df["docno"] = run_df["docno"].astype(str)

                # Run diagnostics
                docs_from_qrels = 0
                docs_from_f1 = 0
                docs_zero = 0
                total_docs_evaluated = 0

                # Sort explicitly by score descending
                run_df = run_df.sort_values(
                    by=["qid", "score"],
                    ascending=[True, False]
                )

                grouped = run_df.groupby("qid")

                aps = []

                for qid, group in grouped:

                    # Top-K documents retrieved by this system
                    ranked_docs = group["docno"].tolist()[:K]

                    q_truth = ground_truth.get(qid, {})

                    # Number of relevant documents in the complete ground truth
                    total_relevant = relevant_docs_per_query.get(qid, 0)

                    if total_relevant == 0:
                        aps.append(0.0)
                        continue

                    # ----------------------------------------------------------
                    # Calculate AP@K
                    # ----------------------------------------------------------
                    num_relevant_retrieved = 0
                    precision_sum = 0.0

                    for rank, docno in enumerate(ranked_docs, start=1):

                        gain = q_truth.get(docno, 0.0)

                        # Check whether document came from QRELs
                        if docno in qrels.loc[
                            qrels["qid"] == qid, "docno"
                        ].values:

                            docs_from_qrels += 1

                            # QREL relevance
                            is_relevant = gain > 0.0

                        else:
                            # BERT F1 relevance
                            if gain >= bert_f1_threshold:
                                docs_from_f1 += 1
                                is_relevant = True
                            else:
                                docs_zero += 1
                                is_relevant = False

                        # ------------------------------------------------------
                        # Binary relevance for MAP
                        # ------------------------------------------------------
                        if is_relevant:

                            num_relevant_retrieved += 1

                            precision_at_rank = (
                                num_relevant_retrieved / rank
                            )

                            precision_sum += precision_at_rank

                    total_docs_evaluated += len(ranked_docs)

                    # ----------------------------------------------------------
                    # AP@K
                    # ----------------------------------------------------------
                    denominator = min(total_relevant, K)

                    ap = (
                        precision_sum / denominator
                        if denominator > 0
                        else 0.0
                    )

                    aps.append(float(ap))

                # ------------------------------------------------------------------
                # 4. MAP@K
                # ------------------------------------------------------------------
                map_score = (
                    float(np.mean(aps))
                    if aps
                    else 0.0
                )

                file_name = (
                    str(file_loc)
                    .replace("\\", "/")
                    .split("/")[-1]
                )

                all_results[file_name] = round(
                    map_score,
                    4
                )

                all_results_pq[file_name] = aps

                execution_time[file_name] = (
                    time.perf_counter() - start_time_per
                )

            return all_results, all_results_pq, execution_time

    def load_relevant_docs_dict(self, model_responses_loc):
        
        # Open the file and load its content
        with open(model_responses_loc, 'r', encoding='utf-8') as file:
            data = json.load(file)

        relevant_docs_dict = {}

        for qid in data.keys():
            relevant_docs_dict[qid]=[data[qid]['generated_doc']]
        
        return relevant_docs_dict

    def rbo(self, ranks_a, ranks_b, p=0.9):
        """
        Calculate Rank-Biased Overlap (RBO) between two ranked lists.

        Parameters
        ----------
        ranks_a : list
            First ranked list.
        ranks_b : list
            Second ranked list.
        p : float, optional
            Persistence parameter. Higher values give more weight to deeper
            ranks. Commonly 0.9 or 0.99.

        Returns
        -------
        float
            RBO score between 0 and 1.
        """

        try:
            p = float(p)
        except (TypeError, ValueError):
            raise ValueError(f"p must be a number, got {p!r}")

        if not 0 <= p < 1:
            raise ValueError("p must be between 0 and 1.")

        if len(ranks_a) != len(ranks_b):
            raise ValueError(
                f"Length mismatch: {len(ranks_a)} vs {len(ranks_b)}"
            )

        if len(ranks_a) == 0:
            return 1.0

        # Convert the ranking arrays into ordered item lists.
        # Lower rank value = higher position.
        list_a = sorted(range(len(ranks_a)), key=lambda i: ranks_a[i])
        list_b = sorted(range(len(ranks_b)), key=lambda i: ranks_b[i])

        seen_a = set()
        seen_b = set()

        rbo_score = 0.0

        for depth, (item_a, item_b) in enumerate(
            zip(list_a, list_b), start=1
        ):
            seen_a.add(item_a)
            seen_b.add(item_b)
            overlap = len(seen_a & seen_b)
            agreement = overlap / depth
            rbo_score += agreement * (p ** (depth - 1)) * (1 - p)

        return rbo_score
    
    def save_kendall_taus(
        self,
        summary_file,
        evaluation,
        original_filtered_tau,
        original_llmjudge_tau,
        original_llmjudge_lor_tau,
        original_llmjudge_lfs_tau,
        original_generated_tau,
        generated_dist_tau,
        filtered_qrels_count,
        llmjudge_qrels_count
        ):
        """
        Add or update Kendall tau values for an evaluation.

        ORI (Original QRELS) 
        RDP (Reduced Depth Pooling QRELS)
        LMJ (LLM Judgment QRELS) 
        RBE (Reference Based Evaluation) - Ours
        RBD (Reference Based Evaluation with Distribution) - Ours
        

        """
        new_row = {
            "Evaluation": evaluation,
            "ORI_RDP": original_filtered_tau,
            "ORI_LMJ": original_llmjudge_tau,
            "ORI_LOR": original_llmjudge_lor_tau,
            "ORI_LFS": original_llmjudge_lfs_tau,
            "ORI_RBE": original_generated_tau,
            "ORI_RBD": generated_dist_tau,
            "RDP_QRELS_Count": filtered_qrels_count,
            "LMJ_QRELS_Count": llmjudge_qrels_count,
        }

        if os.path.exists(summary_file):
            df = pd.read_csv(summary_file, sep='\t')

            mask = df["Evaluation"] == evaluation

            if mask.any():
                # Update existing row
                df.loc[mask, "ORI_RDP"] = original_filtered_tau
                df.loc[mask, "ORI_LMJ"] = original_llmjudge_tau
                df.loc[mask, "ORI_LOR"] = original_llmjudge_lor_tau
                df.loc[mask, "ORI_LFS"] = original_llmjudge_lfs_tau
                df.loc[mask, "ORI_RBE"] = original_generated_tau
                df.loc[mask, "ORI_RBD"] = generated_dist_tau
                df.loc[mask, "RDP_QRELS_Count"] = filtered_qrels_count
                df.loc[mask, "LMJ_QRELS_Count"] = llmjudge_qrels_count
            else:
                # Add new row
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            # Create new file
            df = pd.DataFrame([new_row])

        df.to_csv(summary_file, index=False,  sep='\t')

    def save_rbo_results(
        self,
        summary_file,
        evaluation,
        original_filtered_rbo,
        original_llmjudge_rbo,
        original_llmjudge_lor_rbo,
        original_llmjudge_lfs_rbo,
        original_generated_rbo,
        generated_dist_rbo,
        filtered_qrels_count,
        llmjudge_qrels_count
    ):
        """
        Add or update RBO values for an evaluation.

        ORI (Original QRELS)
        RDP (Reduced Depth Pooling QRELS)
        LMJ (LLM Judgment QRELS)
        LOR (LLM Judgment with LOR)
        LFS (LLM Judgment with LFS)
        RBE (Reference Based Evaluation) - Ours
        RBD (Reference Based Evaluation with Distribution) - Ours
        """

        new_row = {
            "Evaluation": evaluation,
            "ORI_RDP": original_filtered_rbo,
            "ORI_LMJ": original_llmjudge_rbo,
            "ORI_LOR": original_llmjudge_lor_rbo,
            "ORI_LFS": original_llmjudge_lfs_rbo,
            "ORI_RBE": original_generated_rbo,
            "ORI_RBD": generated_dist_rbo,
            "RDP_QRELS_Count": filtered_qrels_count,
            "LMJ_QRELS_Count": llmjudge_qrels_count,
        }

        if os.path.exists(summary_file):
            df = pd.read_csv(summary_file, sep='\t')

            mask = df["Evaluation"] == evaluation

            if mask.any():
                # Update existing row
                df.loc[mask, "ORI_RDP"] = original_filtered_rbo
                df.loc[mask, "ORI_LMJ"] = original_llmjudge_rbo
                df.loc[mask, "ORI_LOR"] = original_llmjudge_lor_rbo
                df.loc[mask, "ORI_LFS"] = original_llmjudge_lfs_rbo
                df.loc[mask, "ORI_RBE"] = original_generated_rbo
                df.loc[mask, "ORI_RBD"] = generated_dist_rbo
                df.loc[mask, "RDP_QRELS_Count"] = filtered_qrels_count
                df.loc[mask, "LMJ_QRELS_Count"] = llmjudge_qrels_count
            else:
                # Add new row
                df = pd.concat(
                    [df, pd.DataFrame([new_row])],
                    ignore_index=True
                )
        else:
            # Create new file
            df = pd.DataFrame([new_row])

        df.to_csv(summary_file, index=False, sep='\t')

    
    def build_qrel_texts(self, qrels):
        """
        Add document text to qrels by looking up all qrel docnos
        from the IR-Datasets corpus.
        """

        # Dataset Name Mapping
        if self.ds=='TREC28' or self.ds=='TREC19':
            dataset_name = "msmarco-passage/trec-dl-2019/judged"
        elif self.ds=='TREC29' or self.ds=='TREC20':
            dataset_name = "msmarco-passage/trec-dl-2020/judged"
        elif self.ds=='TRECCOVID':
            dataset_name = "beir/trec-covid"
        elif self.ds=='TREC30':
            dataset_name = "msmarco-passage/dev/small"

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

        qrels_with_text = qrels_with_text.dropna(
            subset=["text"]
        )

        return qrels_with_text


    def main(self):
        # Record the starting time
        start_time = time.perf_counter()

        print(f"Starting pooled evaluation for dataset: {self.ds}, pooling top-k: {self.k}, depth: {self.eval_depth}")

        file_loc_name = Path(f'GenJudge/results/pooled_runs/pooled_{self.ds}_runs_{self.k}.tsv')

        pooled_runs = self.load_pooled_runs(file_loc_name)

        # Dataset Name Mapping
        dataset_name = self.utility_obj.get_dataset_name(self.ds)
        
        # Get QRELS from Pyterrier Datasets
        pyt_ds = pt.datasets.get_dataset('irds:'+dataset_name)
        qrels = pyt_ds.get_qrels()

        qrels = self.load_qrels(self.qrels_loc)
        filtered_qrels = self.filter_qrels(qrels, pooled_runs)

        taj_qrels_loc = Path(f'GenJudge/results/taj_qrels/taj_qrels_{self.ds}_runs_{self.k}.qrels') # topic-adapters-as-judges
        taj_qrels = self.load_qrels(taj_qrels_loc)
        
        gen_qrels_loc = Path(f'GenJudge/results/gen_qrels/gen_qrels_{self.ds}_runs_{self.k}.qrels')
        gen_qrels = self.load_qrels(gen_qrels_loc)

        fs_gen_qrels_loc = Path(f'GenJudge/results/gen_qrels/gen_qrels_{self.ds}_runs_{self.k}_few1.qrels')
        fs_gen_qrels = self.load_qrels(fs_gen_qrels_loc)

        print('Total QRELS: ', len(qrels))
        print('Filtered QRELS: ', len(filtered_qrels))
        print('Total Gen QRELS: ', len(gen_qrels))
        print('Total Fewshot Gen QRELS: ', len(fs_gen_qrels))

        # print(gen_qrels.head())

        passage_runs_name, test_passage_runs = self.pooling_obj.separate_runs()

        # test_passage_runs = test_passage_runs[:5] ## testing only
        # scores = self.evaluate_ndcg_pyterrier(test_passage_runs, qrels, 10)

        '''
        ###################  nDCG Evaluation   ###################
        '''

        

        ## nDCG with official runs and QRELS
        desc_text='official QRELS'
        scores_ori, scores_pq_ori, metric_name = self.evaluate_ndcg_pyterrier(desc_text, test_passage_runs, qrels, self.eval_depth)
        scores_df_ori = pd.DataFrame(list(scores_ori.items()), columns=["run", metric_name])

        map_scores_ori, map_scores_pq_ori, map_metric_name = self.evaluate_map_pyterrier(desc_text, test_passage_runs, qrels, self.eval_depth)
        map_scores_df_ori = pd.DataFrame(list(map_scores_ori.items()), columns=["run", map_metric_name])

        ## nDCG with official runs and filtered QRELS
        desc_text='filtered QRELS'
        scores, scores_pq, metric_name = self.evaluate_ndcg_pyterrier(desc_text, test_passage_runs, filtered_qrels, self.eval_depth)
        scores_df = pd.DataFrame(list(scores.items()), columns=["run", metric_name])

        map_scores, map_scores_pq, map_metric_name = self.evaluate_map_pyterrier(desc_text, test_passage_runs, filtered_qrels, self.eval_depth)
        map_scores_df = pd.DataFrame(list(map_scores.items()), columns=["run", map_metric_name])
        
        ## nDCG with official runs for Topic Adapter Judge with original QRELS as priority
        desc_text='Topic Adapter Judge QRELS'
        
        final_taj_qrels = self.combine_qrels(filtered_qrels, taj_qrels)

        scores_gen, scores_pq_gen, metric_name = self.evaluate_ndcg_pyterrier(desc_text, test_passage_runs, final_taj_qrels, self.eval_depth)
        scores_df_gen = pd.DataFrame(list(scores_gen.items()), columns=["run", metric_name])

        map_scores_gen, map_scores_pq_gen, map_metric_name = self.evaluate_map_pyterrier(desc_text, test_passage_runs, final_taj_qrels, self.eval_depth)
        map_scores_df_gen = pd.DataFrame(list(map_scores_gen.items()), columns=["run", map_metric_name])

        ## nDCG with official runs and LLM as Judge QRELS with original QRELS as priority
        desc_text='LLM generated QRELS with original QRELS as priority'
        lor_qrels = self.combine_qrels(filtered_qrels, gen_qrels)
        scores_lor, scores_pq_lor, metric_name = self.evaluate_ndcg_pyterrier(desc_text, test_passage_runs, lor_qrels, self.eval_depth)
        scores_df_lor = pd.DataFrame(list(scores_lor.items()), columns=["run", metric_name])

        map_scores_lor, map_scores_pq_lor, map_metric_name = self.evaluate_map_pyterrier(desc_text, test_passage_runs, lor_qrels, self.eval_depth)
        map_scores_df_lor = pd.DataFrame(list(map_scores_lor.items()), columns=["run", map_metric_name])

        ## nDCG with official runs and LLM as Judge QRELS with fewshot from original QRELS
        desc_text='Fewshot LLM generated QRELS'
        scores_lfs, scores_pq_lfs, metric_name = self.evaluate_ndcg_pyterrier(desc_text, test_passage_runs, fs_gen_qrels, self.eval_depth)
        scores_df_lfs = pd.DataFrame(list(scores_lfs.items()), columns=["run", metric_name])

        map_scores_lfs, map_scores_pq_lfs, map_metric_name = self.evaluate_map_pyterrier(desc_text, test_passage_runs, fs_gen_qrels, self.eval_depth)
        map_scores_df_lfs = pd.DataFrame(list(map_scores_lfs.items()), columns=["run", map_metric_name])

        ## Threshold based nDCG with official runs and generated judgment
        context_file_loc_name = Path(f'GenJudge/results/context_store/context_{self.ds}_runs_{self.k}.json')
        relevant_docs_dict = self.load_relevant_docs_dict(context_file_loc_name)
        qrel_texts = self.build_qrel_texts(filtered_qrels)
        
        # Initialize inference cache filename
        self.set_cache_filename("BERTScore")
        
        # Use dynamic thresholds (per-query quantile-based)
        gen_scores, gen_scores_pq, execution_time, similarity_lookup = self.ndcg_all_runs(
            test_passage_runs, 
            relevant_docs_dict, 
            qrel_texts, 
            filtered_qrels, 
            self.eval_depth,
            use_dynamic_thresholds=False,
            quantiles=[0.67, 0.50, 0.33],  # Adjust as needed
            per_query_quantiles=True       # Enable per-query thresholds
        )
        gen_scores_df = pd.DataFrame(list(gen_scores.items()), columns=["run", metric_name])
        execution_time_df = pd.DataFrame(list(execution_time.items()), columns=["run", "Execution Time"])

        ## Extract BERT F1 scores from similarity_lookup and build dict
        ## Aggregate scores from ALL runs, not just the last one
        bert_f1_scores_dict = {}
        
        # Use cached similarity lookups from all runs
        if hasattr(self, '_similarity_lookup_cache'):
            for file_loc_str, lookup in self._similarity_lookup_cache.items():
                for (qid, docno), matches in lookup.items():
                    # Take maximum F1 score across all references for this doc-query pair
                    max_f1 = max(m["score"] for m in matches) if matches else 0.0
                    
                    # Keep the maximum F1 score across all runs for each (qid, docno)
                    if (qid, docno) not in bert_f1_scores_dict:
                        bert_f1_scores_dict[(qid, docno)] = max_f1
                    else:
                        bert_f1_scores_dict[(qid, docno)] = max(bert_f1_scores_dict[(qid, docno)], max_f1)
        else:
            # Fallback to single similarity_lookup if cache not available
            for (qid, docno), matches in similarity_lookup.items():
                max_f1 = max(m["score"] for m in matches) if matches else 0.0
                bert_f1_scores_dict[(qid, docno)] = max_f1

        print(f"Built bert_f1_scores_dict with {len(bert_f1_scores_dict)} (qid, docno) pairs from all runs")

        ## Distribution based nDCG with official runs and F1 scores
        gen_scores_dist, gen_scores_pq_dist, execution_time_dist = self.ndcg_all_runs_dist(test_passage_runs, bert_f1_scores_dict, filtered_qrels, self.eval_depth, True)
        gen_scores_df_dist = pd.DataFrame(list(gen_scores_dist.items()), columns=["run", metric_name])
        execution_time_df_dist = pd.DataFrame(list(execution_time_dist.items()), columns=["run", "Execution Time"])

        ## Distribution based MAP 
        map_gen_scores_dist, map_gen_scores_pq_dist, map_execution_time_dist = self.map_all_runs_dist(test_passage_runs, bert_f1_scores_dict, filtered_qrels, self.eval_depth)
        map_gen_scores_df_dist = pd.DataFrame(list(map_gen_scores_dist.items()), columns=["run", map_metric_name])
        map_execution_time_df_dist = pd.DataFrame(list(map_execution_time_dist.items()), columns=["run", "Execution Time"])


        ## Calculate Kendall's Tau for nDCG
        ranks_original = scores_df_ori[metric_name].rank(ascending=False)
        ranks_filtered = scores_df[metric_name].rank(ascending=False)
        ranks_llmjudge = scores_df_gen[metric_name].rank(ascending=False)
        ranks_llmjudge_lor = scores_df_lor[metric_name].rank(ascending=False)
        ranks_llmjudge_lfs = scores_df_lfs[metric_name].rank(ascending=False)
        ranks_generated = gen_scores_df[metric_name].rank(ascending=False)
        ranks_generated_dist = gen_scores_df_dist[metric_name].rank(ascending=False)

        tau_of, p_of = kendalltau(ranks_original, ranks_filtered)
        tau_lg, p_lg = kendalltau(ranks_original, ranks_llmjudge)
        tau_lo, p_lo = kendalltau(ranks_original, ranks_llmjudge_lor)
        tau_lf, p_lf = kendalltau(ranks_original, ranks_llmjudge_lfs)
        tau_og, p_og = kendalltau(ranks_original, ranks_generated)
        tau_gd, p_gd = kendalltau(ranks_original, ranks_generated_dist)
        
        # Save Kendall's Tau values to a summary file
        summary_path = Path(f'GenJudge/results/evaluation/Summary{self.eval_depth}.tsv')
        num_runs = len(test_passage_runs)
        eval_name = f"eval_{self.ds}_runs_K{self.k}_R{num_runs}"
        self.save_kendall_taus(summary_path,eval_name, tau_of, tau_lg, tau_lo, tau_lf, tau_og, tau_gd, len(filtered_qrels),len(gen_qrels))

        ## Calculate RBO for nDCG
        rbo_of = self.rbo(ranks_original, ranks_filtered)
        rbo_lg = self.rbo(ranks_original, ranks_llmjudge)
        rbo_lo = self.rbo(ranks_original, ranks_llmjudge_lor)
        rbo_lf = self.rbo(ranks_original, ranks_llmjudge_lfs)
        rbo_og = self.rbo(ranks_original, ranks_generated)
        rbo_gd = self.rbo(ranks_original, ranks_generated_dist)

        # Save RBO values to a summary file
        summary_path_rbo = Path(f'GenJudge/results/evaluation/Summary{self.eval_depth}_RBO.tsv')
        eval_name = f"eval_{self.ds}_runs_K{self.k}_R{num_runs}"
        self.save_rbo_results(summary_path_rbo,eval_name, rbo_of, rbo_lg, rbo_lo, rbo_lf, rbo_og, rbo_gd, len(filtered_qrels),len(gen_qrels))


        final_df = (
            scores_df_ori.rename(columns={metric_name: f"{metric_name}_original_QRELS"})
            .merge(
                scores_df.rename(columns={metric_name: f"{metric_name}_RDP_QRELS"}),
                on="run",
                how="inner"
            )
            .merge(
                scores_df_gen.rename(columns={metric_name: f"{metric_name}_LLMJ"}),
                on="run",
                how="inner"
            )
            .merge(
                scores_df_lor.rename(columns={metric_name: f"{metric_name}_LOR"}),
                on="run",
                how="inner"
            )
            .merge(
                scores_df_lfs.rename(columns={metric_name: f"{metric_name}_LFS"}),
                on="run",
                how="inner"
            )
            .merge(
                gen_scores_df.rename(columns={metric_name: f"{metric_name}_RBE"}),
                on="run",
                how="inner"
            )
            .merge(
                gen_scores_df_dist.rename(columns={metric_name: f"{metric_name}_RBD"}),
                on="run",
                how="inner"
            )
            .merge(
                execution_time_df,
                on="run",
                how="inner"
            )
        )

        # print(final_df)
        print('Total Runs: ',len(scores_df))


        # Save to a TSV file
        file_loc_name = Path(f'GenJudge/results/evaluation/eval{self.eval_depth}_{self.ds}_runs_{self.k}.tsv')
        final_df.to_csv(file_loc_name, index=False, sep='\t')
        print('File Saved: ',file_loc_name)


        '''
                ###################  MAP Evaluation   ###################
        '''
        ## Calculate Kendall's Tau for nDCG
        ### Overall Rank Basis
        map_ranks_original = map_scores_df_ori[map_metric_name].rank(ascending=False)
        map_ranks_filtered = map_scores_df[map_metric_name].rank(ascending=False)
        map_ranks_llmjudge = map_scores_df_gen[map_metric_name].rank(ascending=False)
        map_ranks_llmjudge_lor = map_scores_df_lor[map_metric_name].rank(ascending=False)
        map_ranks_llmjudge_lfs = map_scores_df_lfs[map_metric_name].rank(ascending=False)
        map_ranks_generated = map_gen_scores_df_dist[map_metric_name].rank(ascending=False)
        map_ranks_generated_dist = map_gen_scores_df_dist[map_metric_name].rank(ascending=False)

        map_tau_of, p_of = kendalltau(map_ranks_original, map_ranks_filtered)
        map_tau_lg, p_lg = kendalltau(map_ranks_original, map_ranks_llmjudge)
        map_tau_lo, p_lo = kendalltau(map_ranks_original, map_ranks_llmjudge_lor)
        map_tau_lf, p_lf = kendalltau(map_ranks_original, map_ranks_llmjudge_lfs)
        map_tau_og, p_og = kendalltau(map_ranks_original, map_ranks_generated)
        map_tau_gd, p_gd = kendalltau(map_ranks_original, map_ranks_generated_dist)

        map_summary_path = Path(f'GenJudge/results/evaluation/Summary{self.eval_depth}-MAP.tsv')
        map_num_runs = len(test_passage_runs)
        map_eval_name = f"evalmap{self.eval_depth}_{self.ds}_runs_K{self.k}_R{num_runs}"
        self.save_kendall_taus(map_summary_path,map_eval_name, map_tau_of, map_tau_lg, map_tau_lo, map_tau_lf, map_tau_og, map_tau_gd, len(filtered_qrels),len(gen_qrels))

        '''
                ###################  RBO Evaluation   ###################
        '''
        ## Calculate RBO for nDCG
        map_rbo_of = self.rbo(map_ranks_original, map_ranks_filtered)
        map_rbo_lg = self.rbo(map_ranks_original, map_ranks_llmjudge)
        map_rbo_lo = self.rbo(map_ranks_original, map_ranks_llmjudge_lor)
        map_rbo_lf = self.rbo(map_ranks_original, map_ranks_llmjudge_lfs)
        map_rbo_og = self.rbo(map_ranks_original, map_ranks_generated)
        map_rbo_gd = self.rbo(map_ranks_original, map_ranks_generated_dist)

        # Save RBO values to a summary file
        map_summary_path_rbo = Path(f'GenJudge/results/evaluation/Summary{self.eval_depth}-MAP_RBO.tsv')
        self.save_rbo_results(map_summary_path_rbo,map_eval_name, map_rbo_of, map_rbo_lg, map_rbo_lo, map_rbo_lf, map_rbo_og, map_rbo_gd, len(filtered_qrels),len(gen_qrels))


        final_df = (
            map_scores_df_ori.rename(columns={map_metric_name: f"{map_metric_name}_original_QRELS"})
            .merge(
                map_scores_df.rename(columns={map_metric_name: f"{map_metric_name}_RDP_QRELS"}),
                on="run",
                how="inner"
            )
            .merge(
                map_scores_df_gen.rename(columns={map_metric_name: f"{map_metric_name}_LLMJ"}),
                on="run",
                how="inner"
            )
            .merge(
                map_scores_df_lor.rename(columns={map_metric_name: f"{map_metric_name}_LOR"}),
                on="run",
                how="inner"
            )
            .merge(
                map_scores_df_lfs.rename(columns={map_metric_name: f"{map_metric_name}_LFS"}),
                on="run",
                how="inner"
            )
            .merge(
                map_gen_scores_df_dist.rename(columns={map_metric_name: f"{map_metric_name}_RBE"}),
                on="run",
                how="inner"
            )
            .merge(
                map_gen_scores_df_dist.rename(columns={map_metric_name: f"{map_metric_name}_RBD"}),
                on="run",
                how="inner"
            )
            .merge(
                execution_time_df,
                on="run",
                how="inner"
            )
        )

        # print(final_df)
        print('Total Runs: ',len(scores_df))


        # Save to a TSV file
        map_file_loc_name = Path(f'GenJudge/results/evaluation/evalmap{self.eval_depth}_{self.ds}_runs_{self.k}.tsv')
        final_df.to_csv(map_file_loc_name, index=False, sep='\t')
        print('File Saved: ',map_file_loc_name)
        
        # Calculate elapsed time
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"Total Execution time: {execution_time:.6f} seconds")
        
        

if __name__ == "__main__":
    PooledEval().main()