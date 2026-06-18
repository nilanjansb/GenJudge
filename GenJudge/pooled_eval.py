import gzip
import math
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import pyterrier as pt
from GenJudge.pooling import Pooling
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

        args = CLI.parse_args()

        self.runs_loc = args.runs_loc
        self.qrels_loc = args.qrels_loc
        self.k = args.k
        self.ds = self.runs_loc.split('\\')[-1]

    def load_qrels(self):
        # Load qrels into DataFrame
        qrels = pd.read_csv(
            Path(self.qrels_loc),
            sep=r"\s+",
            names=["qid", "iter", "docno", "label"]
        )
        qrels["qid"] = qrels["qid"].astype(str)
        qrels["docno"] = qrels["docno"].astype(str)

        return qrels
    
    def filter_qrels(self, qrels, pooled_runs):
        filtered_qrels = qrels.merge(
            pooled_runs[["qid", "docno"]],
            on=["qid", "docno"],
            how="inner"
        )
        return filtered_qrels
    
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
    
    def evaluate_ndcg_pyterrier(self, passage_runs_loc, qrels, K):
        """
        Evaluate nDCG@k using PyTerrier
        """
        results = {}
        per_query_results = {}

        for file_loc in tqdm(passage_runs_loc, total=len(passage_runs_loc)):

            run_df = self.load_runs_to_df(file_loc)

            # Evaluate using PyTerrier
            eval_res = pt.Evaluate(
                run_df,
                qrels,
                metrics=[f"ndcg_cut_{K}"]
            )
            file_name = str(file_loc).split('\\')[-1]
            results[file_name] = round(eval_res[f"ndcg_cut_{K}"],4)

            per_query = pt.Evaluate(
                run_df,
                qrels,
                metrics=[f"ndcg_cut_{K}"],
                perquery=True
            )

            per_query_results[file_name] = [
                v[f"ndcg_cut_{K}"]
                for v in per_query.values()
            ]

        return results, per_query_results

    def ndcg_all_runs(self, passage_runs_loc, relevant_docs_dict, qrels, K):

        def bert_to_rel(score):
            # graded relevance from similarity (ONLY for unjudged docs)
            if score >= 0.95:
                return 3
            elif score >= 0.90:
                return 2
            elif score >= 0.80:
                return 1
            else:
                return 0

        def dcg(scores):
            return sum(
                (2**g - 1) / math.log2(i + 2)
                for i, g in enumerate(scores)
            )

        # -----------------------------
        # Build QREL lookup
        # -----------------------------
        qrel_lookup = defaultdict(dict)
        for _, row in qrels.iterrows():
            qrel_lookup[row["qid"]][row["docno"]] = row["label"]

        all_results = {}
        all_results_pq = {}
        execution_time = {}

        for file_loc in tqdm(passage_runs_loc, total=len(passage_runs_loc)):

            start_time_per = time.perf_counter()

            run_df = self.load_runs_to_df(file_loc)
            effective_run_df = run_df[run_df["qid"].isin(qrels["qid"].unique())]

            all_cands = []
            all_refs = []
            meta_qids = []
            meta_docnos = []

            grouped = effective_run_df.groupby("qid")

            # -----------------------------
            # STEP 1: BUILD BATCH FOR BERTSCORE
            # -----------------------------
            for qid, group in grouped:

                ranked_docs = group["docno"].tolist()[:K]

                if qid not in relevant_docs_dict:
                    continue

                relevant_doc = relevant_docs_dict[qid]

                for doc in ranked_docs:
                    all_cands.append(doc)
                    all_refs.append(relevant_doc)
                    meta_qids.append(qid)
                    meta_docnos.append(doc)

            if len(all_cands) == 0:
                file_name = str(file_loc).split('\\')[-1]
                all_results[file_name] = 0.0
                all_results_pq[file_name] = []
                execution_time[file_name] = time.perf_counter() - start_time_per
                continue

            # -----------------------------
            # STEP 2: BERTSCORE (GPU BATCHED)
            # -----------------------------
            _, _, F1 = bscore(
                all_cands,
                all_refs,
                lang="en",
                device="cuda",
                batch_size=256,
                verbose=False
            )

            gains = F1.tolist()

            # (qid, docno) -> similarity
            bert_lookup = {}
            for qid, docno, score in zip(meta_qids, meta_docnos, gains):
                bert_lookup[(qid, docno)] = score

            # -----------------------------
            # STEP 3: BUILD GRADES PER QUERY
            # -----------------------------
            grouped_scores = defaultdict(list)

            for qid, group in grouped:

                ranked_docs = group["docno"].tolist()[:K]
                qrels_for_q = qrel_lookup.get(qid, {})

                rels = []

                for docno in ranked_docs:

                    # PRIORITY 1: HUMAN JUDGMENT
                    if docno in qrels_for_q:
                        rel = qrels_for_q[docno]

                    # PRIORITY 2: GENERATED SEMANTIC RELEVANCE
                    else:
                        sim = bert_lookup.get((qid, docno), 0.0)
                        rel = bert_to_rel(sim)

                    rels.append(rel)

                grouped_scores[qid] = rels

            # -----------------------------
            # STEP 4: COMPUTE NDCG PER QUERY
            # -----------------------------
            ndcgs = []

            for qid, scores in grouped_scores.items():

                dcg_val = dcg(scores)

                # Ideal ranking from QRELS (ground truth dominates)
                ideal_rels = sorted(
                    list(qrel_lookup.get(qid, {}).values()),
                    reverse=True
                )[:len(scores)]

                # pad if qrels smaller than K
                if len(ideal_rels) < len(scores):
                    ideal_rels += [0] * (len(scores) - len(ideal_rels))

                idcg_val = dcg(ideal_rels)

                ndcgs.append(dcg_val / idcg_val if idcg_val > 0 else 0.0)

            # -----------------------------
            # STEP 5: STORE RESULTS
            # -----------------------------
            file_name = str(file_loc).split('\\')[-1]

            all_results[file_name] = round(
                sum(ndcgs) / len(ndcgs) if ndcgs else 0.0,
                4
            )

            all_results_pq[file_name] = ndcgs

            execution_time[file_name] = time.perf_counter() - start_time_per

        return all_results, all_results_pq, execution_time

    def load_relevant_docs_dict(self, model_responses_loc):
        
        # Open the file and load its content
        with open(model_responses_loc, 'r', encoding='utf-8') as file:
            data = json.load(file)

        relevant_docs_dict = {}

        for qid in data.keys():
            relevant_docs_dict[qid]=[data[qid]['generated_doc']]
        
        return relevant_docs_dict

    def compute_kendall_df(self, per_query_results_a, per_query_results_b):
        rows = []

        common_files = set(per_query_results_a) & set(per_query_results_b)

        for file_name in common_files:
            scores_a = per_query_results_a[file_name]
            scores_b = per_query_results_b[file_name]

            if len(scores_a) != len(scores_b):
                raise ValueError(
                    f"Length mismatch for {file_name}: "
                    f"{len(scores_a)} vs {len(scores_b)}"
                )

            tau, p_value = kendalltau(scores_a, scores_b)

            rows.append({
                "run": file_name,
                "kendalls_tau": tau,
                "p_value": p_value,
                "num_queries": len(scores_a)
            })

        return pd.DataFrame(rows)
    
    def save_kendall_taus(
        self,
        summary_file,
        evaluation,
        original_filtered_tau,
        original_generated_tau,
        filtered_generated_tau,
        effective_qrels
        ):
        """
        Add or update Kendall tau values for an evaluation.
        """

        new_row = {
            "Evaluation": evaluation,
            "Original_Filtered": original_filtered_tau,
            "Original_Generated": original_generated_tau,
            "Filtered_Generated": filtered_generated_tau,
            "Effective_QRELS": effective_qrels,
        }

        if os.path.exists(summary_file):
            df = pd.read_csv(summary_file, sep='\t')

            mask = df["Evaluation"] == evaluation

            if mask.any():
                # Update existing row
                df.loc[mask, "Original_Filtered"] = original_filtered_tau
                df.loc[mask, "Original_Generated"] = original_generated_tau
                df.loc[mask, "Filtered_Generated"] = filtered_generated_tau
                df.loc[mask, "Effective_QRELS"] = effective_qrels
            else:
                # Add new row
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            # Create new file
            df = pd.DataFrame([new_row])

        df.to_csv(summary_file, index=False,  sep='\t')


    def main(self):
        # Record the starting time
        start_time = time.perf_counter()

        ds = self.runs_loc.split('\\')[-1]
        file_loc_name = Path(f'GenJudge/results/pooled_runs/pooled_{ds}_runs_{self.k}.tsv')

        pooled_runs = self.load_pooled_runs(file_loc_name)
        qrels = self.load_qrels()
        filtered_qrels = self.filter_qrels(qrels, pooled_runs)

        print('Total QRELS: ', len(qrels))
        print('Filtered QRELS: ', len(filtered_qrels))

        pooling_obj = Pooling()
        passage_runs_name, test_passage_runs = pooling_obj.separate_runs()

        # test_passage_runs = test_passage_runs[:3] ## testing only
        # scores = self.evaluate_ndcg_pyterrier(test_passage_runs, qrels, 10)

        ## nDCG with official runs and QRELS
        scores_ori, scores_pq_ori = self.evaluate_ndcg_pyterrier(test_passage_runs, qrels, 10)
        scores_df_ori = pd.DataFrame(list(scores_ori.items()), columns=["run", "ndcg@10"])

        ## nDCG with official runs and filtered QRELS
        scores, scores_pq = self.evaluate_ndcg_pyterrier(test_passage_runs, filtered_qrels, 10)
        scores_df = pd.DataFrame(list(scores.items()), columns=["run", "ndcg@10"])

        ## nDCG with official runs and generated judgment
        context_file_loc_name = Path(f'GenJudge/results/context_store/context_{self.ds}_runs_{self.k}.json')
        relevant_docs_dict = self.load_relevant_docs_dict(context_file_loc_name)
        gen_scores, gen_scores_pq, execution_time = self.ndcg_all_runs(test_passage_runs, relevant_docs_dict, filtered_qrels, 10)
        gen_scores_df = pd.DataFrame(list(gen_scores.items()), columns=["run", "ndcg@10"])
        execution_time_df = pd.DataFrame(list(execution_time.items()), columns=["run", "Execution Time"])
        

        ## Calculate Kendall's Tau
        ### Overall Rank Basis
        ranks_original = scores_df_ori['ndcg@10'].rank(ascending=False)
        ranks_filtered = scores_df['ndcg@10'].rank(ascending=False)
        ranks_generated = gen_scores_df['ndcg@10'].rank(ascending=False)

        tau_of, p_of = kendalltau(ranks_original, ranks_filtered)
        tau_og, p_og = kendalltau(ranks_original, ranks_generated)
        tau_fg, p_fg = kendalltau(ranks_filtered, ranks_generated)

        summary_path = Path(f'GenJudge/results/evaluation/Summary.tsv')
        num_runs = len(test_passage_runs)
        eval_name = f"eval_{self.ds}_runs_K{self.k}_R{num_runs}"
        self.save_kendall_taus(summary_path,eval_name, tau_of, tau_og, tau_fg, len(filtered_qrels))

        ### Per-query Basis
        # kendall_df_fil_vs_gen = self.compute_kendall_df(scores_pq, gen_scores_pq)
        # kendall_df_ori_vs_fil = self.compute_kendall_df(scores_pq_ori, scores_pq)
        # kendall_df_ori_vs_gen = self.compute_kendall_df(scores_pq_ori, gen_scores_pq)

        ## Print Summary and Results

        # print(scores_df)
        # print(gen_scores_df)
        # print(kendall_df)

        final_df = (
            scores_df_ori.rename(columns={"ndcg@10": "ndcg@10_original_QRELS"})
            .merge(
                scores_df.rename(columns={"ndcg@10": "ndcg@10_filtered_QRELS"}),
                on="run",
                how="inner"
            )
            .merge(
                gen_scores_df.rename(columns={"ndcg@10": "ndcg@10_generated"}),
                on="run",
                how="inner"
            )
            .merge(
                execution_time_df,
                on="run",
                how="inner"
            )
            # .merge(
            #     kendall_df_ori_vs_fil.rename(columns={"kendalls_tau": "kendalls_tau_ori_vs_fil","p_value": "p_value_ori_vs_fil"}),
            #     on="run",
            #     how="inner"
            # )
            # .merge(
            #     kendall_df_fil_vs_gen[['run','kendalls_tau','p_value']].rename(columns={"kendalls_tau": "kendalls_tau_gen_vs_fil","p_value": "p_value_gen_vs_fil"}),
            #     on="run",
            #     how="inner"
            # )
            # .merge(
            #     kendall_df_ori_vs_gen[['run','kendalls_tau','p_value']].rename(columns={"kendalls_tau": "kendalls_tau_ori_vs_gen","p_value": "p_value_ori_vs_gen"}),
            #     on="run",
            #     how="inner"
            # )
        )

        # print(final_df)
        print('Total Runs: ',len(scores_df))


        # Save to a CSV file
        ds = self.runs_loc.split('\\')[-1]

        file_loc_name = Path(f'GenJudge/results/evaluation/eval_{self.ds}_runs_{self.k}.tsv')
        final_df.to_csv(file_loc_name, index=False, sep='\t')
        print('File Saved: ',file_loc_name)

        
        # Calculate elapsed time
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"Total Execution time: {execution_time:.6f} seconds")
        
        

if __name__ == "__main__":
    PooledEval().main()

'''
python -m GenJudge.pooled_eval --runs_loc "E:\\Nilanjan\\DataStore\\TREC_runs\\TREC28" --qrels_loc "E:\\Nilanjan\\KShot_IREval\\GenJudge\\data\\official_qrels\\2019qrels-pass.txt" --k 10


python -m GenJudge.pooled_eval --runs_loc "E:\\Nilanjan\\DataStore\\TREC_runs\\TREC28" --qrels_loc "E:\\Nilanjan\\KShot_IREval\\GenJudge\\data\\official_qrels\\2019qrels-pass.txt" --k 1
'''