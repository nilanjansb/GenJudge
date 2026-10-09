#!/usr/bin/env python3
"""
Command‑line driver for generating LLM based qrels.
"""

from GenJudge.modules.llm_judge import LLMJudge
from GenJudge.pooled_eval import PooledEval
import argparse
import pandas as pd
import pyterrier as pt
import ir_datasets
from pathlib import Path
import os
from tqdm import tqdm

class GenJudgment:
    """
    Driver that creates text_res files from pooled runs and then uses the
    LLMJudge to produce final qrels. Supports optional few‑shot examples.
    """

    def __init__(self, **kwargs) -> None:
        # Parse command line
        parser = argparse.ArgumentParser()
        parser.add_argument("--runs_loc", type=str, required=True)
        parser.add_argument("--qrels_loc", type=str, required=True)
        parser.add_argument("--k", type=int, required=True)
        parser.add_argument(
            "--fewshot_count",
            type=int,
            default=1,
            help="Number of few‑shot examples per query.",
        )

        args = parser.parse_args()

        self.runs_loc = args.runs_loc
        self.qrels_loc = args.qrels_loc
        self.k = args.k
        self.fewshot_count = args.fewshot_count

        self.ds = self.runs_loc.replace('\\', '/').rstrip('/').split('/')[-1]
        self.model_name = "Llama-3.1-8B-Instruct"

        self.hf_model_name_map = {
            "Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct"
        }

    def _load_fewshot_examples(self):
        """
        Load few‑shot examples from the specified QRELS file.
        Returns a dict mapping qid to {'query':..., 'text':..., 'label':...}.
        Samples self.fewshot_count rows per query (first rows).
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

        # Load IR_Datasets 
        dataset = ir_datasets.load(dataset_name)
        
        # Get QRELS from Pyterrier Datasets
        pyt_ds = pt.datasets.get_dataset('irds:'+dataset_name)
        # df = pyt_ds.get_qrels()
        
        # if not self.qrels_loc:
        #     return {}

        # Load original qrels from file
        df = pd.read_csv(
            Path(self.qrels_loc),
            sep=r"\s+",
            names=["qid", "iter", "docno", "label"],
        )

        df = df.loc[df["label"] > 1] # Keep only relevant documents for few-shot examples

        

        query_lookup = {}
        for query in tqdm(dataset.queries_iter(), desc="Loading Queries"):
            qid = query.query_id
            query_lookup[str(qid)] = query.text
        
        doc_lookup = {}
        # print(needed_docnos)
        for doc in tqdm(dataset.docs_iter(), desc="Loading Documents"):
            #if doc.doc_id in needed_docnos:
            doc_lookup[str(doc.doc_id)] = doc.text

        examples = {}

        fs_examples = []

        for qid, group in df.groupby("qid"):
            df.sort_values(["label"], ascending=False, inplace=True) # Sort by label to get the most relevant documents first
            sample = group.head(self.fewshot_count)
            docnos = sample["docno"].tolist()

            for docno in docnos:
                if str(docno) not in doc_lookup:
                    continue
                if str(qid) not in query_lookup:
                    continue

                fs_examples.append({
                    "query": query_lookup[str(qid)], 
                    "text": doc_lookup[str(docno)],
                    "label": str(sample["label"].loc[(sample["qid"] == qid) & (sample["docno"] == docno)].values[0]),
                })

            examples[qid] = fs_examples

        return examples

    def load_text_res(self, text_res_file_loc_name):
        """Load text_res file into a DataFrame."""
        return pd.read_csv(
            Path(text_res_file_loc_name),
            sep="\t",
        )

    def create_text_res_file(self, text_res_file_loc_name):
        """
        Load pooled runs and extract docids of relevant documents from QRELS.
        Save the TSV with columns: qid, docno, query, text.
        """
        pooled_eval_obj = PooledEval()
        file_loc_name = Path(
            f"GenJudge/results/pooled_runs/pooled_{self.ds}_runs_{self.k}.tsv"
        )
        pooled_runs = pooled_eval_obj.load_pooled_runs(file_loc_name)

        qids = []

        # Dataset Name Mapping
        if self.ds=='TREC28' or self.ds=='TREC19':
            dataset_name = "msmarco-passage/trec-dl-2019/judged"
        elif self.ds=='TREC29' or self.ds=='TREC20':
            dataset_name = "msmarco-passage/trec-dl-2020/judged"
        elif self.ds=='TRECCOVID':
            dataset_name = "beir/trec-covid"
        elif self.ds=='TREC30':
            dataset_name = "msmarco-passage/dev/small"

        # Load IR_Datasets
        dataset = ir_datasets.load(dataset_name)

        query_lookup = {}
        for query in tqdm(dataset.queries_iter(), desc="Loading Queries"):
            qid = query.query_id
            query_lookup[qid] = query.text
            qids.append(qid)

        doc_lookup = {}
        needed_docnos = set(pooled_runs["docno"].unique())
        for doc in tqdm(dataset.docs_iter(), desc="Loading Documents"):
            if doc.doc_id in needed_docnos:
                doc_lookup[doc.doc_id] = doc.text

            if len(doc_lookup) == len(needed_docnos):
                break

        pooled_runs["query"] = pooled_runs["qid"].astype(str).map(query_lookup)
        pooled_runs["text"] = pooled_runs["docno"].map(doc_lookup)

        print("Total Instances Saved:", len(pooled_runs))
        print("Total Queries Saved:", len(pooled_runs["qid"].unique()))

        # Save the TSV file
        pooled_runs.to_csv(text_res_file_loc_name, index=False, sep="\t")
        print("File Saved:", text_res_file_loc_name)

    def generate_and_save_qrels(
        self,
        gen_qrels_loc_name,
        gen_qrels_file_name,
        text_res,
        fewshot_examples,
    ):
        """
        Generate qrels using LLMJudge and write to disk.
        """
        print(">> gen_qrels File Not Found (Generating gen_qrels File.)")
        judge = LLMJudge(
            model_name=self.hf_model_name_map[self.model_name],
            short_model_name=self.model_name,
            trec_run=self.ds,
            num_fewshot_examples=self.fewshot_count,
            fewshot_examples=fewshot_examples,
        )
        gen_qrels = judge.judge_csv(text_res)
        gen_qrels.to_csv(gen_qrels_loc_name, index=False, sep="\t",header=False)
        print("File Saved:", gen_qrels_file_name)

    def start_qrels_generation(self,text_res,fewshot_examples_raw,gen_qrels_loc_name,gen_qrels_file_name):
        sub_text_res = text_res.loc[text_res["qid"].isin(fewshot_examples_raw.keys())]
        sub_text_res = sub_text_res
        # print(fewshot_examples_raw)
        print("Fewshot Examples: ",len(fewshot_examples_raw))
        self.generate_and_save_qrels(
            gen_qrels_loc_name,
            gen_qrels_file_name,
            sub_text_res,
            fewshot_examples_raw,
        )

    def main(self):
        text_res_file_loc_name = Path(
            f"GenJudge/results/context_store/text_res/text_res_{self.ds}_runs_{self.k}.tsv"
        )
        text_res_file_name = (
            f"text_res_{self.ds}_runs_{self.k}.tsv"
        )

        gen_qrels_loc_name = Path(
            f"GenJudge/results/gen_qrels/gen_qrels_{self.ds}_runs_{self.k}_few{self.fewshot_count}.qrels"
        )
        gen_qrels_file_name = (
            f"gen_qrels_{self.ds}_runs_{self.k}_few{self.fewshot_count}.qrels"
        )

        # Prepare few‑shot examples
        fewshot_examples_raw = self._load_fewshot_examples()

        if text_res_file_loc_name.exists():
            print(">> text_res File Found:", text_res_file_name)
            if gen_qrels_loc_name.exists():
                print(
                    ">> gen_qrels File Found:",
                    gen_qrels_file_name,
                )
            else:
                text_res = self.load_text_res(text_res_file_loc_name)
                self.start_qrels_generation(text_res, fewshot_examples_raw, gen_qrels_loc_name, gen_qrels_file_name)
        else:
            print(">> Text RES File Not Found (Creating Text RES File.)")
            self.create_text_res_file(text_res_file_loc_name)
            text_res = self.load_text_res(text_res_file_loc_name)
            self.start_qrels_generation(text_res, fewshot_examples_raw, gen_qrels_loc_name, gen_qrels_file_name)

if __name__ == "__main__":
    GenJudgment().main()