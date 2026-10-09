from GenJudge.modules.llm_judge import LLMJudge
from GenJudge.pooled_eval import PooledEval
from GenJudge.modules.utility import Utility
from pathlib import Path
import argparse
import pandas as pd
import pyterrier as pt
from collections import defaultdict
import ir_datasets
from tqdm import tqdm
import transformers
import torch
from transformers import AutoTokenizer
import json
import time
import os
import yaml

class GenJudgment:

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
        self.ds = self.runs_loc.replace('\\', '/').rstrip('/').split('/')[-1]

        # Load configuration file
        config_path = Path('GenJudge/config.yaml')
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        ### Select Model ###
        self.model_name = self.config['llm']['model_name']
        
        ### Environment Variables ###
        os.environ["HF_TOKEN"] = self.config['env']['HF_TOKEN']
    
    def format_relevant_documents(self, documents: list[str]) -> str:
        return "\n\n".join(
            f"[DOC{i + 1}]\n{doc.strip()}"
            for i, doc in enumerate(documents)
        )
    
    
    def extract_relevant_document(self, filtered_qrels, rel_label):
        filtered_rel_docs_list = filtered_qrels.loc[filtered_qrels['label']==rel_label]
        return filtered_rel_docs_list

    def load_text_res(self, text_res_file_loc_name):
        # Load qrels into DataFrame
        text_res = pd.read_csv(
            Path(text_res_file_loc_name),
            sep="\t"
        )
        return text_res
    

    def load_text_res(self, text_res_file_loc_name):
        # Load qrels into DataFrame
        text_res = pd.read_csv(
            Path(text_res_file_loc_name),
            sep="\t"
        )
        return text_res
    
    def create_text_res_file(self, text_res_file_loc_name, dataset_name, file_loc_name):
        # Load pooled runs amd extract docids of relevant documents from QRELS
        pooled_eval_obj = PooledEval()
        
        pooled_runs = pooled_eval_obj.load_pooled_runs(file_loc_name)        

        qids = []

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

        print('Total Instances Saved: ',len(pooled_runs))
        print('Total Queries Saved: ',len(pooled_runs['qid'].unique()))

        # Save the TSV file
        pooled_runs.to_csv(text_res_file_loc_name, index=False, sep='\t')
        print('File Saved: ',text_res_file_loc_name)
    
    def generate_and_save_qrels(self,gen_qrels_loc_name,gen_qrels_file_name,text_res):
        print(">> gen_qrels File Not Found (Generating gen_qrels File.)")
        judge = LLMJudge(
            model_name=self.config['llm']['model_path'],
            short_model_name = self.model_name, 
            num_fewshot_examples=0, 
            trec_run=self.ds,
            device=self.config['llm']['device'],
            hf_token=self.config['env']['HF_TOKEN']
        )
        gen_qrels=judge.judge_csv(text_res)
        gen_qrels.to_csv(gen_qrels_loc_name,index=False, header=False, sep = '\t')
        print('File Saved: ', gen_qrels_file_name)

    def main(self):

        pooled_runs_dir = Path(
            "GenJudge/results/selective_pooled_runs"
        )

        utility_obj = Utility()
        variants = utility_obj.find_variants(pooled_runs_dir)
        dataset_name = utility_obj.get_dataset_name(self.ds)

        for variant, pooled_file in variants:

            print("\n" + "=" * 70)
            print(
                f"Processing Pool: {variant} "
                f"| Dataset: {self.ds} "
                f"| k: {self.k}"
            )
            print("=" * 70)

            # ========================================================
            # Load pooled run
            # ========================================================

            print(
                f"Loading pooled run: {pooled_file}\n"
            )
            print(
                f"Variant: {variant}\n"
            )
        
            text_res_file_loc_name = Path(f'GenJudge/results/selective_context_store/text_res/text_res_{self.ds}_runs_{self.k}_{variant}.tsv')
            text_res_file_name = f'text_res_{self.ds}_runs_{self.k}_{variant}.tsv'

            gen_qrels_loc_name = Path(f'GenJudge/results/selective_gen_qrels/gen_qrels_{self.ds}_runs_{self.k}_{variant}.qrels')
            gen_qrels_file_name = f'gen_qrels_{self.ds}_runs_{self.k}_{variant}.qrels'

            # Result files with Query and Docs Text without associated original QRELS
            if text_res_file_loc_name.exists():
                print(">> text_res File Found:", text_res_file_name)
                if gen_qrels_loc_name.exists():
                    print(">> gen_qrels File Found:", gen_qrels_file_name)
                else:
                    text_res = self.load_text_res(text_res_file_loc_name)
                    self.generate_and_save_qrels(gen_qrels_loc_name,gen_qrels_file_name,text_res)
            else:
                print(">> Text RES File Not Found (Creating Text RES File.)")
                self.create_text_res_file(text_res_file_loc_name, dataset_name, pooled_file)
                text_res = self.load_text_res(text_res_file_loc_name)
                self.generate_and_save_qrels(gen_qrels_loc_name,gen_qrels_file_name,text_res)
            

if __name__ == "__main__":
    GenJudgment().main()