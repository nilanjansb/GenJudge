from GenJudge.modules.prompts import Prompts
from GenJudge.pooled_eval import PooledEval
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


class GenContext():

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
        self.model_name = 'Llama-3.1-8B-Instruct'

        ### Set Relevance Label ###
        # self.rel_label = 3 # set-relevance label as 3 by default
        self.rel_label = 1 # set-relevance label as 1 for dev.small

        hf_model_name_map = {
            'Llama-3.1-8B-Instruct':"meta-llama/Llama-3.1-8B-Instruct"
            }
        
        # self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name_map[self.model_name])
        self.pipeline = transformers.pipeline("text-generation", model=hf_model_name_map[self.model_name], model_kwargs={"torch_dtype": torch.bfloat16}, device_map="cuda")
    
    def format_relevant_documents(self, documents: list[str]) -> str:
        return "\n\n".join(
            f"[DOC{i + 1}]\n{doc.strip()}"
            for i, doc in enumerate(documents)
        )
    
    
    def extract_relevant_document(self, filtered_qrels, rel_label):
        filtered_rel_docs_list = filtered_qrels.loc[filtered_qrels['label']==rel_label]
        return filtered_rel_docs_list

    def load_pre_context(self, pre_context_file_loc_name):
        # Load qrels into DataFrame
        pre_context = pd.read_csv(
            Path(pre_context_file_loc_name),
            sep="\t"
        )
        return pre_context
    
    def create_pre_context_file(self, pre_context_file_loc_name):
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
        qrels = pyt_ds.get_qrels()

        # Load pooled runs amd extract docids of relevant documents from QRELS
        pooled_eval_obj = PooledEval()
        file_loc_name = Path(f'GenJudge/results/pooled_runs/pooled_{self.ds}_runs_{self.k}.tsv')
        pooled_runs = pooled_eval_obj.load_pooled_runs(file_loc_name)
        # qrels = pooled_eval_obj.load_qrels(self.qrels_loc)
        filtered_qrels = pooled_eval_obj.filter_qrels(qrels, pooled_runs)
        relevant_filtered_qrels = self.extract_relevant_document(filtered_qrels, self.rel_label)
        
        qids = []

        query_lookup = {}
        for query in tqdm(dataset.queries_iter(), desc="Loading Queries"):
            qid = query.query_id
            query_lookup[qid] = query.text
            qids.append(qid)
        
        doc_lookup = {}
        needed_docnos = set(relevant_filtered_qrels["docno"].unique())
        for doc in tqdm(dataset.docs_iter(), desc="Loading Documents"):
            if doc.doc_id in needed_docnos:
                doc_lookup[doc.doc_id] = doc.text

            if len(doc_lookup) == len(needed_docnos):
                break

        relevant_filtered_qrels["query"] = relevant_filtered_qrels["qid"].astype(str).map(query_lookup)
        relevant_filtered_qrels["text"] = relevant_filtered_qrels["docno"].map(doc_lookup)

        # Add missing QIDs
        missing_qids = list(set(qids) - set(relevant_filtered_qrels['qid'].tolist()))

        # print('Total QIDs', len(qids))
        # print('Missing QIDs',missing_qids)

        new_rows = []
        for qid in missing_qids:
            new_rows.append({
                "qid": qid, 
                "iter":'Q0',
                "docno":None,
                "label":None,
                "query": query_lookup[qid],
                "text":None
            })
        
        if new_rows:
            relevant_filtered_qrels = pd.concat([relevant_filtered_qrels, pd.DataFrame(new_rows)], ignore_index=True)

        print('Total Instances Saved: ',len(relevant_filtered_qrels))
        print('Total Queries Saved: ',len(relevant_filtered_qrels['qid'].unique()))

        # Save the TSV file
        relevant_filtered_qrels.to_csv(pre_context_file_loc_name, index=False, sep='\t')
        print('File Saved: ',pre_context_file_loc_name)
    
    def generate_context_file(self, pre_context_file_loc_name, context_file_loc_name):
        # Define cache directory and file path
        cache_dir = Path("GenJudge/results/context_store/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create cache filename based on model name and dataset
        cache_filename = f"{self.model_name}_{self.ds}_cache.json"
        cache_file_path = cache_dir / cache_filename
        
        # Load existing cache if it exists
        existing_context = {}
        if cache_file_path.exists():
            with open(cache_file_path, "r", encoding="utf-8") as f:
                existing_context = json.load(f)
        
        # Load pre-context
        pre_context = self.load_pre_context(pre_context_file_loc_name)

        # Filter out already processed QIDs from the pre_context DataFrame
        processed_qids = {str(qid) for qid in existing_context.keys()}

        if processed_qids:
            qid_as_string = pre_context['qid'].astype('string')

            pre_context_filtered = pre_context[
                ~qid_as_string.isin(processed_qids)
            ]
        else:
            pre_context_filtered = pre_context

        # Make Zeroshot/Fewshot Prompt
        # pre_context_filtered = pre_context_filtered.head(5)  # Limit to first 5 rows for testing

        grouped = pre_context_filtered.groupby("qid")

        # Initialize context_json with existing data
        context_json = existing_context.copy()

        # Process only unprocessed queries
        total_groups = len(grouped)
        processed_count = len(processed_qids)
        print(f"Already processed: {processed_count}/{total_groups} queries")
        print(f"Remaining to process: {total_groups} queries")
        
        for qid, group in tqdm(grouped, total=total_groups, desc="Generating Context using LLM"):
            # Record the starting time
            start_time = time.perf_counter()
            
            query = group["query"].iloc[0]
            docs = group["text"].tolist()
            docs = [x for x in docs if pd.notna(x)]

            sys_prompt = Prompts.SYSTEM_PROMPT
            icl_docs = ""
            if docs:
                # few-shot
                icl_docs = self.format_relevant_documents(docs)
                user_prompt = Prompts.FEW_SHOT_PROMPT.format(
                    query=query,
                    documents=icl_docs
                )
            else:
                # zero-shot
                user_prompt = Prompts.ZERO_SHOT_PROMPT.format(
                    query=query
                )

            messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            
            # Generate LLM responses
            outputs = self.pipeline(messages)
            
            generated_doc = outputs[0]["generated_text"][-1]["content"]

            # Calculate elapsed time
            end_time = time.perf_counter()
            generation_time = (end_time - start_time)
            generation_time_seconds = f"{generation_time:.6f}"

            ## Store in JSON

            context_json[qid] = {
                    'query_text': query,
                    'icl_examples': icl_docs,
                    'k_shots': len(docs),
                    'generator_model': self.model_name,
                    'generation_prompt':messages,
                    'generated_doc':generated_doc,
                    'generation_time_seconds':generation_time_seconds
                }
            
            # Save cache incrementally every 10 queries to avoid data loss
            if len(context_json) % 10 == 0:
                with open(cache_file_path, "w", encoding="utf-8") as f:
                    json.dump(context_json, f, indent=4, ensure_ascii=False)
                print(f"Cache updated: {len(context_json)} queries processed")

        # Save final JSON file
        with open(context_file_loc_name, "w", encoding="utf-8") as f:
            json.dump(context_json, f, indent=4, ensure_ascii=False)
        
        # Save final cache file
        with open(cache_file_path, "w", encoding="utf-8") as f:
            json.dump(context_json, f, indent=4, ensure_ascii=False)
        
        print('File Saved: ',context_file_loc_name)
        print('Cache File Saved: ',cache_file_path)


    def main(self):
        pre_context_file_loc_name = Path(f'GenJudge/results/context_store/pre_context/pre_context_{self.ds}_runs_{self.k}.tsv')
        pre_context_file_name = f'pre_context_{self.ds}_runs_{self.k}.tsv'

        context_file_loc_name = Path(f'GenJudge/results/context_store/context_{self.ds}_runs_{self.k}.json')
        context_file_name = f'context_{self.ds}_runs_{self.k}.json'

        if pre_context_file_loc_name.exists():
            print(">> Pre-Context File Found:", pre_context_file_name)
            if context_file_loc_name.exists():
                print(">> Context File Found:", context_file_name)
            else:
                print(">> Context File Not Found (Generating Context File.)")
                self.generate_context_file(pre_context_file_loc_name, context_file_loc_name)
        else:
            print(">> Pre-Context File Not Found (Creating Context File.)")
            self.create_pre_context_file(pre_context_file_loc_name)
            print(">> Generating Context File.")
            self.generate_context_file(pre_context_file_loc_name, context_file_loc_name)

if __name__ == "__main__":
    GenContext().main()