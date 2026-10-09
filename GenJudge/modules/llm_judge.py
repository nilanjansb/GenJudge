from __future__ import annotations
from pathlib import Path
import re
import json
import pandas as pd
import torch
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import yaml

class LLMJudge:
    """
    LLM-based relevance judge that converts PyTerrier retrieval results
    into TREC QRELS.

    Expected dataframe columns:
        query   : query text
        text    : document text

    Optional columns:
        qid
        docno

    Output columns:
        qid
        iteration
        docno
        relevance
    """

    def __init__(
        self,
        model_name,
        short_model_name,
        trec_run,
        device,
        hf_token: str,
        max_new_tokens: int = 8,
        batch_size: int = 4,
        cache_file_loc: str="GenJudge/results/gen_qrels/cache/",
        num_fewshot_examples:int=0,
        fewshot_examples:dict=None
    ):
        ### Environment Variables ###
        os.environ["HF_TOKEN"] = hf_token
        
        self.device = device

        # Store few-shot parameters
        self.num_fewshot_examples = num_fewshot_examples
        self.fewshot_examples = fewshot_examples if fewshot_examples is not None else {}

        catch_file_loc_name = (
            cache_file_loc + f'{short_model_name}_{trec_run}'
            f'_few{self.num_fewshot_examples}_judge_cache.json'
        )
        self.cache_file = Path(catch_file_loc_name)

        if self.cache_file.exists():
            print('Catch: ',self.cache_file)
            with open(self.cache_file, "r") as f:
                self.cache = json.load(f)
        else:
            self.cache = {}

        print("Loaded cache entries:", len(self.cache))

        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_name,device_map=self.device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )

        self.model.eval()

    ##########################################################################
    # Prompt
    ##########################################################################

    def build_prompt(self, query: str, document: str) -> str:
        return f"""You are an expert information retrieval assessor.

        Evaluate how relevant the document is to the query.

        Relevance scale:

        0 = Not relevant
        1 = Slightly relevant
        2 = Relevant
        3 = Highly relevant

        Return ONLY the integer (0,1,2,3).

        Query:
        {query}

        Document:
        {document}

        Relevance:
        """

    def build_fs_prompt(self, query: str, document: str, examples: dict) -> str:

        lines = []

        header = """You are an expert information retrieval assessor.

        Evaluate how relevant the document is to the query. Use provided few‑shot examples as guidance.

        Relevance scale:

        0 = Not relevant
        1 = Slightly relevant
        2 = Relevant
        3 = Highly relevant

        Return ONLY the integer (0,1,2,3).


        """
        lines.append(header)
        count = 1
        for ex in examples:
            if count <= self.num_fewshot_examples:
                lines.append(f"Example {count}:")
                lines.append(
                    f"Query: \n{ex['query']}\n\n"
                    f"Document: \n{ex['text']}\n\n"
                    f"Relevance: \n{ex['label']}"
                )
                count += 1

        lines.append("\n---\n")
        lines.append(f"Query: \n{query}\n\nDocument:\n{document}\n\nRelevance:\n")
        return "\n".join(lines)

    ##########################################################################
    # Parsing
    ##########################################################################

    @staticmethod
    def parse_score(text: str) -> int:
        """
        Robust parser for LLM output.
        """

        text = text.strip()

        m = re.search(r"\b([0-3])\b", text)

        if m:
            return int(m.group(1))

        text_lower = text.lower()

        if "high" in text_lower:
            return 3

        if "relevant" in text_lower:
            return 2

        if "slight" in text_lower:
            return 1

        return 0

    ##########################################################################
    # Batch inference
    ##########################################################################

    def judge_batch(self, prompts):
        enc = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**enc)

        # Logits for the next token after the prompt
        next_token_logits = outputs.logits[:, -1, :]   # (batch_size, vocab_size)

        # Token ids for "0", "1", "2", "3"
        label_ids = [
            self.tokenizer.encode(str(i), add_special_tokens=False)[0]
            for i in range(4)
        ]

        # Extract logits corresponding to those labels
        label_logits = next_token_logits[:, label_ids]   # (batch_size, 4)

        # Highest logit wins
        preds = torch.argmax(label_logits, dim=-1)

        return preds.cpu().tolist()

    ##########################################################################
    # DataFrame judging
    ##########################################################################

    def judge_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:

        if "query" not in df.columns:
            raise ValueError("Missing 'query' column.")

        if "text" not in df.columns:
            raise ValueError("Missing 'text' column.")

        if "qid" not in df.columns:
            raise ValueError("Missing 'qid' column.")
            # df = df.copy()
            # df["qid"] = range(len(df))

        if "docno" not in df.columns:
            raise ValueError("Missing 'docno' column.")
            # df = df.copy()
            # df["docno"] = [f"D{i}" for i in range(len(df))]

        scores = [None] * len(df)
        df = df.reset_index(drop=True)

        to_judge = []
        to_judge_idx = []

        fs_counter = 0

        for idx, row in df.iterrows():
            cached = self.get_cached(row["qid"], row["docno"])
            if cached is not None:
                scores[idx] = cached
            else:
                if self.fewshot_examples and len(self.fewshot_examples[row["qid"]]) > 0:
                    to_judge.append(
                        self.build_fs_prompt(row["query"], row["text"], self.fewshot_examples[row["qid"]])
                    )
                    fs_counter += 1
                else:
                    to_judge.append(
                        self.build_prompt(row["query"], row["text"])
                    )
                to_judge_idx.append(idx)

        
        
        print("==========================================")
        print(f"Few-shot Judgments: {fs_counter}") 
        print(f"Zero-shot Judgments: {len(to_judge) - fs_counter}") 
        print("==========================================")

        # Run inference only on missing items:
        new_scores = []
        for i in tqdm(range(0, len(to_judge), self.batch_size)):
            batch = to_judge[i:i+self.batch_size]
            new_scores.extend(self.judge_batch(batch))

        # Update the cache
        for idx, rel in zip(to_judge_idx, new_scores):
            scores[idx] = rel
            self.save_cached(
                df.loc[idx, "qid"],
                df.loc[idx, "docno"],
                rel
            )

        self.flush_cache()

        qrels = pd.DataFrame({
            "qid": df["qid"],
            "iteration": 0,
            "docno": df["docno"],
            "relevance": scores,
        })

        return qrels

    ## Cache Accelerate Functions
    def _cache_key(self, qid, docno):
        return f"{qid}::{docno}"

    def get_cached(self, qid, docno):
        return self.cache.get(self._cache_key(qid, docno))

    def save_cached(self, qid, docno, rel):
        self.cache[self._cache_key(qid, docno)] = int(rel)

    def flush_cache(self):
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f, indent=2)

    ##########################################################################
    # CSV interface
    ##########################################################################

    def judge_csv(
        self,
        input_csv: str
    ):
        qrels = self.judge_dataframe(input_csv)

        return qrels

    ##########################################################################
    # TREC interface
    ##########################################################################

    def save_trec_qrels(
        self,
        qrels: pd.DataFrame,
        output_path: str,
    ):
        qrels.to_csv(
            output_path,
            sep="\t",
            index=False,
            header=False,
        )