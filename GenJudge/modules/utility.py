from pathlib import Path
import argparse
import pandas as pd
import pyterrier as pt
from collections import defaultdict
import ir_datasets
from tqdm import tqdm
import json
import time
import os

class Utility():
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

        args, unknown = parser.parse_known_args()

        self.runs_loc = args.runs_loc
        self.qrels_loc = args.qrels_loc
        self.k = args.k
        self.ds = self.runs_loc.replace('\\', '/').rstrip('/').split('/')[-1]
    
    def get_dataset_name(self, ds):
        dataset_name = ""

        # Dataset Name Mapping
        if ds=='TREC28' or ds=='TREC19':
            dataset_name = "msmarco-passage/trec-dl-2019/judged"
        elif ds=='TREC29' or ds=='TREC20':
            dataset_name = "msmarco-passage/trec-dl-2020/judged"
        elif ds=='TRECCOVID':
            dataset_name = "beir/trec-covid"
        elif ds=='TREC30':
            dataset_name = "msmarco-passage/dev/small"

        if dataset_name== "":
            print(f"Cannot find a dataset with name: {ds}")
            
        return dataset_name
        
    def find_variants(self, pooled_runs_dir):
        # ============================================================
        # Find all pooled-run variants dynamically
        #
        # Expected format:
        #
        # pooled_<dataset>_runs_<k>_R1.tsv
        # pooled_<dataset>_runs_<k>_R2.tsv
        # ...
        # pooled_<dataset>_runs_<k>_T1.tsv
        # pooled_<dataset>_runs_<k>_T2.tsv
        # ...
        #
        # No fixed number of models/variants is assumed.
        # ============================================================

        pooled_file_pattern = (
            f"pooled_{self.ds}_runs_{self.k}_*.tsv"
        )

        pooled_files = list(
            pooled_runs_dir.glob(pooled_file_pattern)
        )

        if not pooled_files:
            raise FileNotFoundError(
                f"No pooled run files found for "
                f"dataset={self.ds}, k={self.k} in "
                f"{pooled_runs_dir}"
            )

        # ============================================================
        # Extract variant names from filenames
        #
        # Example:
        #
        # pooled_TREC19_runs_1_R3.tsv -> R3
        # pooled_TREC19_runs_1_T7.tsv -> T7
        # ============================================================

        variants = []

        prefix = (
            f"pooled_{self.ds}_runs_{self.k}_"
        )

        for pooled_file in pooled_files:

            filename = pooled_file.name

            if not filename.startswith(prefix):
                continue

            variant = filename[
                len(prefix):
            ].replace(".tsv", "")

            # Only accept variants of the form R<number>
            # or T<number>
            if (
                len(variant) >= 2
                and variant[0] in ['R', 'T']
                and variant[1:].isdigit()
            ):
                variants.append(
                    (variant, pooled_file)
                )

        if not variants:
            raise FileNotFoundError(
                f"No valid R/T pooled variants found for "
                f"dataset={self.ds}, k={self.k}"
            )

        # ============================================================
        # Sort variants
        #
        # R1, R2, R3, ...
        # T1, T2, T3, ...
        # ============================================================

        variants.sort(
            key=lambda x: (
                x[0][0],
                int(x[0][1:])
            )
        )

        print("\n" + "=" * 70)
        print(
            f"Found {len(variants)} pooled variants "
            f"for dataset={self.ds}, k={self.k}"
        )
        print("=" * 70)

        for variant, pooled_file in variants:
            print(
                f"{variant}: {pooled_file}"
            )

        return variants