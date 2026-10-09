import gzip
import os
import argparse
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
from pathlib import Path


class Pooling():

    def __init__(self, **kwargs) -> None:
        # parse the command line 
        CLI=argparse.ArgumentParser()

        CLI.add_argument("--runs_loc", type=str)
        CLI.add_argument("--qrels_loc", type=str, default='NA')
        CLI.add_argument("--k", type=int)

        args, unknown = CLI.parse_known_args()

        self.runs_loc = args.runs_loc
        self.qrels_loc = args.qrels_loc
        self.k = args.k

    def detect_run_type(self, file_path, sample_size=50):
        """
        Detect whether a run file is 'passage' or 'document'
        based on docno format.
        """
        with gzip.open(file_path, 'rt') as f:
            for i, line in enumerate(f):
                if i >= sample_size:
                    break

                parts = line.strip().split()
                if len(parts) < 3:
                    continue

                docno = parts[2]

                if docno.startswith("D"):
                    return "document"
                elif docno.isdigit():
                    return "passage"

        return "unknown"
    
    def separate_runs(self):
        passage_runs_name = []
        document_runs_name = []
        unknown_runs_name = []

        passage_runs_loc = []
        document_runs_loc = []
        unknown_runs_loc = []

        for file_name in os.listdir(self.runs_loc):
            if not file_name.endswith(".gz"):
                continue

            path = os.path.join(self.runs_loc, file_name)

            # Skip run type detection for COVID runs and directly classify them as passage runs
            if "covid" in str(path):
                run_type = "passage"
            else:
                run_type = self.detect_run_type(path)

            if run_type == "passage":
                passage_runs_name.append(file_name)
                passage_runs_loc.append(path)
            elif run_type == "document":
                document_runs_name.append(file_name)
                document_runs_loc.append(path)
            else:
                unknown_runs_name.append(file_name)
                unknown_runs_loc.append(path)
            
        print("Passage runs:", len(passage_runs_name))
        print("Document runs:", len(document_runs_name))
        print("Unknown runs:", len(unknown_runs_name))

        return passage_runs_name, passage_runs_loc
    
    # take top-k from each run, then merge and deduplicate (pool top-k per run, then union)
    def pool_top_k_per_run(self, runs_to_pool_loc):

        pooled = defaultdict(set)

        for file_loc in tqdm(runs_to_pool_loc, total=len(runs_to_pool_loc)):
            if not file_loc.endswith(".gz"):
                continue

            with gzip.open(file_loc, 'rt') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 6:
                        continue

                    qid, _, docno, rank, _, _ = parts
                    if int(rank) <= self.k:
                        pooled[qid].add(docno)

        return [{'qid': qid, 'docno': docno}
                for qid, docs in pooled.items()
                for docno in docs]
    

    def main(self):
        passage_runs_name, passage_runs_loc = self.separate_runs()
        
        print(f'Pooling Passage Runs: k = {self.k}')
        
        pooled_trec_runs = self.pool_top_k_per_run(passage_runs_loc)
        pooled_trec_runs_df = pd.DataFrame(pooled_trec_runs)

        # Save to a CSV file
        ds = self.runs_loc.split('\\')[-1]
        print(Path.cwd())
        file_loc_name = Path(f'GenJudge/results/pooled_runs/pooled_{ds}_runs_{self.k}.tsv')
        pooled_trec_runs_df.to_csv(file_loc_name, index=False, sep='\t')
        print('File Saved: ',file_loc_name)
    
if __name__ == "__main__":
    Pooling().main()