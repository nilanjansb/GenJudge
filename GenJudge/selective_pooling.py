import gzip
import os
import argparse
import random
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
from pathlib import Path


class Pooling():

    def __init__(self, **kwargs) -> None:
        # parse the command line
        CLI = argparse.ArgumentParser()

        CLI.add_argument("--runs_loc", type=str)
        CLI.add_argument("--qrels_loc", type=str, default='NA')
        CLI.add_argument("--k", type=int)

        args = CLI.parse_args()

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

            # Skip run type detection for COVID runs
            # and directly classify them as passage runs
            if "covid" in str(path).lower():
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

    def get_ranker_name(self, file_name):
        """
        Identify the ranker/model name from the run filename.

        The expected ranker names are:
        BM25, RM3, DPH, RoBERTa, monoT5, ColBERT,
        ColBERTRR, Contriever, RankZephyr.

        Matching is case-sensitive.
        """

        ranker = file_name.split('.')[0]
        return ranker

        # rankers = [
        #     "BM25",
        #     "RM3",
        #     "DPH",
        #     "RoBERTa",
        #     "monoT5",
        #     "ColBERTRR",
        #     "ColBERT",
        #     "Contriever",
        #     "RankZephyr"
        # ]

        # file_name_lower = file_name.lower()

        # # Check ColBERT-RR before ColBERT
        # # so that it is not incorrectly classified as ColBERT.
        # for ranker in rankers:
        #     if ranker.lower() in file_name_lower:
        #         return ranker

        #return None

    def identify_ranker_runs(self, passage_runs_name, passage_runs_loc):
        """
        Map ranker/model names to their corresponding run files.

        Returns:
            {
                "BM25": path,
                "RM3": path,
                ...
            }
        """
        ranker_runs = {}

        for file_name, file_loc in zip(
                passage_runs_name,
                passage_runs_loc):

            ranker_name = self.get_ranker_name(file_name)

            if ranker_name is not None:
                if ranker_name in ranker_runs:
                    print(
                        f"Warning: multiple runs found for {ranker_name}. "
                        f"Using: {file_name}"
                    )
                else:
                    ranker_runs[ranker_name] = file_loc

        return ranker_runs

    def pool_top_k_per_run(self, runs_to_pool_loc):
        """
        Take top-k results from each run and merge them.

        Results are deduplicated per query using a set.

        Parameters:
            runs_to_pool_loc: list of run file paths

        Returns:
            List of dictionaries with qid and docno.
        """

        pooled = defaultdict(set)

        for file_loc in tqdm(
                runs_to_pool_loc,
                total=len(runs_to_pool_loc)):

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

        return [
            {'qid': qid, 'docno': docno}
            for qid, docs in pooled.items()
            for docno in docs
        ]

    def save_pool(self, runs_to_pool_loc, dataset_name, strategy, num_models):
        """
        Create and save a pool for a particular set of runs.

        Output format:
        GenJudge/results/selective_pooled_runs/
        pooled_<dataset>_runs_<k>_{R/T}{m}.tsv
        """

        pooled_trec_runs = self.pool_top_k_per_run(
            runs_to_pool_loc
        )

        pooled_trec_runs_df = pd.DataFrame(
            pooled_trec_runs,
            columns=["qid", "docno"]
        )

        output_dir = Path(
            "GenJudge/results/selective_pooled_runs"
        )

        # Create output directory if it does not exist
        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        output_file = (
            output_dir /
            f"pooled_{dataset_name}_runs_{self.k}_"
            f"{strategy}{num_models}.tsv"
        )

        pooled_trec_runs_df.to_csv(
            output_file,
            index=False,
            sep='\t'
        )

        print(
            f"Saved: {output_file} "
            f"({len(pooled_trec_runs_df)} rows)"
        )

    def main(self):

        passage_runs_name, passage_runs_loc = (
            self.separate_runs()
        )

        print(f"\nPooling Passage Runs: k = {self.k}")

        # ---------------------------------------------------------
        # Dataset name
        # ---------------------------------------------------------
        ds = os.path.basename(
            os.path.normpath(self.runs_loc)
        )

        # ---------------------------------------------------------
        # Ranker order for temporal pooling
        # ---------------------------------------------------------
        temporal_order = [
            "BM25",
            "RM3",
            "DPH",
            "monoT5",
            "ColBERT",
            "ColBERTRR",
            "Contriever",
            "RetroMAE",
            "RankZephyr"
        ]

        # ---------------------------------------------------------
        # Identify the run file belonging to each ranker
        # ---------------------------------------------------------
        ranker_runs = self.identify_ranker_runs(
            passage_runs_name,
            passage_runs_loc
        )

        print("\nRanker runs found:")

        for ranker in temporal_order:
            if ranker in ranker_runs:
                print(
                    f"  {ranker}: "
                    f"{os.path.basename(ranker_runs[ranker])}"
                )
            else:
                print(
                    f"  {ranker}: NOT FOUND"
                )

        # Only use rankers for which a run was found
        available_rankers = [
            ranker
            for ranker in temporal_order
            if ranker in ranker_runs
        ]

        if len(available_rankers) == 0:
            raise RuntimeError(
                "No recognized ranker runs were found."
            )

        # =========================================================
        # TEMPORAL POOLING
        # =========================================================
        #
        # T1 = BM25
        # T2 = BM25 + RM3
        # T3 = BM25 + RM3 + DPH
        # ...
        #
        # =========================================================

        print("\n" + "=" * 60)
        print("TEMPORAL POOLING")
        print("=" * 60)

        temporal_runs = []

        for m, ranker in enumerate(
                available_rankers,
                start=1):

            temporal_runs.append(
                ranker_runs[ranker]
            )

            print(
                f"\nT{m}: Adding {ranker}"
            )

            print(
                "Models in pool:",
                ", ".join(
                    available_rankers[:m]
                )
            )

            self.save_pool(
                temporal_runs,
                ds,
                "T",
                m
            )

        # =========================================================
        # RANDOM POOLING
        # =========================================================
        #
        # Randomly shuffle the models once.
        #
        # R1 = first random model
        # R2 = first two random models
        # ...
        # R9 = all nine models
        #
        # =========================================================

        print("\n" + "=" * 60)
        print("RANDOM POOLING")
        print("=" * 60)

        random_rankers = available_rankers.copy()
        random.shuffle(random_rankers)

        print("\nRandom ranker order:")
        for i, ranker in enumerate(
                random_rankers,
                start=1):
            print(
                f"  {i}. {ranker}"
            )

        random_runs = []

        for m, ranker in enumerate(
                random_rankers,
                start=1):

            random_runs.append(
                ranker_runs[ranker]
            )

            print(
                f"\nR{m}: Adding {ranker}"
            )

            print(
                "Models in pool:",
                ", ".join(
                    random_rankers[:m]
                )
            )

            self.save_pool(
                random_runs,
                ds,
                "R",
                m
            )

        print("\n" + "=" * 60)
        print("POOLING COMPLETE")
        print("=" * 60)

        print(
            f"\nGenerated "
            f"{len(available_rankers) * 2} pooled files."
        )


if __name__ == "__main__":
    Pooling().main()