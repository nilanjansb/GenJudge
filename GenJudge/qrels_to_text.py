import os
import pandas as pd
import ir_datasets
from pathlib import Path
from tqdm import tqdm


def load_qrels(required_loc):
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

def build_qrel_texts(dataset, qrels):
    """
    Add document text to qrels by looking up all qrel docnos
    from the IR-Datasets corpus.
    """

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

    query_lookup = {}
    qids = []
    for query in tqdm(dataset.queries_iter(), desc="Loading Queries"):
        qid = query.query_id
        query_lookup[qid] = query.text
        qids.append(qid)

    qrels_with_text["query"] = qrels_with_text["qid"].astype(str).map(query_lookup)

    return qrels_with_text

def filter_qrels(qrels, pooled_runs):
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

def main():
    #########  Replace with the actual dataset identifier and QRELS   #########
    ### Setup paths and parameters
    pooled_runs_dir = ""
    output_dir = ""
    depths = range(1, 11)

    
    ds = "TRECCOVID"  
    qrels_path = ""  # Replace with the actual path to your QRELS file


    # Dataset Name Mapping
    if ds=='TREC28' or ds=='TREC19':
        dataset_name = "msmarco-passage/trec-dl-2019/judged"
    elif ds=='TREC29' or ds=='TREC20':
        dataset_name = "msmarco-passage/trec-dl-2020/judged"
    elif ds=='TRECCOVID':
        dataset_name = "beir/trec-covid"
    elif ds=='TREC30':
        dataset_name = "msmarco-passage/dev/small"

    dataset = ir_datasets.load(dataset_name)
    qrels_df = load_qrels(qrels_path)

    for depth in tqdm(depths, total=len(depths), desc="Processing depths"):
        pooled_run_path = os.path.join(pooled_runs_dir, f"pooled_{ds}_runs_{depth}.tsv")
        if not os.path.exists(pooled_run_path):
            print(f"Pooled run file for depth {depth} does not exist.")
            continue
        
        pooled_run_df = pd.read_csv(pooled_run_path, sep='\t') #'qid', 'docno'
        
        # Filter QRELS based on the current depth
        filtered_qrels_df = filter_qrels(qrels_df, pooled_run_df)
        
        # Populate the text for the filtered QRELS
        qrels_with_text_df = build_qrel_texts(dataset, filtered_qrels_df)
        
        # Write the combined data to a TSV file
        output_path = os.path.join(output_dir, f"pooled_qrels_{ds}_{depth}.tsv")
        qrels_with_text_df.to_csv(output_path, sep='\t', index=False, header=False)

if __name__ == "__main__":
    main()

'''
python -X utf8 qrels_to_text.py
'''