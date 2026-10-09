def ndcg_all_runs_dist(
    self,
    passage_runs_loc,
    bert_f1_scores_dict,
    qrels,
    K,
):
    """
    Calculate nDCG using BERT F1 scores directly as gains.
    Respects original QRELS as priority (gives them full weight).
    Optimized for GPU batch processing with NumPy vectorization.
    
    Args:
        passage_runs_loc: List of run file locations
        bert_f1_scores_dict: Dict mapping (qid, docno) -> BERT F1 score
        qrels: DataFrame with qrel judgments
        K: cutoff for nDCG calculation
        
    Returns:
        all_results: Dict of run_name -> nDCG@K score
        all_results_pq: Dict of run_name -> list of per-query nDCG scores
        execution_time: Dict of run_name -> execution time
    """

    # --------------------------------------------------
    # QREL LOOKUP - Build reference for original QRELS
    # --------------------------------------------------
    qrel_lookup = defaultdict(dict)
    for _, row in qrels.iterrows():
        qrel_lookup[row["qid"]][row["docno"]] = float(row["label"])

    def dcg_vectorized(f1_scores_array):
        """
        Vectorized DCG calculation using NumPy.
        More efficient than Python loops, can leverage GPU via CuPy if needed.
        
        Args:
            f1_scores_array: numpy array of F1 scores
            
        Returns:
            DCG value
        """
        if len(f1_scores_array) == 0:
            return 0.0
        positions = np.arange(len(f1_scores_array))
        discounts = np.log2(positions + 2)  # log2(i+2) for i starting at 0
        return float(np.sum(f1_scores_array / discounts))

    all_results = {}
    all_results_pq = {}
    execution_time = {}
    
    # Diagnostic counters
    total_docs_evaluated = 0
    docs_from_qrels = 0
    docs_from_f1_scores = 0
    docs_without_f1_scores = 0

    for file_loc in tqdm(
        passage_runs_loc,
        total=len(passage_runs_loc),
        desc="Computing nDCG with BERT F1 scores Distribution (QRELS-aware)"
    ):
        start_time_per = time.perf_counter()

        run_df = self.load_runs_to_df(file_loc)

        # Filter to queries in qrels
        effective_run_df = run_df[
            run_df["qid"].isin(qrels["qid"].unique())
        ]

        grouped = effective_run_df.groupby("qid")

        # Batch collect queries and F1 scores for vectorized processing
        query_ids = []
        f1_scores_per_query = []

        for qid, group in grouped:
            ranked_docs = group["docno"].tolist()[:K]
            qrels_for_q = qrel_lookup.get(qid, {})

            # Get gain scores for ranked documents
            # Priority: original QRELS > F1 scores > 0.0
            # Normalize QRELS labels (0-3) to 0-1 range for consistency with F1 scores
            gain_scores = []
            for docno in ranked_docs:
                if docno in qrels_for_q:
                    # Use original QRELS label, normalized to 0-1 range
                    # 0 -> 0.0, 1 -> 0.33, 2 -> 0.67, 3 -> 1.0
                    normalized_label = float(qrels_for_q[docno]) / 3.0
                    gain_scores.append(normalized_label)
                    docs_from_qrels += 1
                else:
                    # Use F1 score
                    f1_score = bert_f1_scores_dict.get((qid, docno), 0.0)
                    gain_scores.append(f1_score)
                    if f1_score > 0.0:
                        docs_from_f1_scores += 1
                    else:
                        docs_without_f1_scores += 1
            
            f1_scores = np.array(gain_scores, dtype=np.float32)
            
            # Diagnostic tracking
            total_docs_evaluated += len(ranked_docs)

            query_ids.append(qid)
            f1_scores_per_query.append(f1_scores)

        # Vectorized nDCG calculation
        ndcgs = []

        for qid, f1_scores in zip(query_ids, f1_scores_per_query):
            # Calculate DCG for actual ranking
            dcg_val = dcg_vectorized(f1_scores) ### Converting to integer for DCG calculation

            # Calculate IDCG (ideal ranking sorted descending)
            ideal_f1_scores = np.sort(f1_scores)[::-1]  # Descending sort
            idcg_val = dcg_vectorized(ideal_f1_scores)

            # Calculate nDCG with numerical stability
            ndcg = (
                dcg_val / idcg_val
                if idcg_val > 1e-10  # Use small epsilon for stability
                else 0.0
            )

            ndcgs.append(float(ndcg))

        # Store results for this run
        file_name = str(file_loc).split("\\")[-1]

        all_results[file_name] = round(
            float(np.mean(ndcgs)) if ndcgs else 0.0,
            4,
        )

        all_results_pq[file_name] = ndcgs

        execution_time[file_name] = (
            time.perf_counter()
            - start_time_per
        )

    # Print diagnostic information
    if total_docs_evaluated > 0:
        qrel_coverage = (docs_from_qrels / total_docs_evaluated) * 100
        f1_coverage = (docs_from_f1_scores / total_docs_evaluated) * 100
        print(f"\n=== nDCG Distribution (QRELS-aware) ===")
        print(f"Total documents evaluated: {total_docs_evaluated}")
        print(f"  From original QRELS: {docs_from_qrels} ({qrel_coverage:.1f}%)")
        print(f"  From F1 scores: {docs_from_f1_scores} ({f1_coverage:.1f}%)")
        print(f"  Without F1 scores (defaulted to 0.0): {docs_without_f1_scores}")
        print(f"Unique (qid, docno) pairs in F1 dict: {len(bert_f1_scores_dict)}\n")

    return (
        all_results,
        all_results_pq,
        execution_time,
    )