# prompts.py

class Prompts:
    SYSTEM_PROMPT = """
        You are an expert information retrieval and question-answering system.
        """
    
    ZERO_SHOT_PROMPT = """
        Generate a relevant answer for the following search query.

        Query:
        {query}

        Generate a single well-formed answer that fully satisfies the information need expressed by the query.

        Answer:
        """

    FEW_SHOT_PROMPT = """
        Generate a relevant answer for the following search query.

        The accompanying documents are known relevant documents from the TREC qrels for this query. Use them as evidence of what information is considered relevant.

        Query:
        {query}

        Relevant Documents:
        {documents}

        Generate a single well-formed answer that fully satisfies the information need expressed by the query.

        Answer:
        """