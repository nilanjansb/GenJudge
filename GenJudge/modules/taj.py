# predict_qrels.py

import os
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
)

from peft import PeftModel


# ============================================================
# Device
# ============================================================

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ============================================================
# PEFT utilities
# ============================================================

def is_peft_model(model_dir):
    """
    Check whether model_dir contains a PEFT/LoRA adapter.
    """
    return os.path.exists(
        os.path.join(model_dir, "adapter_config.json")
    )


def get_base_model_name(model_dir):
    """
    Read the base model name from adapter_config.json.
    """
    adapter_config_path = os.path.join(
        model_dir,
        "adapter_config.json"
    )

    with open(adapter_config_path, "r") as f:
        config = json.load(f)

    return config["base_model_name_or_path"]


# ============================================================
# Cross Encoder Predictor
# ============================================================

class CrossEncoderPredictor:

    def __init__(
        self,
        model_dir,
        base_model=None,
        max_length=512,
        device=None,
    ):
        self.model_dir = str(model_dir)
        self.max_length = max_length
        self.device = device or get_device()

        print(f"Loading Cross-Encoder")
        print(f"Model directory: {self.model_dir}")
        print(f"Device: {self.device}")

        # ----------------------------------------------------
        # LoRA / PEFT model
        # ----------------------------------------------------

        if is_peft_model(self.model_dir):

            if base_model is None:
                base_model = get_base_model_name(
                    self.model_dir
                )

            print(f"Base model: {base_model}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model
            )

            base = AutoModelForSequenceClassification.from_pretrained(
                base_model,
                num_labels=1,
            )

            self.model = PeftModel.from_pretrained(
                base,
                self.model_dir,
            )

        # ----------------------------------------------------
        # Full Hugging Face model
        # ----------------------------------------------------

        else:

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_dir
            )

            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_dir
            )

        self.model.to(self.device)
        self.model.eval()

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    @torch.no_grad()
    def predict_batch(
        self,
        queries,
        documents,
    ):

        encoding = self.tokenizer(
            queries,
            documents,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        encoding = {
            key: value.to(self.device)
            for key, value in encoding.items()
        }

        outputs = self.model(**encoding)

        # Same scoring as the training code:
        #
        # sigmoid(logits)
        scores = torch.sigmoid(
            outputs.logits.squeeze(-1)
        )

        return scores.cpu().tolist()


# ============================================================
# MonoT5 Predictor
# ============================================================

class MonoT5Predictor:

    def __init__(
        self,
        model_dir,
        base_model=None,
        max_length=512,
        device=None,
        prediction_token_false="▁false",
        prediction_token_true="▁true",
    ):
        self.model_dir = str(model_dir)
        self.max_length = max_length
        self.device = device or get_device()

        print(f"Loading MonoT5")
        print(f"Model directory: {self.model_dir}")
        print(f"Device: {self.device}")

        # ----------------------------------------------------
        # LoRA / PEFT model
        # ----------------------------------------------------

        if is_peft_model(self.model_dir):

            if base_model is None:
                base_model = get_base_model_name(
                    self.model_dir
                )

            print(f"Base model: {base_model}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model
            )

            base = AutoModelForSeq2SeqLM.from_pretrained(
                base_model
            )

            self.model = PeftModel.from_pretrained(
                base,
                self.model_dir,
            )

        # ----------------------------------------------------
        # Full Hugging Face model
        # ----------------------------------------------------

        else:

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_dir
            )

            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_dir
            )

        # ----------------------------------------------------
        # MonoT5 true / false token IDs
        # ----------------------------------------------------

        vocab = self.tokenizer.get_vocab()

        if prediction_token_false not in vocab:
            raise ValueError(
                f"Token '{prediction_token_false}' "
                "not found in tokenizer vocabulary."
            )

        if prediction_token_true not in vocab:
            raise ValueError(
                f"Token '{prediction_token_true}' "
                "not found in tokenizer vocabulary."
            )

        self.token_false_id = vocab[
            prediction_token_false
        ]

        self.token_true_id = vocab[
            prediction_token_true
        ]

        self.model.to(self.device)
        self.model.eval()

    # --------------------------------------------------------
    # Same logits computation as training code
    # --------------------------------------------------------

    @torch.no_grad()
    def logits(
        self,
        input_ids,
        attention_mask,
    ):

        decode_ids = torch.full(
            (
                input_ids.size(0),
                1,
            ),
            self.model.config.decoder_start_token_id,
            dtype=torch.long,
            device=input_ids.device,
        )

        encoder_outputs = self.model.get_encoder()(
            input_ids,
            attention_mask=attention_mask,
        )

        model_inputs = (
            self.model.prepare_inputs_for_generation(
                decode_ids,
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
            )
        )

        outputs = self.model(**model_inputs)

        next_token_logits = outputs[0][:, -1, :]

        return next_token_logits

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    @torch.no_grad()
    def predict_batch(
        self,
        queries,
        documents,
    ):

        # IMPORTANT:
        # The training DataModule tokenizes query and document
        # as a pair, exactly like a cross encoder.
        encoding = self.tokenizer(
            queries,
            documents,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        encoding = {
            key: value.to(self.device)
            for key, value in encoding.items()
        }

        logits = self.logits(
            encoding["input_ids"],
            encoding["attention_mask"],
        )

        # Keep only:
        #
        # false
        # true
        #
        logits = logits[
            :,
            [
                self.token_false_id,
                self.token_true_id,
            ],
        ]

        # Same operation as training code.
        logits = torch.nn.functional.log_softmax(
            logits,
            dim=1,
        )

        # Probability of "true"
        scores = torch.exp(
            logits[:, 1]
        )

        return scores.cpu().tolist()


# ============================================================
# Main Prediction Function
# ============================================================

def predict_qrels(
    model_dir,
    encoder_type,
    input_df,
    output_file=None,
    threshold=0.5,
    batch_size=16,
    max_length=512,
    base_model=None,
):
    """
    Predict relevance labels for query-document pairs.

    Parameters
    ----------
    model_dir : str
        Directory containing the trained model / LoRA adapter.

    encoder_type : str
        Either:
            "cross"
            "mono"

    input_df : pandas.DataFrame
        DataFrame containing the following columns:

            qid
            docno
            query
            text

    output_file : str or None
        Path to output TREC QREL file.

        If None, no QREL file is written.

    threshold : float
        Relevance threshold.

        score >= threshold -> 1
        score < threshold  -> 0

    batch_size : int
        Prediction batch size.

    max_length : int
        Maximum tokenizer sequence length.

    base_model : str or None
        Optional base model name.

        Normally not required for LoRA models because the
        base model is read from adapter_config.json.

    Returns
    -------
    pandas.DataFrame

        Input DataFrame with two additional columns:

            score
            label

    """

    # --------------------------------------------------------
    # Validate encoder
    # --------------------------------------------------------

    if encoder_type not in [
        "cross",
        "mono",
    ]:
        raise ValueError(
            "encoder_type must be either "
            "'cross' or 'mono'."
        )

    # --------------------------------------------------------
    # Validate DataFrame
    # --------------------------------------------------------

    if not isinstance(input_df, pd.DataFrame):
        raise TypeError(
            "input_df must be a pandas DataFrame."
        )

    required_columns = {
        "qid",
        "docno",
        "query",
        "text",
    }

    missing_columns = (
        required_columns
        - set(input_df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Input DataFrame is missing required "
            f"columns: {sorted(missing_columns)}"
        )

    # --------------------------------------------------------
    # Copy input
    # --------------------------------------------------------

    df = input_df.copy()

    if len(df) == 0:
        print("Input DataFrame is empty.")

        df["score"] = pd.Series(
            dtype=float
        )

        df["label"] = pd.Series(
            dtype=int
        )

        return df

    print(
        f"Predicting relevance for "
        f"{len(df):,} query-document pairs."
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device()

    # --------------------------------------------------------
    # Load predictor
    # --------------------------------------------------------

    if encoder_type == "cross":

        predictor = CrossEncoderPredictor(
            model_dir=model_dir,
            base_model=base_model,
            max_length=max_length,
            device=device,
        )

    else:

        predictor = MonoT5Predictor(
            model_dir=model_dir,
            base_model=base_model,
            max_length=max_length,
            device=device,
        )

    # --------------------------------------------------------
    # Predict in batches
    # --------------------------------------------------------

    all_scores = []

    for start in tqdm(
        range(
            0,
            len(df),
            batch_size,
        ),
        desc="Predicting",
    ):

        end = min(
            start + batch_size,
            len(df),
        )

        batch = df.iloc[start:end]

        queries = (
            batch["query"]
            .astype(str)
            .tolist()
        )

        documents = (
            batch["text"]
            .astype(str)
            .tolist()
        )

        scores = predictor.predict_batch(
            queries,
            documents,
        )

        all_scores.extend(scores)

    # --------------------------------------------------------
    # Add scores
    # --------------------------------------------------------

    df["score"] = all_scores

    # --------------------------------------------------------
    # Convert score to binary relevance
    # --------------------------------------------------------

    df["label"] = (
        df["score"] >= threshold
    ).astype(int)

    # --------------------------------------------------------
    # Save standard TREC QREL
    #
    # Format:
    #
    # qid iteration docno relevance
    #
    # Example:
    #
    # 1 0 DOC001 1
    # 1 0 DOC002 0
    # --------------------------------------------------------

    if output_file is not None:

        output_path = Path(
            output_file
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as f:

            for _, row in df.iterrows():

                f.write(
                    f"{row['qid']} "
                    f"0 "
                    f"{row['docno']} "
                    f"{row['label']}\n"
                )

        print(
            f"QREL file saved to: "
            f"{output_path}"
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    relevant = int(
        df["label"].sum()
    )

    non_relevant = (
        len(df) - relevant
    )

    print()
    print("Prediction complete.")
    print(f"Total       : {len(df):,}")
    print(f"Relevant    : {relevant:,}")
    print(f"Non-relevant: {non_relevant:,}")
    print(f"Threshold   : {threshold}")

    # --------------------------------------------------------
    # Return predictions
    # --------------------------------------------------------

    return df


# ============================================================
# Optional CLI
# ============================================================

def main():

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Predict relevance labels using a "
            "trained Cross-Encoder or MonoT5 model."
        )
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--encoder_type",
        type=str,
        required=True,
        choices=[
            "cross",
            "mono",
        ],
    )

    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help=(
            "TSV containing columns: "
            "qid, docno, query, text"
        ),
    )

    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Read input only for CLI usage
    # --------------------------------------------------------

    df = pd.read_csv(
        args.input_file,
        sep="\t",
    )

    predict_qrels(
        model_dir=args.model_dir,
        encoder_type=args.encoder_type,
        input_df=df,
        output_file=args.output_file,
        threshold=args.threshold,
        batch_size=args.batch_size,
        max_length=args.max_length,
        base_model=args.base_model,
    )


if __name__ == "__main__":
    main()