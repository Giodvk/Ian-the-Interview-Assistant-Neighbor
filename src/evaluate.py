import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from bert_score import score as bert_score



def load_predictions(path: str) -> pd.DataFrame:
    return pd.read_json(path, lines=True)




def clean_predictions(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "question",
        "candidate_answer",
        "reference_feedback",
        "reference_follow_up",
        "generated_feedback",
        "generated_follow_up",
    ]

    available = [c for c in required if c in df.columns]
    print(available)

    before = len(df)

    df = df.dropna(
        subset=available
    ).reset_index(drop=True)

    print(
        f"Removed {before - len(df)} incomplete rows"
    )

    return df



def compute_rouge_l(
    references: list[str],
    predictions: list[str],
) -> float:

    scorer = rouge_scorer.RougeScorer(
        ["rougeL"],
        use_stemmer=True,
    )

    scores = []

    for reference, prediction in zip(
        references,
        predictions,
    ):
        result = scorer.score(
            reference,
            prediction,
        )

        scores.append(
            result["rougeL"].fmeasure
        )

    return float(np.mean(scores))




def compute_bertscore(
    references: list[str],
    predictions: list[str],
) -> float:

    _, _, f1 = bert_score(
        cands=predictions,
        refs=references,
        lang="en",
        verbose=False,
        device="cpu"
    )

    return float(f1.mean())




def mean_cosine_similarity(
    model,
    texts_a: list[str],
    texts_b: list[str],
) -> float:

    embeddings_a = model.encode(
        texts_a,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    embeddings_b = model.encode(
        texts_b,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    similarities = np.sum(
        embeddings_a * embeddings_b,
        axis=1,
    )

    return float(
        similarities.mean()
    )



def distinct_n(
    texts: list[str],
    n: int,
) -> float:

    all_ngrams = []
    total_ngrams = 0

    for text in texts:

        tokens = text.lower().split()

        ngrams = [
            tuple(tokens[i:i+n])
            for i in range(
                len(tokens) - n + 1
            )
        ]

        all_ngrams.extend(ngrams)
        total_ngrams += len(ngrams)

    if total_ngrams == 0:
        return 0.0

    return (
        len(set(all_ngrams))
        / total_ngrams
    )




def average_words(
    texts: list[str]
) -> float:

    lengths = [
        len(text.split())
        for text in texts
    ]

    return float(
        np.mean(lengths)
    )




def evaluate_condition(
    df: pd.DataFrame,
    embedding_model,
    condition_name: str,
) -> dict:

    ref_feedback = (
        df["reference_feedback"]
        .astype(str)
        .tolist()
    )

    pred_feedback = (
        df["generated_feedback"]
        .astype(str)
        .tolist()
    )

    ref_follow = (
        df["reference_follow_up"]
        .astype(str)
        .tolist()
    )

    pred_follow = (
        df["generated_follow_up"]
        .astype(str)
        .tolist()
    )


    questions = (
        df['question']
        .astype(str)
        .tolist()
    )


    candidate_answers = (
        df["candidate_answer"]
        .astype(str)
        .tolist()
    )

 

    feedback_rouge = compute_rouge_l(
        ref_feedback,
        pred_feedback,
    )

    feedback_bertscore = compute_bertscore(
        ref_feedback,
        pred_feedback,
    )
    feedback_semantic = mean_cosine_similarity(
        embedding_model,
        ref_feedback,
        pred_feedback,
    )


    follow_rouge = compute_rouge_l(
        ref_follow,
        pred_follow,
    )

    follow_bertscore = compute_bertscore(
        ref_follow,
        pred_follow,
    )

    follow_semantic = mean_cosine_similarity(
        embedding_model,
        ref_follow,
        pred_follow,
    )



    follow_grounding = mean_cosine_similarity(
        embedding_model,
        pred_follow,
        candidate_answers,
    )



    follow_question_similarity = (
        mean_cosine_similarity(
            embedding_model,
            pred_follow,
            questions,
        )
    )

    non_redundancy = (
        1.0
        - follow_question_similarity
    )


    feedback_distinct_1 = distinct_n(
        pred_feedback,
        1,
    )

    feedback_distinct_2 = distinct_n(
        pred_feedback,
        2,
    )

    follow_distinct_1 = distinct_n(
        pred_follow,
        1,
    )

    follow_distinct_2 = distinct_n(
        pred_follow,
        2,
    )

    return {
        "condition": condition_name,

        "feedback_rougeL":
            feedback_rouge,

        "feedback_bertscore":
            feedback_bertscore,

        "feedback_semantic_similarity":
            feedback_semantic,

        "followup_rougeL":
            follow_rouge,

        "followup_bertscore":
            follow_bertscore,

        "followup_semantic_similarity":
            follow_semantic,

        "followup_grounding":
            follow_grounding,

        "followup_question_similarity":
            follow_question_similarity,

        "followup_non_redundancy":
            non_redundancy,

        "feedback_distinct1":
            feedback_distinct_1,

        "feedback_distinct2":
            feedback_distinct_2,

        "followup_distinct1":
            follow_distinct_1,

        "followup_distinct2":
            follow_distinct_2,

        "feedback_avg_words":
            average_words(
                pred_feedback
            ),

        "followup_avg_words":
            average_words(
                pred_follow
            ),

        "n_samples":
            len(df),
    }


# =========================================================
# Main
# =========================================================

def main():

    files = {
        "zero_shot":
            "./outputs/predictions/zero_shot_predictions.jsonl",

        "few_shot":
            "./outputs/predictions/prompt_predictions.jsonl",

        "qlora":
            "./outputs/predictions/qlora_predictions.jsonl",
    }

    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device = "cpu"
    )

    results = []

    for condition, path in files.items():

        print(
            f"\nEvaluating {condition}..."
        )

        df = load_predictions(path)

        required = [
            "role",
            "question",
            "candidate_answer",
        ]
                
        if df[required].isna().all(axis=1).all():

            df[required] = df['input_text'].str.split('\n\n', expand=True)
        

        result = evaluate_condition(
            df=df,
            embedding_model=embedding_model,
            condition_name=condition,
        )

        results.append(result)

    results_df = pd.DataFrame(
        results
    )

    print("\nFinal results:")
    print(
        results_df
        .set_index("condition")
        .round(4)
        .T
    )

    Path(
        "outputs/evaluation"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        "outputs/evaluation/metrics.csv",
        index=False,
    )


if __name__ == "__main__":
    main()