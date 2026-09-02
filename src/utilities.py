import argparse
from pathlib import Path
import pandas as pd
import yaml
from datasets import Dataset




REQUIRED_COLUMNS = [
    "role",
    "question",
    "candidate_answer",
    "feedback",
    "follow_up_question",
]


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_and_clean_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    before = len(df)

    df = (
        df.dropna(subset=REQUIRED_COLUMNS)
        .reset_index(drop=True)
    )

    removed = before - len(df)

    print(f"Dataset rows: {before}")
    print(f"Removed incomplete rows: {removed}")
    print(f"Usable rows: {len(df)}")

    return df

def build_sft_example(row: dict) -> dict:

    user_content = (
        f"Role: {row['role']}\n\n"
        f"Interview question:\n"
        f"{row['question']}\n\n"
        f"Candidate answer:\n"
        f"{row['candidate_answer']}"
    )

    assistant_content = (
        f"Feedback:\n"
        f"{row['feedback']}\n\n"
        f"Follow-up question:\n"
        f"{row['follow_up_question']}"
    )

    return {
        "prompt": [
            {
                "role": "user",
                "content": user_content,
            }
        ],
        "completion": [
            {
                "role": "assistant",
                "content": assistant_content,
            }
        ],
    }

def prepare_dataset(
    df: pd.DataFrame,
    validation_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = 42,
):
    hf_dataset = Dataset.from_pandas(
        df,
        preserve_index=False,
    )

    remove_columns = hf_dataset.column_names

    hf_dataset = hf_dataset.map(
        build_sft_example,
        remove_columns=remove_columns,
    )

    first_split = hf_dataset.train_test_split(
        test_size=test_size,
        seed=seed,
    )

    train_val = first_split["train"]
    test = first_split["test"]


    relative_validation_size = (
        validation_size /
        (1.0 - test_size)
    )

    second_split = train_val.train_test_split(
        test_size=relative_validation_size,
        seed=seed,
    )

    train = second_split["train"]
    validation = second_split["test"]

    print(f"Train:      {len(train)}")
    print(f"Validation: {len(validation)}")
    print(f"Test:       {len(test)}")

    return train, validation, test


