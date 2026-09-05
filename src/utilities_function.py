
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


def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_configuration(path: str) -> dict:
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


def format_few_shot_prompt(prompt_path, dataset_path,
    test_size: float = 0.1, seed: int = 42):
    prompt_template = load_prompt(prompt_path)

    dataset = load_and_clean_dataset(dataset_path)

    hf_dataset = Dataset.from_pandas(
        dataset,
        preserve_index=False,)

    split = hf_dataset.train_test_split(
        test_size=test_size,
        seed=seed)

    train_set = split["train"]

    weak_examples = train_set.filter(
        lambda x: x["candidate_quality"] == "weak"
    )

    average_examples = train_set.filter(
        lambda x: x["candidate_quality"] == "average"
    )

    strong_examples = train_set.filter(
        lambda x: x["candidate_quality"] == "strong"
    )

    example1 = weak_examples[0]
    example2 = strong_examples[0]
    example3 = average_examples[0]

    prompt_template = prompt_template.format(
        Role1=example1['role'],
        Interview_question1=example1['question'],
        Candidate_answer1=example1['candidate_answer'],
        Feedback1=example1['feedback'],
        Follow_up_question1=example1['follow_up_question'],
        
        Role2=example2['role'],
        Interview_question2=example2['question'],
        Candidate_answer2=example2['candidate_answer'],
        Feedback2=example2['feedback'],
        Follow_up_question2=example2['follow_up_question'],

        Role3=example3['role'],
        Interview_question3=example3['question'],
        Candidate_answer3=example3['candidate_answer'],
        Feedback3=example3['feedback'],
        Follow_up_question3=example3['follow_up_question'],

        input_text="{input_text}"
    )

    
    return prompt_template

    

    


