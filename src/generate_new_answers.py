import argparse
import json
import random
from pathlib import Path
import time
import pandas as pd
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


REQUIRED_OUTPUT_FIELDS = {
    "candidate_answer",
    "feedback",
    "follow_up_question",
}

QUALITY_LEVELS = ["weak", "average", "strong"]


def sample_dataset(
    df: pd.DataFrame,
    n_samples: int = 10_000,
    random_state: int = 42,
    columns_to_check: list[str] | None = None,
):

    if n_samples > len(df):
        raise ValueError(
            f"n_samples={n_samples} is greater than "
            f"dataset size={len(df)}."
        )

    if columns_to_check is None:
        columns_to_check = [
            "role",
            "sector",
            "question_category",
            "question_level",
            "interview_stage",
        ]

    columns_to_check = [
        column
        for column in columns_to_check
        if column in df.columns
    ]


    sampled_df = (
        df.sample(
            n=n_samples,
            random_state=random_state
        )
        .reset_index(drop=True)
    )



    distribution_comparison = {}

    for column in columns_to_check:

        original_distribution = (
            df[column]
            .value_counts(normalize=True)
            .rename("original")
        )

        sampled_distribution = (
            sampled_df[column]
            .value_counts(normalize=True)
            .rename("sample")
        )

        comparison = pd.concat(
            [
                original_distribution,
                sampled_distribution
            ],
            axis=1
        ).fillna(0)


        comparison["difference"] = (
            comparison["sample"]
            - comparison["original"]
        )

        distribution_comparison[column] = comparison

    return sampled_df, distribution_comparison


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def assign_candidate_quality(
    df: pd.DataFrame,
    seed: int = 42
) -> pd.DataFrame:
    """
    Assign candidate quality approximately uniformly across the dataset.
    """

    rng = random.Random(seed)

    qualities = [
        QUALITY_LEVELS[i % len(QUALITY_LEVELS)]
        for i in range(len(df))
    ]

    rng.shuffle(qualities)

    df = df.copy()
    df["candidate_quality"] = qualities

    return df


def clean_value(value) -> str:
    """
    Convert NaN/None values into a safe textual representation.
    """
    if pd.isna(value):
        return "Not specified"

    return str(value).strip()


def build_prompt(
    template: str,
    row: pd.Series
) -> str:
    """
    Populate the teacher prompt using one InterviewForge row.
    """

    return template.format(
        role=clean_value(row.get("role")),
        sector=clean_value(row.get("sector")),
        interview_stage=clean_value(row.get("interview_stage")),
        interviewer=clean_value(row.get("interviewer")),
        question_category=clean_value(row.get("question_category")),
        question_level=clean_value(row.get("question_level")),
        candidate_quality=clean_value(row.get("candidate_quality")),
        question=clean_value(row.get("question"))
    )


def extract_json(text: str) -> dict:
    """
    Parse the model response as JSON.

    Also handles accidental surrounding text by extracting the first
    {...} block.
    """

    text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1 or end <= start:
            raise ValueError(
                f"No valid JSON object found in response:\n{text}"
            )

        candidate_json = text[start:end + 1]

        try:
            data = json.loads(candidate_json)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Malformed JSON returned by model:\n{text}"
            ) from exc

    missing = REQUIRED_OUTPUT_FIELDS - data.keys()

    if missing:
        raise ValueError(
            f"Missing required output fields: {missing}"
        )

    return {
        "candidate_answer": str(data["candidate_answer"]).strip(),
        "feedback": str(data["feedback"]).strip(),
        "follow_up_question": str(
            data["follow_up_question"]
        ).strip(),
    }


def load_teacher(model_name: str):
    print(f"Loading teacher model: {model_name}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    
    

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
    )

    model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cuda",
    dtype="auto",
    low_cpu_mem_usage=True,
)

    model.eval()

 

    return tokenizer, model


@torch.inference_mode()
def generate_teacher_response(
    prompts: list[str],
    tokenizer,
    model,
    generation_config: dict,
) -> str:

    messages_batch = [
    [
        {
            "role": "user",
            "content": prompt,
        }
    ]
    for prompt in prompts
    ]

    formatted_prompts = [
    tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    for messages in messages_batch
    ]

    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    inputs = tokenizer(
        formatted_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=generation_config.get(
            "max_new_tokens", 500
        ),
        do_sample=True,
        temperature=generation_config.get(
            "temperature", 0.7
        ),
        top_p=generation_config.get(
            "top_p", 0.8
        ),
        top_k=generation_config.get(
            "top_k", 20
        ),
        pad_token_id=tokenizer.eos_token_id,
    )

    input_length = inputs["input_ids"].shape[1]

    responses = []

    for output_ids in generated_ids:
        new_tokens = output_ids[input_length:]

        response = tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        )

        responses.append(response.strip())

    return responses



def save_checkpoint(
    df: pd.DataFrame,
    output_path: Path,
):
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output_path,
        index=False,
    )


def generate_dataset(
    input_path: str,
    output_path: str,
    config_path: str,
    prompt_path: str,
    seed: int = 42,
    checkpoint_every: int = 25,
    limit: int | None = None,
):

    config = load_yaml(config_path)
    prompt_template = load_prompt(prompt_path)

    teacher_config = config["teacher"]
    generation_config = config.get(
        "generation",
        {}
    )

    if teacher_config["provider"] != "huggingface":
        raise ValueError(
            "This script currently supports "
            "provider='huggingface' only."
        )

    model_name = teacher_config["model"]

    

    df = pd.read_csv(input_path)
    df, distribution = sample_dataset(df)

    if limit is not None:
        df = df.head(limit).copy()

    output_path = Path(output_path)

   

    if output_path.exists():
        print(
            f"Existing output found: {output_path}"
        )
        print("Resuming generation...")

        df = pd.read_csv(output_path)

        if "candidate_quality" not in df.columns:
            df = assign_candidate_quality(
                df,
                seed=seed,
            )

    else:
        df = assign_candidate_quality(
            df,
            seed=seed,
        )

        df["candidate_answer"] = None
        df["feedback"] = None
        df["follow_up_question"] = None
        df["generation_status"] = "pending"
        df["generation_error"] = None



    tokenizer, model = load_teacher(
        model_name
    )



    pending_indices = df.index[
        df["generation_status"] != "success"
    ].tolist()

    print(
        f"Examples to generate: "
        f"{len(pending_indices)}"
    )

    batch_size = 8

    for start in range(0, len(pending_indices), batch_size):

        batch_indices = pending_indices[
        start:start + batch_size
        ]

        prompts = [
        build_prompt(
            prompt_template,
            df.loc[idx]
        )
        for idx in batch_indices
        ]
        inizio = time.time()
        responses = generate_teacher_response(
        prompts=prompts,
        tokenizer=tokenizer,
        model=model,
        generation_config=generation_config,
        )
        print(time.time() - inizio)
        for idx, raw_response in zip(
        batch_indices,
        responses
        ):
            try:
                parsed = extract_json(raw_response)

                df.loc[
                idx,
                "candidate_answer"
                ] = parsed["candidate_answer"]

                df.loc[
                idx,
                "feedback"
                ] = parsed["feedback"]

                df.loc[
                idx,
                "follow_up_question"
                ] = parsed["follow_up_question"]

                df.loc[
                idx,
                "generation_status"
                ] = "success"

            except Exception as exc:

                df.loc[
                idx,
                "generation_status"
                ] = "failed"

                df.loc[
                idx,
                "generation_error"
                ] = str(exc)

    # Final save.
    save_checkpoint(
        df,
        output_path,
    )

    successful = (
        df["generation_status"] == "success"
    ).sum()

    failed = (
        df["generation_status"] == "failed"
    ).sum()

    print("\nGeneration completed.")
    print(f"Successful: {successful}")
    print(f"Failed:     {failed}")
    print(f"Saved to:  {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Input InterviewForge CSV",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output labelled CSV",
    )

    parser.add_argument(
        "--config",
        default="../configuration/models.yaml",
    )

    parser.add_argument(
        "--prompt",
        default="../prompts/teacher_generation.txt",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--limit",
        type=int,
        help=(
            "Optional number of samples. "
            "Useful for testing."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    generate_dataset(
        input_path=args.input,
        output_path=args.output,
        config_path=args.config,
        prompt_path=args.prompt,
        seed=args.seed,
        checkpoint_every=args.checkpoint_every,
        limit=args.limit,
    )