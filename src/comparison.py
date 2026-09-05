from utilities_function import load_configuration, format_few_shot_prompt, load_prompt
from fine_tuning_LoRa import load_student
import argparse
import gc
from pathlib import Path
import pandas as pd
import torch
from peft import PeftModel
from tqdm import tqdm


def load_test_dataset(path: str) -> pd.DataFrame:


    df = pd.read_json(
        path,
        lines=True,
    )

    print(f"Test samples: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")

    return df


def extract_test_example(row: pd.Series) -> dict:

    required = {
        "role",
        "question",
        "candidate_answer",
    }

    if required.issubset(row.index):

        role = str(row["role"])
        question = str(row["question"])
        candidate_answer = str(
            row["candidate_answer"]
        )


        reference_feedback = (
            str(row["feedback"])
            if "feedback" in row
            else None
        )

        reference_follow_up = (
            str(row["follow_up_question"])
            if "follow_up_question" in row
            else None
        )

        return {
            "role": role,
            "question": question,
            "candidate_answer": candidate_answer,
            "reference_feedback": reference_feedback,
            "reference_follow_up": reference_follow_up,
        }

    if "prompt" in row and "completion" in row:

        prompt_messages = row["prompt"]
        completion_messages = row["completion"]

        # Pandas may represent them as numpy/list objects.
        prompt_messages = list(prompt_messages)
        completion_messages = list(
            completion_messages
        )

        input_text = prompt_messages[0]["content"]

        completion_text = (
            completion_messages[0]["content"]
        )

        feedback, follow_up = (
            parse_reference_completion(
                completion_text
            )
        )

        return {
            "input_text": input_text,
            "role": None,
            "question": None,
            "candidate_answer": None,
            "reference_feedback": feedback,
            "reference_follow_up": follow_up,
        }

    raise ValueError(
        "Unsupported test-set schema. "
        f"Available columns: {row.index.tolist()}"
    )


def parse_reference_completion(
    text: str
) -> tuple[str | None, str | None]:

    feedback_marker = "Feedback:"
    follow_up_marker = "Follow-up question:"

    if (
        feedback_marker in text
        and follow_up_marker in text
    ):
        feedback_part = text.split(
            feedback_marker,
            1
        )[1]

        feedback, follow_up = feedback_part.split(
            follow_up_marker,
            1
        )

        return (
            feedback.strip(),
            follow_up.strip(),
        )

    return text.strip(), None


def build_prompt(template: str, input_text: dict) -> str:
    return template.format(
        input_text=input_text["input_text"],
    )   


def load_qlora_adapter(
    base_model,
    adapter_path: str,
):

    print(
        f"\nLoading QLoRA adapter: "
        f"{adapter_path}"
    )

    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        is_trainable=False,
    )

    model.eval()

    return model


# =========================================================
# Generation
# =========================================================

@torch.inference_mode()
def generate_batch(
    prompts: list[str],
    model,
    processor,
    max_new_tokens: int,
) -> list[str]:



    formatted_prompts = []

    for prompt in prompts:

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ],
            }
        ]

        formatted = (
            processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

        formatted_prompts.append(
            formatted
        )

    tokenizer = processor.tokenizer
    tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = (
            tokenizer.eos_token_id
        )

    inputs = tokenizer(
        formatted_prompts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(model.device)

    generated = model.generate(
        **inputs,

        # deterministic evaluation
        do_sample=False,

        max_new_tokens=max_new_tokens,

        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    input_length = (
        inputs["input_ids"].shape[1]
    )

    outputs = []

    for generated_ids in generated:

        new_tokens = generated_ids[
            input_length:
        ]

        text = tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        outputs.append(
            text.strip()
        )

    return outputs



def parse_generation(
    text: str
) -> tuple[str | None, str | None]:

    feedback_marker = "Feedback:"
    follow_up_marker = "Follow-up question:"

    if (
        feedback_marker in text
        and follow_up_marker in text
    ):
        body = text.split(
            feedback_marker,
            1
        )[1]

        feedback, follow_up = body.split(
            follow_up_marker,
            1
        )

        return (
            feedback.strip(),
            follow_up.strip(),
        )

    # Generation is still saved even if parsing fails.
    return None, None



def run_condition(
    test_examples: list[dict],
    condition: str,
    inference_config,
    model,
    processor,
    output_path: Path,
    batch_size: int,
    max_new_tokens: int,
):

    if condition == "zero_shot":
        prompt_template = load_prompt(
            path=inference_config["prompts"]["minimal"]
        )

    elif condition == "few_shot":
        prompt_template = format_few_shot_prompt(
            inference_config["prompts"]["engineered"],
            inference_config["few_shot_dataset"],
        )
    elif condition == "qlora":
        prompt_template = None
    else:
        raise ValueError(
            f"Unknown condition: {condition}"
        )

    records = []

    for start in tqdm(
        range(
            0,
            len(test_examples),
            batch_size,
        ),
        desc=condition,
    ):

        batch = test_examples[
            start:start + batch_size
        ]
        if condition != "qlora":
            prompts = [
            build_prompt(
                prompt_template,
                example
            )
            for example in batch
            ]
        else:
            prompts = [
                example['input_text']
                for example in batch
            ]

        generations = generate_batch(
            prompts=prompts,
            model=model,
            processor=processor,
            max_new_tokens=max_new_tokens,
        )

        for example, generated_text in zip(
            batch,
            generations,
        ):

            generated_feedback, \
            generated_follow_up = (
                parse_generation(
                    generated_text
                )
            )

            records.append(
                {
                    "id": example["id"],

                    "condition": condition,

                    "input_text":
                        example["input_text"],

                    "role":
                        example["role"],

                    "question":
                        example["question"],

                    "candidate_answer":
                        example["candidate_answer"],

                    "reference_feedback":
                        example[
                            "reference_feedback"
                        ],

                    "reference_follow_up":
                        example[
                            "reference_follow_up"
                        ],

                    "generated_feedback":
                        generated_feedback,

                    "generated_follow_up":
                        generated_follow_up,

                    "raw_generation":
                        generated_text,

                    "parse_success":
                        (
                            generated_feedback
                            is not None
                            and generated_follow_up
                            is not None
                        ),
                }
            )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(records).to_json(
        output_path,
        orient="records",
        lines=True,
        force_ascii=False,
    )

    parsed = sum(
        record["parse_success"]
        for record in records
    )

    print(
        f"{condition}: "
        f"{parsed}/{len(records)} "
        f"outputs parsed successfully."
    )


def cleanup():

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()



def main(
    test_path: str,
    config_path: str,
):

    config = load_configuration(
        config_path
    )

    model_name = (
        config["student"]["model"]
    )

    adapter_path = (
        config["student"]["adapter"]
    )

    inference_config = (
        config["inference"]
    )

    output_dir = Path(
        inference_config["output_dir"]
    )

    batch_size = (
        inference_config.get(
            "batch_size",
            4,
        )
    )

    max_new_tokens = (
        inference_config.get(
            "max_new_tokens",
            350,
        )
    )



    df = load_test_dataset(
        test_path
    )

    examples = []

    for idx, row in df.iterrows():

        example = extract_test_example(
            row
        )

        example["id"] = int(idx)

        examples.append(
            example
        )

    processor, base_model = (
        load_student(
            model_name,
            is_training=False,
        )
    )
    """
    run_condition(
        test_examples=examples,
        condition="zero_shot",
        inference_config=inference_config,
        model=base_model,
        processor=processor,
        output_path=
            output_dir /
            "zero_shot_predictions.jsonl",
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )

    run_condition(
        test_examples=examples,
        condition="few_shot",
        inference_config=inference_config,
        model=base_model,
        processor=processor,
        output_path=
            output_dir /
            "prompt_predictions.jsonl",
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )
    """
    # =========================================
    # QLoRA
    # =========================================

    qlora_model = load_qlora_adapter(
        base_model=base_model,
        adapter_path=adapter_path,
    )

    run_condition(
        test_examples=examples,
        condition="qlora",
        inference_config=inference_config,
        model=qlora_model,
        processor=processor,
        output_path=
            output_dir /
            "qlora_predictions.jsonl",
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )

    print(
        "\nInference completed."
    )



def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test",
        required=True,
    )

    parser.add_argument(
        "--config",
        default="../configuration/models.yaml",
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    main(
        test_path=args.test,
        config_path=args.config,
    )