import argparse
from pathlib import Path

from utilities_function import(
    load_configuration,
    load_and_clean_dataset,
    prepare_dataset,
)

import pandas as pd
import torch

from transformers import (
    AutoProcessor,
    AutoModelForMultimodalLM,
    BitsAndBytesConfig,
    set_seed,
)

from trl import SFTConfig, SFTTrainer
from peft import prepare_model_for_kbit_training, LoraConfig
from bitsandbytes.nn import Linear4bit


def get_qlora_target_modules(model):
    allowed_suffixes = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }

    targets = []

    for name, module in model.named_modules():
        if (
            isinstance(module, Linear4bit)
            and name.split(".")[-1] in allowed_suffixes
        ):
            targets.append(name)

    return targets


def load_student(model_name: str, is_training: bool = True):

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    processor = AutoProcessor.from_pretrained(
                    model_name
            )

    model = AutoModelForMultimodalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    )

    if is_training:
        model.train()
        # Prepare quantized model for LoRA training
        model = prepare_model_for_kbit_training(
        model
        )

        model.config.use_cache = False
    else:
        model.eval()

    return processor, model

def build_lora_config(config: dict, target_modules: list):

    lora = config["lora"]

    return LoraConfig(
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],

        target_modules=target_modules,

        bias="none",
        task_type="CAUSAL_LM",
    )


def train(
    dataset_path: str,
    config_path: str,
):

    config = load_configuration(config_path)

    training_config = config["training"]
    student_config = config["student"]

    seed = training_config.get("seed", 42)

    set_seed(seed)

    # -------------------------
    # Dataset
    # -------------------------

    df = load_and_clean_dataset(
        dataset_path
    )

    train_dataset, eval_dataset, test_set = prepare_dataset(
        df=df,
        seed=seed,
    )
    test_set.to_json(
    "data/processed/test_dataset.jsonl"
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(eval_dataset)}")

    print("\nExample:")
    print(train_dataset[0])

    # -------------------------
    # Model
    # -------------------------

    tokenizer, model = load_student(
        student_config["model"]
    )

    # -------------------------
    # LoRA
    # -------------------------

    target_modules = get_qlora_target_modules(model)

    peft_config = build_lora_config(
        config,
        target_modules=target_modules
    )

    # -------------------------
    # SFT
    # -------------------------

    sft_config = SFTConfig(
    output_dir=training_config["output_dir"],

    num_train_epochs=training_config["epochs"],

    learning_rate=training_config["learning_rate"],

    per_device_train_batch_size=
        training_config["per_device_train_batch_size"],

    per_device_eval_batch_size=
        training_config["per_device_eval_batch_size"],

    gradient_accumulation_steps=
        training_config["gradient_accumulation_steps"],

    max_length=training_config["max_length"],

    completion_only_loss=True,

    bf16=True,

    optim="paged_adamw_8bit",

    logging_steps=training_config["logging_steps"],

    eval_strategy="steps",
    eval_steps=training_config["eval_steps"],

    save_strategy="steps",
    save_steps=training_config["save_steps"],

    save_total_limit=2,

    warmup_steps=training_config["warmup_ratio"],
    weight_decay=training_config["weight_decay"],

    report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,

        train_dataset=train_dataset,
        eval_dataset=eval_dataset,

        processing_class=tokenizer,
        peft_config=peft_config,
    )



    trainer.train()



    output_dir = Path(
        training_config["output_dir"]
    )

    adapter_dir = output_dir / "final_adapter"

    trainer.save_model(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    print(
        f"\nLoRA adapter saved to: {adapter_dir}"
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
    )

    parser.add_argument(
        "--config",
        default="../configuration/models.yaml",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    train(
        dataset_path=args.dataset,
        config_path=args.config,
    )