"""
ViTucano LoRA fine-tuning (standalone entrypoint)
=================================================

Same pipeline as `train.py`, restricted to ViTucano and adapted for GPUs
without bf16 support (Kaggle T4/P100, Colab T4):

- the model is loaded in bf16 when the GPU supports it, otherwise fp16;
- the matching mixed-precision flag (bf16/fp16) is set automatically;
- LoRA weights are kept in fp32 (required for fp16 mixed precision);
- `--train-images` / `--test-images` allow quick smoke tests.

The original `train.py`, collators and metrics are reused unchanged.

Usage
-----
    python train_vitucano.py                                   # full run, config/config_vitucano.yml
    python train_vitucano.py --train-images 20 --test-images 10 --monitoring-steps 10 --no-push
"""

import os
import argparse

import yaml
import torch
import wandb
import pandas as pd

from peft import get_peft_model, prepare_model_for_kbit_training
from pprint import pprint
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments,BitsAndBytesConfig, set_seed

from config.config import config_vars, create_lora_config, ViTucanoProcessor
from data_prep.data_processing import load_datasets, preprocess, transform_datasets
from data_prep.data_collator import DataCollatorForTraining, DataCollatorForGeneration
from evaluation.eval_prediction import evaluate_predictions
from evaluation.eval_finetuning import compute_metrics
from generation.generation import batch_generation


def bf16_supported():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8


def load_vitucano(model_id, use_flash_attention=False,use_bnb = False):
    """Load ViTucano and its processor, in bf16 or fp16 depending on the GPU."""
    dtype = torch.bfloat16 if bf16_supported() else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_storage="nf4",
        bnb_4bit_compute_dtype=dtype, #fp16 on specific GPUs
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
        _attn_implementation='flash_attention_2' if use_flash_attention else 'eager',
        quantization_config = bnb_config,
        device_map={"":0} if use_bnb else None,
    )
    processor = ViTucanoProcessor(
        image_processor=model.vision_tower._image_processor,
        tokenizer=AutoTokenizer.from_pretrained(model_id, trust_remote_code=True),
    )
    return model, processor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config_vitucano.yml")
    parser.add_argument("--train-images", type=int, default=None, help="use only the first N training images")
    parser.add_argument("--test-images", type=int, default=None, help="use only the first N test images")
    parser.add_argument("--monitoring-steps", type=int, default=None)
    parser.add_argument("--no-push", action="store_true", help="never push to the HF Hub")
    return parser.parse_args()


def main():
    args = parse_args()
   
    load_dotenv(dotenv_path="../.env")
    if os.getenv("HF_API_KEY"):
        login(os.getenv("HF_API_KEY"))
    if not os.getenv("WANDB_API_KEY"):
        os.environ["WANDB_MODE"] = "disabled"

    with open(args.config, "r") as file:
        raw = yaml.safe_load(file)

    # CLI overrides
    if args.monitoring_steps is not None:
        raw["config"]["monitoring_steps"] = args.monitoring_steps
    if args.no_push:
        raw["config"]["push_to_hub"] = False
    train_n = args.train_images if args.train_images is not None else raw["config"].get("train_images")
    test_n = args.test_images if args.test_images is not None else raw["config"].get("test_images")

    setups = config_vars(raw)
    config = setups["config"]
    qlora_args = setups["qlora_args"]
    training_args = setups["training_args"]
    generate_args = setups["generate_args"]

    # Precision from the GPU: exactly one of bf16 / fp16
    training_args.pop("bf16", None)
    training_args.pop("fp16", None)
    training_args["bf16" if bf16_supported() else "fp16"] = True

    print("\nPrecision:", "bf16" if bf16_supported() else "fp16")
    pprint({k: config[k] for k in ["model_id", "dataset", "push_to_hub", "monitoring_steps"]})

    os.makedirs(config["results_dir"], exist_ok=True)
    
    set_seed(config.get("seed",42))

    # 1. Model + LoRA
    use_bnb = qlora_args.get("use_bnb",False)
    model, processor = load_vitucano(config["model_id"], config["use_flash_attention"],use_bnb)
    if use_bnb:
        model = prepare_model_for_kbit_training(
            model,use_gradient_checkpointing = True,
            gradient_checkpoiting_kwargs = {"use_reetrant":False},
        )

    model = get_peft_model(model, create_lora_config(
        model_id=config["model_id"],
        rank=qlora_args["lora_rank"],
        linear_modules=qlora_args["linear_modules"],
        alpha_to_rank_ratio=qlora_args["alpha_to_rank_ratio"],
        dropout=qlora_args["dropout"],
        is_all_linear=qlora_args["lora_all_linear"],
    ))
    # fp16 mixed precision needs trainable weights in fp32
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    model.print_trainable_parameters()

    # 2. Data
    train_ds, valid_ds, test_ds = load_datasets(
        data_dir=config["data_dir"],
        hf_dataset=config["hf_dataset"],
        dataset_from_hub=config["dataset_from_hub"],
    )
    if train_n:
        train_ds = train_ds.select(range(min(train_n, len(train_ds))))
        valid_ds = valid_ds.select(range(min(10, len(valid_ds))))
    if test_n:
        test_ds = test_ds.select(range(min(test_n, len(test_ds))))

    print(f"\nDataset\n\tTrain: {len(train_ds)}\n\tVal: {len(valid_ds)}\n\tTest: {len(test_ds)}\n")

    train_dataset, valid_dataset, _ = transform_datasets(
        train_ds=train_ds,
        valid_ds=valid_ds,
        test_ds=test_ds,
        preprocess_fn=preprocess(
            question=config["question"].format(max_length=config["max_length"]),
            text_per_image=config["text_per_image"],
            image_column=config["image_column"],
            text_column=config["text_column"],
        ),
    )

    model.config.use_cache = False
    model.generation_config.max_new_tokens = config["max_length"]

    kwargs_collator = {
        "model_id": config["model_id"],
        "device": config["device"],
        "processor": processor,
        "max_length": config["max_length"],
        "question": config["question"].format(max_length=config["max_length"]),
    }

    # 3. Train (resumes from the last checkpoint if output_dir already has one)
    trainer = Seq2SeqTrainer(
        model=model,
        args=Seq2SeqTrainingArguments(**training_args),
        compute_metrics=compute_metrics(processor=processor, model_id=config["model_id"]),
        data_collator=DataCollatorForTraining(**kwargs_collator),
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=processor,
    )

    output_dir = training_args["output_dir"]
    with wandb.init(project=os.getenv("WANDB_PROJECT_NAME", "vitucano-ptbr")) as run:
        run.name = f'{config["model_name"]}-ft-{config["dataset"]}'
        if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
            try:
                trainer.train(resume_from_checkpoint=True)
            except Exception:
                trainer.train()
        else:
            trainer.train()

    # 4. Save the adapter (a failure here must not lose the evaluation)
    pd.DataFrame(trainer.state.log_history).to_csv(
        os.path.join(config["results_dir"], "training_log_history.csv"), index=False
    )
    try:
        model.save_pretrained(config["model_dir"])
        processor.save_pretrained(config["model_dir"])
        if config["push_to_hub"]:
            hub_name = f'{config["model_name"]}-{config["dataset"]}'
            model.push_to_hub(hub_name, private=True)
            processor.push_to_hub(hub_name, private=True)
    except Exception as error:
        print(f"WARNING: saving/pushing failed ({error!r}); checkpoints are still in {output_dir}")

    # 5. Evaluate on the test set
    evaluate_predictions(
        raw_dataset=test_ds,
        predictions=batch_generation(
            raw_dataset=test_ds,
            model=trainer.model,
            config=config,
            collate_fn=DataCollatorForGeneration(**kwargs_collator),
            processor=processor,
            generate_args=generate_args,
        ),
        text_column=config["text_column"],
        results_dir=config["results_dir"],
    )


if __name__ == "__main__":
    main()
