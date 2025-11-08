# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # SEG-PEFT - Targeted Ablation on 2D Segmentation
#
# This notebook performs a comprehensive ablation study comparing Full Fine-Tuning (FFT) against LoRA with varying rank and alpha configurations on the Kvasir-SEG dataset.
#
# ## Objective
# Evaluate how LoRA hyperparameters (rank r and scaling factor α) affect segmentation performance compared to FFT baseline.
#
# ## Experimental Setup
# - **Dataset**: Kvasir-SEG (gastrointestinal polyp segmentation)
# - **Model**: SegFormer-B0 pretrained on ADE20K
# - **Training**: 50 epochs, learning rate 5e-4, dropout 0.0
# - **Evaluation**: Epoch-based metrics (Mean Dice, Mean IoU, Mean Accuracy)
#
# ## Experiments
# 1. **FFT Baseline**: Full model fine-tuning
# 2. **LoRA Ablation**: 15 configurations with rank r ∈ {4, 8, 16, 32, 64} and α/r ratios ∈ {1, 2, 4}
# %%
!git clone https://github.com/rossoc/SEG-PEFT
%cd SEG-PEFT
!pip install evaluate
# %%
import torch
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from src.segpeft import kvasir_dataset, compute_metrics_fn, segformer, set_seed, Metrics
import time
import yaml
import pandas as pd
import os
import zipfile
from peft import get_peft_model, LoraConfig

set_seed(42)

# %%
# Colab utility: Download experiment results
# After each experiment, results are zipped and downloaded automatically
import shutil
from google.colab import files


def download_results(save_dir):
    """Zip and download experiment results folder"""
    output_path = f"./outputs/{save_dir}"
    zip_name = f"{save_dir}_results"

    # Create zip file
    shutil.make_archive(zip_name, "zip", output_path)

    # Download the zip file
    files.download(f"{zip_name}.zip")
    print(f"Downloaded {zip_name}.zip")


# %% [markdown]
# ## Dataset Setup
#
# Download and prepare the Kvasir-SEG dataset containing 1000 polyp images with corresponding segmentation masks. The dataset is split into 80% training and 20% validation.
#
# Dataset: [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/)

# %%
dataset_dir = "data"
os.makedirs(dataset_dir, exist_ok=True)
!wget --no-check-certificate https://datasets.simula.no/downloads/kvasir-seg.zip -O kvasir-seg.zip

with zipfile.ZipFile("kvasir-seg.zip", "r") as zip_ref:
    zip_ref.extractall(dataset_dir)


# %% [markdown]
# ## Baseline: Full Fine-Tuning (FFT)
#
# Train the complete SegFormer model with all parameters trainable. This serves as the performance upper bound for comparison with LoRA experiments.
#
# **Key characteristics:**
# - All 3.7M parameters are trainable
# - Higher computational cost and memory requirements
# - Evaluation and logging performed every epoch


# %%
batch_size = 64
gradient_accumulation_steps = 4
use_bf16 = True
dataloader_num_workers = 8


def train_segformer_fft(epochs, lr, save_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_size = 0.2
    model, model_name, _ = segformer()
    train_dataset, test_dataset = kvasir_dataset(model_name, test_size)
    N = len(train_dataset)

    training_args = TrainingArguments(
        output_dir="./outputs/" + save_dir,
        num_train_epochs=epochs,
        # A100 Optimization: Larger batch sizes
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,  # Can use larger batch for eval
        # A100 Optimization: Gradient accumulation for effective larger batches
        gradient_accumulation_steps=gradient_accumulation_steps,
        # A100 Optimization: Mixed precision with bfloat16 (A100's specialty)
        bf16=use_bf16 and torch.cuda.is_available(),
        bf16_full_eval=use_bf16 and torch.cuda.is_available(),
        # A100 Optimization: Efficient data loading
        dataloader_num_workers=dataloader_num_workers,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2,
        # Training settings - EPOCH-BASED
        learning_rate=lr,
        save_total_limit=2,
        prediction_loss_only=False,
        remove_unused_columns=True,
        push_to_hub=False,
        report_to="none",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",  # Log every epoch
        load_best_model_at_end=True,
        logging_dir=f"./outputs/{save_dir}/logs",
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics_fn(model_name),  # type: ignore
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
    )

    print("Starting training...")
    start_time = time.time()
    trainer.train()
    end_time = time.time() - start_time

    all_metrics = {
        "training_history": trainer.state.log_history,
        "final_evaluation": trainer.evaluate(),
        "training_time": end_time,
    }

    model.save_pretrained(f"./outputs/{save_dir}/final")
    log = trainer.state.log_history.copy()
    final_train_metrics = trainer.evaluate(eval_dataset=train_dataset)

    log.append(
        {
            "epoch": epochs,
            "loss": final_train_metrics["eval_loss"],
            "train_accuracy": final_train_metrics["eval_accuracy"],
            "train_dice": final_train_metrics["eval_mean_dice"],
        }
    )
    df = pd.DataFrame(log)
    df.to_csv(f"./outputs/{save_dir}/training_history.csv", index=False)
    with open(f"./outputs/{save_dir}/all_metrics.json", "w") as f:
        yaml.dump(all_metrics, f, indent=2)

    metrics = Metrics(f"./outputs/{save_dir}/")
    metrics.plot_curves(log)
    return trainer


# %%
# FFT Configuration
epochs = 50
learning_rate = 5e-4
save_dir = "segformer_fft_baseline"


# %%
# Train FFT baseline
fft_trainer = train_segformer_fft(epochs, learning_rate, save_dir)


# %%
# Download FFT results
download_results(save_dir)

# %% [markdown] id="cffa349c"
# ## Train
# [SegFormer](https://huggingface.co/docs/transformers/model_doc/segformer) with
# LoRA.
# Namely, we use [PEFT](https://github.com/huggingface/peft) to implmenent LoRA.


# %%
def train_segformer_lora(epochs, lr, r, lora_alpha, lora_dropout, save_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_size = 0.2
    model, model_name, modules = segformer()

    peft_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=modules,
    )

    model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()

    train_dataset, test_dataset = kvasir_dataset(model_name, test_size)
    N = len(train_dataset)

    training_args = TrainingArguments(
        output_dir="./outputs/" + save_dir,
        num_train_epochs=epochs,
        # A100 Optimization: Larger batch sizes
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,  # Can use larger batch for eval
        # A100 Optimization: Gradient accumulation for effective larger batches
        gradient_accumulation_steps=gradient_accumulation_steps,
        # A100 Optimization: Mixed precision with bfloat16 (A100's specialty)
        bf16=use_bf16 and torch.cuda.is_available(),
        bf16_full_eval=use_bf16 and torch.cuda.is_available(),
        # A100 Optimization: Efficient data loading
        dataloader_num_workers=dataloader_num_workers,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2,
        # Training settings - EPOCH-BASED
        learning_rate=lr,
        save_total_limit=2,
        prediction_loss_only=False,
        remove_unused_columns=True,
        push_to_hub=False,
        report_to="none",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",  # Log every epoch
        load_best_model_at_end=True,
        logging_dir=f"./outputs/{save_dir}/logs",
        # Performance optimization
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        # Better learning rate schedule
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics_fn(model_name),  # type: ignore
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
    )

    print("Starting training...")
    start_time = time.time()
    trainer.train()
    end_time = time.time() - start_time

    all_metrics = {
        "training_history": trainer.state.log_history,
        "final_evaluation": trainer.evaluate(),
        "training_time": end_time,
    }

    model.save_pretrained(f"./outputs/{save_dir}/final")
    log = trainer.state.log_history.copy()
    final_train_metrics = trainer.evaluate(eval_dataset=train_dataset)

    log.append(
        {
            "epoch": epochs,
            "loss": final_train_metrics["eval_loss"],
            "train_accuracy": final_train_metrics["eval_accuracy"],
            "train_dice": final_train_metrics["eval_mean_dice"],
        }
    )
    df = pd.DataFrame(log)
    df.to_csv(f"./outputs/{save_dir}/training_history.csv", index=False)
    with open(f"./outputs/{save_dir}/all_metrics.json", "w") as f:
        yaml.dump(all_metrics, f, indent=2)

    metrics = Metrics(f"./outputs/{save_dir}/")
    metrics.plot_curves(log)
    return trainer


# %% [markdown]
# ## LoRA Ablation Study
#
# Parameter-efficient fine-tuning using Low-Rank Adaptation (LoRA). Only low-rank matrices are trained while the base model remains frozen, drastically reducing trainable parameters (~1.7% of full model).
#
# **Fixed hyperparameters across all experiments:**
# - Epochs: 50
# - Learning rate: 5e-4
# - Dropout: 0.0
# - Evaluation: Every epoch
#
# **Variable hyperparameters:**
# - Rank (r): Controls capacity of low-rank adaptation matrices
# - Alpha (α): Scaling factor for LoRA updates
#
# Each experiment follows this pattern: higher rank = more parameters but potentially better performance.
#
# ---
#
# ### Experiment 1: r=4, α ∈ {4, 8, 16}
#
# Minimal parameter overhead (~0.17% trainable). Tests different α/r scaling ratios (1, 2, 4) with the smallest rank.

# %% id="14210062" outputId="a5d02028-af7e-42ed-bc54-0b494a6133d6"
# r=4, alpha=4
epochs = 50
learning_rate = 5e-4
rank = 4
lora_alpha = 4
lora_dropout = 0.0
save_dir = "lora_r4_alpha4"

print(f"Training LoRA with r={rank}, alpha={lora_alpha}")
trainer_r4_a4 = train_segformer_lora(
    epochs, learning_rate, rank, lora_alpha, lora_dropout, save_dir
)
# %%
download_results("lora_r4_alpha4")

# %% id="ye6lQ54BTSil"
# r=4, alpha=8
epochs = 50
learning_rate = 5e-4
rank = 4
lora_alpha = 8
lora_dropout = 0.0
save_dir = "lora_r4_alpha8"

print(f"Training LoRA with r={rank}, alpha={lora_alpha}")
trainer_r4_a8 = train_segformer_lora(
    epochs, learning_rate, rank, lora_alpha, lora_dropout, save_dir
)

# %%
download_results("lora_r4_alpha8")

# %%
# r=4, alpha=16
epochs = 50
learning_rate = 5e-4
rank = 4
lora_alpha = 16
lora_dropout = 0.0
save_dir = "lora_r4_alpha16"

print(f"Training LoRA with r={rank}, alpha={lora_alpha}")
trainer_r4_a16 = train_segformer_lora(
    epochs, learning_rate, rank, lora_alpha, lora_dropout, save_dir
)
# %%
download_results("lora_r4_alpha16")

# %% [markdown]
# ### Experiment 2: r=8, α ∈ {8, 16, 32}
#
# Double the rank capacity of Experiment 1. Evaluates if increased rank improves segmentation quality with the same α/r ratios.
# %%
# r=8, alpha=8
epochs = 50
learning_rate = 5e-4
rank = 8
lora_alpha = 8
lora_dropout = 0.0
save_dir = "lora_r8_alpha8"

print(f"Training LoRA with r={rank}, alpha={lora_alpha}")
trainer_r8_a8 = train_segformer_lora(
    epochs, learning_rate, rank, lora_alpha, lora_dropout, save_dir
)

# %%
download_results("lora_r8_alpha8")

# %%
# r=8, alpha=16
epochs = 50
learning_rate = 5e-4
rank = 8
lora_alpha = 16
lora_dropout = 0.0
save_dir = "lora_r8_alpha16"

print(f"Training LoRA with r={rank}, alpha={lora_alpha}")
trainer_r8_a16 = train_segformer_lora(
    epochs, learning_rate, rank, lora_alpha, lora_dropout, save_dir
)

# %%
download_results("lora_r8_alpha16")

# %%
# r=8, alpha=32
epochs = 50
learning_rate = 5e-4
rank = 8
lora_alpha = 32
lora_dropout = 0.0
save_dir = "lora_r8_alpha32"

print(f"Training LoRA with r={rank}, alpha={lora_alpha}")
trainer_r8_a32 = train_segformer_lora(
    epochs, learning_rate, rank, lora_alpha, lora_dropout, save_dir
)

# %%
download_results("lora_r8_alpha32")

# %% [markdown]
# ### Experiment 3: r=16, α ∈ {16, 32, 64}
#
# Medium rank configuration. Explores the performance-parameter tradeoff in the mid-range.

# %%
# r=16, alpha=16
epochs = 50
learning_rate = 5e-4
rank = 16
lora_alpha = 16
lora_dropout = 0.0
save_dir = "lora_r16_alpha16"

print(f"Training LoRA with r={rank}, alpha={lora_alpha}")
trainer_r16_a16 = train_segformer_lora(
    epochs, learning_rate, rank, lora_alpha, lora_dropout, save_dir
)

# %%
download_results("lora_r16_alpha16")

# %%
# r=16, alpha=32
epochs = 50
learning_rate = 5e-4
rank = 16
lora_alpha = 32
lora_dropout = 0.0
save_dir = "lora_r16_alpha32"

print(f"Training LoRA with r={rank}, alpha={lora_alpha}")
trainer_r16_a32 = train_segformer_lora(
    epochs, learning_rate, rank, lora_alpha, lora_dropout, save_dir
)

# %%
download_results("lora_r16_alpha32")

# %%
# r=16, alpha=64
epochs = 50
learning_rate = 5e-4
rank = 16
lora_alpha = 64
lora_dropout = 0.0
save_dir = "lora_r16_alpha64"

print(f"Training LoRA with r={rank}, alpha={lora_alpha}")
trainer_r16_a64 = train_segformer_lora(
    epochs, learning_rate, rank, lora_alpha, lora_dropout, save_dir
)

# %%
download_results("lora_r16_alpha64")

# %% [markdown]
# ### Experiment 4: r=32, α ∈ {32, 64, 128}
#
# High rank configuration with increased model capacity. Tests if higher rank approaches FFT performance.

# %%
# r=32, alpha=32
epochs = 50
learning_rate = 5e-4
rank = 32
lora_alpha = 32
lora_dropout = 0.0
save_dir = "lora_r32_alpha32"

print(f"Training LoRA with r={rank}, alpha={lora_alpha}")
trainer_r32_a32 = train_segformer_lora(
    epochs, learning_rate, rank, lora_alpha, lora_dropout, save_dir
)

# %%
download_results("lora_r32_alpha32")

# %%
# r=32, alpha=64
epochs = 50
learning_rate = 5e-4
rank = 32
lora_alpha = 64
lora_dropout = 0.0
save_dir = "lora_r32_alpha64"

print(f"Training LoRA with r={rank}, alpha={lora_alpha}")
trainer_r32_a64 = train_segformer_lora(
    epochs, learning_rate, rank, lora_alpha, lora_dropout, save_dir
)

# %%
download_results("lora_r32_alpha64")

# %%
# r=32, alpha=128
epochs = 50
learning_rate = 5e-4
rank = 32
lora_alpha = 128
lora_dropout = 0.0
save_dir = "lora_r32_alpha128"

print(f"Training LoRA with r={rank}, alpha={lora_alpha}")
trainer_r32_a128 = train_segformer_lora(
    epochs, learning_rate, rank, lora_alpha, lora_dropout, save_dir
)

# %%
download_results("lora_r32_alpha128")

# %% [markdown]
# ### Experiment 5: r=64, α ∈ {64, 128, 256}
#
# Maximum rank configuration with highest parameter count among LoRA experiments. Evaluates the upper limit of LoRA's expressiveness.

# %%
# r=64, alpha=64
epochs = 50
learning_rate = 5e-4
rank = 64
lora_alpha = 64
lora_dropout = 0.0
save_dir = "lora_r64_alpha64"

print(f"Training LoRA with r={rank}, alpha={lora_alpha}")
trainer_r64_a64 = train_segformer_lora(
    epochs, learning_rate, rank, lora_alpha, lora_dropout, save_dir
)

# %%
download_results("lora_r64_alpha64")

# %%
# r=64, alpha=128
epochs = 50
learning_rate = 5e-4
rank = 64
lora_alpha = 128
lora_dropout = 0.0
save_dir = "lora_r64_alpha128"

print(f"Training LoRA with r={rank}, alpha={lora_alpha}")
trainer_r64_a128 = train_segformer_lora(
    epochs, learning_rate, rank, lora_alpha, lora_dropout, save_dir
)

# %%
download_results("lora_r64_alpha128")

# %%
# r=64, alpha=256
epochs = 50
learning_rate = 5e-4
rank = 64
lora_alpha = 256
lora_dropout = 0.0
save_dir = "lora_r64_alpha256"

print(f"Training LoRA with r={rank}, alpha={lora_alpha}")
trainer_r64_a256 = train_segformer_lora(
    epochs, learning_rate, rank, lora_alpha, lora_dropout, save_dir
)

# %%
download_results("lora_r64_alpha256")
