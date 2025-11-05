#!/usr/bin/env python
"""
Training script for SegFormer model on Kvasir-SEG dataset.
This script trains a segmentation model to identify polyps in medical images.
"""

import torch
from transformers import (
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from argparse import ArgumentParser
from segpeft import kvasir_dataset, compute_metrics, mask2former, set_seed, Metrics
import time
import yaml
import pandas as pd


def main(epochs, lr, save_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_size = 0.2
    model, model_name, _ = mask2former()
    train_dataset, test_dataset = kvasir_dataset(model_name, test_size)

    training_args = TrainingArguments(
        output_dir="./outputs/" + save_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        learning_rate=lr,
        save_total_limit=2,
        prediction_loss_only=False,
        remove_unused_columns=True,
        push_to_hub=False,
        report_to="none",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir=f"./outputs/{save_dir}/logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset[:2],
        eval_dataset=test_dataset[:2],
        compute_metrics=compute_metrics,  # type: ignore
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    print("Starting training...")
    start_time = time.time()
    trainer.train()
    end_time = time.time() - start_time

    log = trainer.state.log_history.copy()
    final_train_metrics = trainer.evaluate(eval_dataset=train_dataset)
    log.append({"epoch": epochs, "loss": final_train_metrics["eval_loss"]})
    all_metrics = {
        "training_history": log,
        "final_evaluation": trainer.evaluate(),
        "training_time": end_time,
    }
    metrics = Metrics(f"./outputs/{save_dir}/")
    metrics.store_metrics(all_metrics)
    metrics.store_history(log)
    metrics.plot_curves(log)
    return trainer


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--epochs", default=30)
    ap.add_argument("--lr", default=5e-5)
    ap.add_argument("--save-dir")
    ap.add_argument("--seed", default=42)
    args = ap.parse_args()

    set_seed(args.seed)

    main(args.epochs, args.lr, args.save_dir)
