"""Metrics module for LoRAPID project."""

import csv
import random
import torch
import numpy as np
import yaml

import matplotlib.pyplot as plt
from torch import nn
import evaluate

metric = evaluate.load("mean_iou")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(eval_pred):
    with torch.no_grad():
        logits, labels = eval_pred
        # Handle logits and labels with proper dimensions
        if len(logits.shape) == 4:  # [batch, num_classes, height, width]
            logits_tensor = torch.from_numpy(logits)
            # Interpolate to match label dimensions
            if logits_tensor.shape[-2:] != labels.shape[-2:]:
                logits_tensor = nn.functional.interpolate(
                    logits_tensor,
                    size=labels.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            # Apply argmax to get predictions
            logits_tensor = logits_tensor.argmax(dim=1)
        else:  # logits are already [batch, height, width] after argmax
            logits_tensor = torch.from_numpy(logits).argmax(dim=-1) if logits.ndim == 3 else torch.from_numpy(logits)
        
        pred_labels = logits_tensor.detach().cpu().numpy()
        metrics = metric.compute(
            predictions=pred_labels,
            references=labels,
            num_labels=2,
            ignore_index=255,  # Use 255 as ignore_index as it's standard for segmentation
        )

        return metrics


class Metrics:
    def __init__(self, save_dir):
        self.save_dir = save_dir

    def save_yaml(self, vars: dict, file_name: str):
        with open(self.save_dir / file_name + "yaml", "w") as f:
            yaml.safe_dump(vars, f, sort_keys=False)

    def store_args(self, args):
        self.save_yaml(args, "config")

    def store_metrics(self, **args):
        self.save_yaml(args, "metrics")

    def store_training(self, training_log):
        with open(self.save_dir / "metrics.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_loss", "val_loss", "val_dice"])
            for i, (tl, vl, vd) in enumerate(
                zip(
                    training_log["train_losses"],
                    training_log["val_losses"],
                    training_log["val_dices"],
                ),
                start=1,
            ):
                w.writerow([i, tl, vl, vd])

    def plot(self, Y, x_label, y_label, title):
        _, y = Y.sample()
        plt.figure(size=(12, 10))

        for k, y in Y.items():
            plt.plot(y, label=k)

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.save_dir / title.replace(" ", "_"), dpi=120)
        plt.close()

    def plot_curves(self, tr_losses, eval_losses, eval_dices):
        self.plot(
            {"Training": tr_losses, "Evaluation": eval_losses},
            x_label="Epochs",
            y_label="Loss",
            title="Loss curve",
        )

        self.plot(
            {"Evaluation": eval_dices},
            x_label="Epochs",
            y_label="Dice",
            title="Dice Curve",
        )
