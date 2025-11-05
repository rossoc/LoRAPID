import random
import torch
import numpy as np
import yaml

import matplotlib.pyplot as plt
from torch import nn
import evaluate
from sklearn.metrics import f1_score
import pandas as pd


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


metric = evaluate.load("mean_iou")


def compute_metrics(eval_pred):
    with torch.no_grad():
        logits, labels = eval_pred

        logits_tensor = torch.tensor(logits)
        labels_tensor = torch.tensor(labels)

        logits_tensor = nn.functional.interpolate(
            logits_tensor,
            size=labels_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        pred_labels = logits_tensor.argmax(dim=1)

        pred_flat = pred_labels.view(-1).cpu().numpy()
        labels_flat = labels_tensor.view(-1).cpu().numpy()

        num_classes = logits_tensor.shape[1]
        dice_scores = []
        for class_idx in range(num_classes):
            pred_binary = (pred_flat == class_idx).astype(int)
            labels_binary = (labels_flat == class_idx).astype(int)
            dice_scores.append(float(f1_score(pred_binary, labels_binary)))

        mean_dice = float(np.mean(dice_scores))
        results = metric.compute(
            predictions=pred_labels.int(),
            references=labels_tensor.int(),
            num_labels=2,
            ignore_index=True,
        )

        return {
            "mean_iou": float(results["mean_iou"]),
            "mean_dice": mean_dice,
            "accuracy": float(results["mean_accuracy"]),
        }


class Metrics:
    def __init__(self, save_dir):
        self.save_dir = save_dir

    def save_yaml(self, vars: dict, file_name: str):
        with open(self.save_dir + file_name + "yaml", "w") as f:
            yaml.dump(vars, f, sort_keys=False, indent=2)

    def store_history(self, log):
        df = pd.DataFrame(log)
        df.to_csv(self.save_dir + "history.csv")

    def store_metrics(self, args):
        self.save_yaml(args, "metrics")

    def plot(self, Y, x_label, y_label, title):
        plt.figure(figsize=(12, 10))
        for k, y in Y.items():
            plt.plot(y, label=k)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        plt.xlim(left=0)
        plt.ylim(bottom=0)

        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.save_dir + title.replace(" ", "_"), dpi=120)
        plt.close()

    def plot_curves(self, log):
        self.plot(
            Y={
                "Training": [
                    entry["loss"]
                    for entry in log
                    if entry["epoch"] % 1 == 0 and "loss" in entry.keys()
                ],
                "Evaluation": [
                    entry["eval_loss"]
                    for entry in log
                    if entry["epoch"] % 1 == 0 and "eval_loss" in entry.keys()
                ],
            },
            x_label="Epochs",
            y_label="Losses",
            title="Losses over Epochs",
        )

        self.plot(
            Y={
                "Evaluation": [
                    entry["eval_mean_dice"]
                    for entry in log
                    if entry["epoch"] % 1 == 0 and "eval_mean_dice" in entry.keys()
                ],
            },
            x_label="Epochs",
            y_label="Dice",
            title="Dice over Epochs",
        )
