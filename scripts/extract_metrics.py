#!/usr/bin/env python
"""
Script to extract metrics from training history and all_metrics.json files in outputs directory.
"""

import os
import pandas as pd
import yaml
import re
from typing import Dict, List
from segpeft.metrics import Metrics
import matplotlib.pyplot as plt
import numpy as np
import zipfile
import tempfile
import shutil


def extract_metrics_from_folder(folder_path: str) -> Dict:
    """
    Extract metrics from a single folder in outputs directory.

    Args:
        folder_path: Path to the folder containing training_history.csv and all_metrics.json

    Returns:
        Dictionary with the specified metrics format
    """
    # Read training history CSV
    csv_path = os.path.join(folder_path, "training_history.csv")
    training_history = pd.read_csv(csv_path)

    # Read all_metrics.json (which is actually a YAML file)
    yaml_path = os.path.join(folder_path, "all_metrics.json")
    with open(yaml_path, "r") as f:
        all_metrics = yaml.safe_load(f)

    # Extract training loss, eval loss, and eval dice for epochs where epoch % 1 == 0
    train_loss = []
    eval_loss = []
    eval_dice = []

    for _, row in training_history.iterrows():
        # Only include entries where epoch % 1 == 0 (every epoch)
        if pd.notna(row["epoch"]) and row["epoch"] % 1 == 0:
            # Training loss
            if "train_loss" in row and pd.notna(row["train_loss"]):
                train_loss.append(row["train_loss"])
            elif "loss" in row and pd.notna(row["loss"]):
                train_loss.append(row["loss"])

            # Evaluation loss
            if "eval_loss" in row and pd.notna(row["eval_loss"]):
                eval_loss.append(row["eval_loss"])

            # Evaluation dice
            if "eval_mean_dice" in row and pd.notna(row["eval_mean_dice"]):
                eval_dice.append(row["eval_mean_dice"])

    # Extract training time from all_metrics
    training_time = all_metrics.get("training_time", 0)  # training_time is in seconds

    return {
        "train_loss": train_loss,
        "eval_loss": eval_loss,
        "eval_dice": eval_dice,
        "training_time": training_time,
    }


def extract_all_metrics(outputs_dir: str = "./outputs") -> Dict:
    """
    Extract metrics from all folders in the outputs directory.

    Args:
        outputs_dir: Path to the outputs directory

    Returns:
        Dictionary mapping folder names to their respective metrics
    """
    results = {}

    # List all subdirectories in outputs directory
    for folder_name in os.listdir(outputs_dir):
        folder_path = os.path.join(outputs_dir, folder_name)

        # Check if it's a directory and contains the required files
        if os.path.isdir(folder_path):
            csv_path = os.path.join(folder_path, "training_history.csv")
            yaml_path = os.path.join(folder_path, "all_metrics.json")

            if os.path.exists(csv_path) and os.path.exists(yaml_path):
                print(f"Processing {folder_name}...")
                try:
                    metrics = extract_metrics_from_folder(folder_path)
                    results[folder_name] = metrics
                except Exception as e:
                    print(f"Error processing {folder_name}: {str(e)}")
            else:
                # Some folders have the files in a subfolder (like 'final', or in the case of the non-results folders,
                # the files are directly in a subfolder like lora_r4_alpha16/lora_r4_alpha16/)
                for subfolder_name in os.listdir(folder_path):
                    subfolder_path = os.path.join(folder_path, subfolder_name)
                    if os.path.isdir(subfolder_path):
                        sub_csv_path = os.path.join(
                            subfolder_path, "training_history.csv"
                        )
                        sub_yaml_path = os.path.join(subfolder_path, "all_metrics.json")

                        if os.path.exists(sub_csv_path) and os.path.exists(
                            sub_yaml_path
                        ):
                            print(f"Processing {folder_name}/{subfolder_name}...")
                            try:
                                metrics = extract_metrics_from_folder(subfolder_path)
                                results[folder_name] = (
                                    metrics  # Use parent folder name as key
                                )
                                break  # Found the files, stop searching in subfolders
                            except Exception as e:
                                print(
                                    f"Error processing {folder_name}/{subfolder_name}: {str(e)}"
                                )

    return results


def get_lora_params_from_folder_name(folder_name: str) -> tuple:
    """
    Extract LoRA parameters from folder name.

    Args:
        folder_name: Name of the folder

    Returns:
        Tuple of (rank, alpha) or (None, None) if not a LoRA folder
    """
    # Look for patterns like "lora_r{rank}_alpha{alpha}" in the folder name
    lora_match = re.match(r".*lora_r(\d+)_alpha(\d+).*", folder_name)
    if lora_match:
        rank = int(lora_match.group(1))
        alpha = int(lora_match.group(2))
        return rank, alpha

    return None, None


def create_plots(metrics_dict: Dict):
    """
    Create the required plots using the Metrics class.

    Args:
        metrics_dict: Dictionary with folder names as keys and metrics as values
    """
    # Create a Metrics instance to save the plots
    metrics_instance = Metrics("./outputs/")

    # Separate LoRA results from baseline (full fine-tuning)
    lora_results = {}
    baseline_results = {}

    for folder_name, metrics in metrics_dict.items():
        rank, alpha = get_lora_params_from_folder_name(folder_name)
        if rank is not None and alpha is not None:
            lora_results[folder_name] = {
                "metrics": metrics,
                "rank": rank,
                "alpha": alpha,
            }
        else:
            # This is likely the baseline (full fine-tuning) folder
            baseline_results[folder_name] = metrics

    # 1. LoRA alpha vs dice, connecting models with the same rank (excluding full fine-tuning)
    if lora_results:
        fig, ax = plt.subplots(figsize=(12, 8))

        # Group by rank to connect models with same rank
        ranks = {}
        for folder_name, data in lora_results.items():
            rank = data["rank"]
            alpha = data["alpha"]
            metrics = data["metrics"]

            if rank not in ranks:
                ranks[rank] = {"alphas": [], "dices": [], "folders": []}

            if metrics["eval_dice"]:  # Only plot if there are eval dices
                final_eval_dice = metrics["eval_dice"][-1]  # Last eval dice
                ranks[rank]["alphas"].append(alpha)
                ranks[rank]["dices"].append(final_eval_dice)
                ranks[rank]["folders"].append(folder_name)

        # Plot for each rank, without connecting the dots and without alpha value labels
        for rank, data in ranks.items():
            # Sort by alpha
            sorted_data = sorted(zip(data["alphas"], data["dices"], data["folders"]))
            sorted_alphas, sorted_dices, sorted_folders = zip(*sorted_data)

            ax.scatter(sorted_alphas, sorted_dices, label=f"r={rank}", s=80, zorder=5)

        ax.set_xlabel(r"LoRA $\alpha$", fontsize=14)
        ax.set_ylabel("Dice", fontsize=14)
        ax.set_title(r"LoRA $\alpha$ vs Final Evaluation Dice", fontsize=14)
        ax.tick_params(
            axis="both", which="major", labelsize=14
        )  # Increase tick label font size
        ax.legend(fontsize=14)
        ax.grid(True)
        plt.tight_layout()
        plt.savefig("./outputs/lora_alpha_vs_dice.png", dpi=120)
        plt.close()

    # Find the best LoRA model based on final eval loss
    best_lora_folder = None
    best_lora_eval_loss = float("inf")

    for folder_name, data in lora_results.items():
        metrics = data["metrics"]
        if metrics["eval_loss"] and metrics["eval_loss"][-1] < best_lora_eval_loss:
            best_lora_eval_loss = metrics["eval_loss"][-1]
            best_lora_folder = folder_name

    # 2. Epochs vs loss of best LoRA model vs full fine-tuning
    if best_lora_folder and baseline_results:
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot best LoRA model losses
        best_lora_metrics = lora_results[best_lora_folder]["metrics"]

        # For training loss, we need to determine x-axis (epochs)
        epochs_train_lora = list(range(len(best_lora_metrics["train_loss"])))
        epochs_eval_lora = list(range(len(best_lora_metrics["eval_loss"])))

        ax.plot(
            epochs_train_lora,
            best_lora_metrics["train_loss"],
            label=f"Best LoRA Train (r={lora_results[best_lora_folder]['rank']}, a={lora_results[best_lora_folder]['alpha']})",
            linestyle="--",
        )
        ax.plot(
            epochs_eval_lora,
            best_lora_metrics["eval_loss"],
            label=f"Best LoRA Eval (r={lora_results[best_lora_folder]['rank']}, a={lora_results[best_lora_folder]['alpha']})",
            linestyle="-",
        )

        # Plot baseline (full fine-tuning) losses
        for folder_name, metrics in baseline_results.items():
            epochs_train_baseline = list(range(len(metrics["train_loss"])))
            epochs_eval_baseline = list(range(len(metrics["eval_loss"])))

            ax.plot(
                epochs_train_baseline,
                metrics["train_loss"],
                label=f"Full Fine-tuning Train ({folder_name})",
                linestyle="--",
            )
            ax.plot(
                epochs_eval_baseline,
                metrics["eval_loss"],
                label=f"Full Fine-tuning Eval ({folder_name})",
                linestyle="-",
            )

        ax.set_xlabel("Epochs", fontsize=14)
        ax.set_ylabel("Loss", fontsize=14)
        ax.set_title("Epochs vs Loss - Best LoRA vs Full Fine-tuning", fontsize=14)
        ax.tick_params(
            axis="both", which="major", labelsize=14
        )  # Increase tick label font size
        ax.legend(fontsize=14)
        ax.grid(True)
        plt.tight_layout()
        plt.savefig("./outputs/epochs_vs_loss.png", dpi=120)
        plt.close()

    # 3. Dice vs training time, models with same rank connected, full fine-tuning on its own
    if lora_results:
        fig, ax = plt.subplots(figsize=(12, 8))

        # Group by rank
        ranks = {}
        for folder_name, data in lora_results.items():
            rank = data["rank"]
            alpha = data["alpha"]
            metrics = data["metrics"]

            if rank not in ranks:
                ranks[rank] = {"dices": [], "times": [], "alphas": [], "folders": []}

            if metrics["eval_dice"] and metrics["training_time"]:
                final_dice = metrics["eval_dice"][-1]  # Last eval dice
                training_time = metrics["training_time"]

                ranks[rank]["dices"].append(final_dice)
                ranks[rank]["times"].append(training_time)
                ranks[rank]["alphas"].append(alpha)  # Store alpha for labeling
                ranks[rank]["folders"].append(folder_name)

        # Plot for each rank, without connecting the dots (as requested)
        for rank, data in ranks.items():
            ax.scatter(data["times"], data["dices"], label=f"r={rank}", s=80, zorder=5)

            # Add alpha value labels to each point
            for x, y, alpha_val in zip(data["times"], data["dices"], data["alphas"]):
                ax.annotate(
                    f"{alpha_val}",
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=14,
                    zorder=6,
                )

        # Plot baseline (full fine-tuning) separately
        for folder_name, metrics in baseline_results.items():
            if metrics["eval_dice"] and metrics["training_time"]:
                final_dice = metrics["eval_dice"][-1]
                training_time = metrics["training_time"]
                ax.scatter(
                    training_time,
                    final_dice,
                    label=f"Full Fine-tuning ({folder_name})",
                    marker="x",
                    s=200,
                    linewidth=3,
                )
                # No label for full fine-tuning as requested

        ax.set_xlabel("Training Time (seconds)", fontsize=14)
        ax.set_ylabel("Final Evaluation Dice", fontsize=14)
        ax.set_title("Dice vs Training Time", fontsize=14)
        ax.tick_params(
            axis="both", which="major", labelsize=14
        )  # Increase tick label font size
        ax.legend(fontsize=14)
        ax.grid(True)
        plt.tight_layout()
        plt.savefig("./outputs/dice_vs_training_time.png", dpi=120)
        plt.close()

    # 4. Epochs vs dice of best LoRA model vs full fine-tuning (no training dice)
    if best_lora_folder and baseline_results:
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot best LoRA model dice
        best_lora_metrics = lora_results[best_lora_folder]["metrics"]
        epochs_eval_lora = list(range(len(best_lora_metrics["eval_dice"])))

        ax.plot(
            epochs_eval_lora,
            best_lora_metrics["eval_dice"],
            label=f"Best LoRA Eval (r={lora_results[best_lora_folder]['rank']}, a={lora_results[best_lora_folder]['alpha']})",
            linestyle="-",
        )

        # Plot baseline (full fine-tuning) dice
        for folder_name, metrics in baseline_results.items():
            epochs_eval_baseline = list(range(len(metrics["eval_dice"])))

            ax.plot(
                epochs_eval_baseline,
                metrics["eval_dice"],
                label=f"Full Fine-tuning ({folder_name})",
                linestyle="-",
            )

        ax.set_xlabel("Epochs", fontsize=14)
        ax.set_ylabel("Dice", fontsize=14)
        ax.set_title("Epochs vs Dice - Best LoRA vs Full Fine-tuning", fontsize=14)
        ax.tick_params(
            axis="both", which="major", labelsize=14
        )  # Increase tick label font size
        ax.legend(fontsize=14)
        ax.grid(True)
        plt.tight_layout()
        plt.savefig("./outputs/epochs_vs_dice.png", dpi=120)
        plt.close()


def main():
    """Main function to extract metrics and create plots."""
    print("Extracting metrics from outputs directory...")

    # Extract all metrics
    metrics_dict = extract_all_metrics("./outputs")

    print(f"Extracted metrics for {len(metrics_dict)} folders:")
    for folder_name in metrics_dict.keys():
        print(f"  - {folder_name}")

    create_plots(metrics_dict)

    print("Done! Plots saved to outputs directory.")


if __name__ == "__main__":
    main()
