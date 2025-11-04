#!/usr/bin/env python
"""
Training script for SegFormer model on Kvasir-SEG dataset.
This script trains a segmentation model to identify polyps in medical images.
"""

import os
import sys

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
    TrainingArguments,
    Trainer,
)
from peft import get_peft_model, LoraConfig
import evaluate
import albumentations as A
from albumentations.pytorch import ToTensorV2


# Define the Kvasir-SEG dataset class
class KvasirSegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, feature_extractor, transforms=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.feature_extractor = feature_extractor
        self.transforms = transforms
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

        # Validate that images and masks match
        assert len(self.images) == len(self.masks)
        for img, mask in zip(self.images, self.masks):
            assert img == mask, f"Image {img} does not match mask {mask}"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image and mask
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Grayscale mask

        # Apply transforms if available - this handles padding to ensure consistent dimensions
        if self.transforms:
            # Convert to numpy arrays for albumentations
            image_np = np.array(image)
            mask_np = np.array(mask)

            # Apply augmentation (this handles resizing and padding consistently)
            augmented = self.transforms(image=image_np, mask=mask_np)
            # After albumentations with ToTensorV2, image and mask are tensors
            image_tensor = augmented["image"]
            mask_tensor = augmented["mask"]
        else:
            # Without transforms, manually pad to 512x512
            # First resize to fit within 512x512 maintaining aspect ratio
            image.thumbnail((512, 512), Image.Resampling.LANCZOS)
            mask.thumbnail((512, 512), Image.Resampling.NEAREST)
            
            # Then pad to exactly 512x512
            image_np = np.array(image)
            mask_np = np.array(mask)
            
            # Pad if needed
            h, w = image_np.shape[:2]
            pad_h = 512 - h
            pad_w = 512 - w
            
            if len(image_np.shape) == 3:  # RGB
                image_np = np.pad(image_np, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
            else:  # Grayscale image
                image_np = np.pad(image_np, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
            
            if len(mask_np.shape) == 3:  # Shouldn't happen for grayscale but just in case
                mask_np = np.pad(mask_np, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
            else:  # Grayscale mask
                mask_np = np.pad(mask_np, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
            
            # Convert to tensor
            image_tensor = torch.tensor(image_np).permute(2, 0, 1).float() if len(image_np.shape) == 3 else torch.tensor(image_np).unsqueeze(0).float()
            mask_tensor = torch.tensor(mask_np)

        # Apply feature extractor to process the image (handles normalization)
        # Convert tensor back to numpy for feature extractor if needed
        if isinstance(image_tensor, torch.Tensor):
            # Make sure it's the right shape for the processor
            if image_tensor.shape[0] == 3:  # If CHW format
                image_np = image_tensor.permute(1, 2, 0).numpy()  # Convert to HWC format
            else:
                image_np = image_tensor.squeeze().numpy()  # For single channel

        inputs = self.feature_extractor(images=[image_np], return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)  # Remove batch dimension

        # Process the mask - ensure it's binary (0, 1) and has correct shape
        if isinstance(mask_tensor, torch.Tensor):
            mask_np = mask_tensor.numpy()
        else:
            mask_np = mask_tensor
            
        mask_np = np.where(mask_np > 127, 1, 0)  # Normalize mask to binary (0, 1) values
        mask_tensor = torch.from_numpy(mask_np).long()

        # Only return the keys that SegFormer expects
        return {"pixel_values": pixel_values, "labels": mask_tensor}


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Configuration
    data_dir = "data/Kvasir-SEG"
    image_dir = os.path.join(data_dir, "images")
    mask_dir = os.path.join(data_dir, "masks")
    model_name = "nvidia/segformer-b0-finetuned-ade-512-512"

    # Initialize image processor
    feature_extractor = SegformerImageProcessor.from_pretrained(
        model_name,
        do_reduce_labels=False,  # Explicitly set to avoid warnings about ignored arguments
        size={"height": 512, "width": 512}  # Explicitly set resize dimensions
    )

    # Define data transforms - pad to ensure consistent dimensions
    # We'll resize to fit within 512x512 and pad to exactly 512x512
    transforms = A.Compose(
        [
            A.LongestMaxSize(max_size=512),
            A.PadIfNeeded(min_height=512, min_width=512, border_mode=0, value=(0, 0, 0), position="center"),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            ToTensorV2(),
        ]
    )

    # Create dataset
    dataset = KvasirSegDataset(image_dir, mask_dir, feature_extractor, transforms)

    # For faster testing, use a smaller subset of the data
    total_size = len(dataset)
    # Limit dataset to 10% of total for faster testing
    subset_size = min(total_size, 20)  # Use at most 20 samples for testing
    indices = list(range(subset_size))
    subset_dataset = torch.utils.data.Subset(dataset, indices)
    
    # Calculate train/val split on the subset
    train_size = int(0.8 * subset_size)
    val_size = subset_size - train_size

    # Split the subset dataset
    train_dataset, val_dataset = torch.utils.data.random_split(
        subset_dataset, [train_size, val_size]
    )

    print(f"Total samples: {total_size}")
    print(f"Using subset: {subset_size} samples")
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")

    # Load model
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_name,
        num_labels=2,  # Background and polyp
        id2label={0: "background", 1: "polyp"},
        label2id={"background": 0, "polyp": 1},
        ignore_mismatched_sizes=True,
    )

    # Apply LoRA
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=[
            "query",
            "key",
            "value",
        ],
    )

    model = get_peft_model(model, peft_config)
    print(f"Model parameters: {model.num_parameters()}")
    print(f"Trainable parameters: {model.num_parameters(only_trainable=True)}")

    # Define metrics
    metric = evaluate.load("mean_iou")

    def compute_metrics(eval_pred):
        with torch.no_grad():
            logits, labels = eval_pred
            logits_tensor = torch.tensor(logits)

            # Ensure labels are the expected shape [batch, height, width]
            if len(labels.shape) == 4 and labels.shape[1] == 1:  # [batch, 1, height, width]
                labels = labels.squeeze(1)  # Remove the channel dimension to get [batch, height, width]

            # Resize predictions to match labels
            # The logits shape should be [batch, num_classes, height, width]
            # and labels should be [batch, height, width]
            target_size = labels.shape[-2:]  # (height, width)
            resized_logits = nn.functional.interpolate(
                logits_tensor,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )

            # Convert to predictions
            pred_labels = resized_logits.argmax(dim=1)

            # Compute metrics
            results = metric.compute(
                predictions=pred_labels.int(),
                references=labels.int(),
                num_labels=2,
                ignore_index=255,
                reduce_labels=False,
            )

            return {
                "mean_iou": results["mean_iou"],
                "mean_f1": results["mean_f1_score"],
                "mean_accuracy": results["mean_accuracy"],
                "per_class_iou": results["per_class_iou"],
                "per_class_f1": results["per_class_f1_scores"],
            }

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./outputs/segformer-kvasir",
        num_train_epochs=2,  # Reduced for faster testing
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        save_steps=40,  # Changed to be multiple of eval_steps
        eval_steps=20,  # Reduced for faster evaluation
        logging_steps=5,  # Reduced
        learning_rate=5e-5,
        save_total_limit=2,
        prediction_loss_only=True,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to=None,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        dataloader_pin_memory=False,  # Disable pin_memory to avoid warning
    )

    # Define a simple data collator
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.stack([example["labels"] for example in examples])
        
        # Ensure tensors are contiguous in memory to avoid stride issues
        pixel_values = pixel_values.contiguous()
        labels = labels.contiguous()
        
        return {"pixel_values": pixel_values, "labels": labels}

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )

    # Train the model
    print("Starting training...")
    trainer.train()

    # Save the model
    trainer.save_model("./outputs/segformer-kvasir-final")
    print("Model saved to ./outputs/segformer-kvasir-final")


if __name__ == "__main__":
    main()
