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
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation, TrainingArguments, Trainer
from datasets import Dataset as HFDataset
from peft import get_peft_model, LoraConfig, TaskType
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
        
        # Apply transforms if available
        if self.transforms:
            # Convert to numpy arrays for albumentations
            image_np = np.array(image)
            mask_np = np.array(mask)
            
            # Apply augmentation
            augmented = self.transforms(image=image_np, mask=mask_np)
            image_np = augmented['image']
            mask_np = augmented['mask']
            
            # Convert back to PIL Images
            image = Image.fromarray(image_np)
            mask = Image.fromarray(mask_np)
        
        # Prepare for the model
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values.squeeze()  # Remove batch dimension added by feature extractor
        
        # Process the mask
        mask = np.array(mask)
        mask = np.where(mask > 127, 1, 0)  # Normalize mask to binary (0, 1) values
        mask = torch.from_numpy(mask).long()
        
        return {
            "pixel_values": pixel_values,
            "labels": mask
        }


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Configuration
    data_dir = "data/Kvasir-SEG"
    image_dir = os.path.join(data_dir, "images")
    mask_dir = os.path.join(data_dir, "masks")
    model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
    
    # Initialize feature extractor
    feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name)
    
    # Define data transforms
    transforms = A.Compose([
        A.Resize(512, 512),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    # Create dataset
    dataset = KvasirSegDataset(image_dir, mask_dir, feature_extractor, transforms)
    
    # Calculate train/val split
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    # Split the dataset
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    print(f"Total samples: {total_size}")
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")
    
    # Load model
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_name,
        num_labels=2,  # Background and polyp
        id2label={0: "background", 1: "polyp"},
        label2id={"background": 0, "polyp": 1}
    )
    
    # Apply LoRA
    peft_config = LoraConfig(
        task_type=TaskType.SEMANTIC_SEGMENTATION,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=[
            "query",
            "key",
            "value",
            "attention.output.dense",
            "mlp.dense.1",
            "mlp.dense.2",
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
            
            # Resize predictions to match labels
            logits_tensor = nn.functional.interpolate(
                logits_tensor,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            
            # Convert to predictions
            pred_labels = logits_tensor.argmax(dim=1)
            
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
        num_train_epochs=10,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        save_steps=100,
        eval_steps=100,
        logging_steps=10,
        learning_rate=5e-5,
        save_total_limit=2,
        prediction_loss_only=True,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="tensorboard",
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Save the model
    trainer.save_model("./outputs/segformer-kvasir-final")
    print("Model saved to ./outputs/segformer-kvasir-final")


if __name__ == "__main__":
    main()