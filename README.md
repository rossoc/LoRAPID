<h1 style="text-align:center;"> SEG-PEFT </h1>

SEG-PEFT is a repository for medical image segmentation research, focused on 
polyp detection in endoscopic images. The project explores the effectiveness of 
LoRA (Low-Rank Adaptation) for medical image segmentation tasks, specifically 
comparing full fine-tuning vs. parameter-efficient fine-tuning approaches on the 
Kvasir-SEG dataset.

## Dataset

You shall download the dataset at the following link
[https://datasets.simula.no/downloads/kvasir-seg.zip](https://datasets.simula.no/downloads/kvasir-seg.zip) 

The Kvasir-SEG dataset is a medical image dataset specifically designed for polyp segmentation in endoscopic images. The dataset contains:

- **Images**: Endoscopic images of the GI tract containing polyps
- **Masks**: Binary segmentation masks where the polyp regions are marked
- **Size**: 1000 image-mask pairs for training and validation
- **Task**: Semantic segmentation to identify and segment polyp regions

To use the dataset with this project, you must place it in the following directory structure:
```
SEG-PEFT/
├── data/
│   └── Kvasir-SEG/
│       ├── images/
│       │   ├── cju0bygl98aar07367zog10cj.jpg
│       │   ├── cju0byglb8abs073685yd3kjn.jpg
│       │   └── ... (998 more image files)
│       └── masks/
│           ├── cju0bygl98aar07367zog10cj.jpg
│           ├── cju0byglb8abs073685yd3kjn.jpg
│           └── ... (998 more mask files)
```

The dataset is organized in two directories:
- `data/Kvasir-SEG/images/`: Contains the endoscopic images in RGB format
- `data/Kvasir-SEG/masks/`: Contains the corresponding binary masks where polyps are labeled as white (255) and background as black (0)

During training, the images are preprocessed with the following augmentations:
- Resize to 512x512 pixels
- Horizontal flipping (50% probability)
- Random brightness and contrast adjustments (20% probability)

## Scripts

### Initialization

Forst of all you must execute the following commmands to clone and install the
dependencies:

```sh
git clone https://github.com/rossoc/SEG-PEFT
cd SEG-PEFT
pip install -r requirements
```

Furthermore, you may want to activate the environment:

```sh
source .venv/bin/activate
```

### Training SegFormer without LoRA

To train the SegFormer model without LoRA (full fine-tuning):

```bash
python scripts/train_segformer.py --epochs 30 --lr 5e-5 --save-dir my_experiment
```

Additional options:
- `--epochs`: Number of training epochs (default: 30)
- `--lr`: Learning rate (default: 5e-5)
- `--save-dir`: Directory name to save the model (required)
- `--seed`: Random seed for reproducibility (default: 42)

### Training SegFormer with LoRA

To train the SegFormer model with LoRA (parameter-efficient fine-tuning):

```bash
python scripts/train_segformer_lora.py --epochs 30 --lr 5e-5 --rank 8 --lora-alpha 32 --lora-dropout 0.1 --save-dir my_lora_experiment
```

Additional options:
- `--epochs`: Number of training epochs (default: 30)
- `--lr`: Learning rate (default: 5e-5)
- `--rank`: LoRA rank (default: 8)
- `--lora-alpha`: LoRA alpha parameter (default: 32)
- `--lora-dropout`: LoRA dropout rate (default: 0.1)
- `--save-dir`: Directory name to save the model (required)
- `--seed`: Random seed for reproducibility (default: 42)
