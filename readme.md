# CCTN: CascadeTableNet for intflex Drawing Table Detection

## Overview

## Version Requirements
- Python 3.8.20
- PyTorch 1.13.1
- CUDA 11.7
- MMDetection 2.28.1
- MMCV 1.7.0

## Installation

### 1. Environment Setup
```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate cctn-v2
```

### 2. MMDetection Installation
```bash
# Clone specific version of MMDetection
git clone --branch v2.28.1 https://github.com/open-mmlab/mmdetection.git
cd mmdetection

# Install dependencies
pip install -r "./requirements/optional.txt"
python setup.py install
python setup.py develop
pip install -r "requirements.txt"
```

### 3. Additional Dependencies
```bash
# Clone CascadeTabNet project
git clone https://github.com/DevashishPrasad/CascadeTabNet.git
```

### 4. Model Preparation
```bash
# Download the pretrained model
gdown --folder https://drive.google.com/drive/folders/1iBZ_3owTzg4_q3luxaJ1EsWsiIMV3rqm?usp=drive_link

# Create working directory
mkdir CCTN_Finetune
cd CCTN_Finetune

# If you choose to build from scratch, first you need to upgrade model from original CascadeTabNet
python ./CascadeTabNet/Tools/upgrade_model_version.py ./checkpoints/epoch_36.pth ./checkpoints/epoch_36_v2.pth --num-classes 81

# Alternatively, you can choose: epoch 13 (the best map result), or epoch 20 (the latest result)
```

### 5. Train the model with tools/train_cctn.py and run the prediction with CCTN_Demo.ipynb

## Project Structure (Dataset is in Coco mmdetection format)
```
- CCTN_Finetune/  
  - CCTN_dataset/       
    - annotations/
      - train.json
      - valid.json
      - test.json
    train/
    valid/
    test/
  - checkpoints/         
  - configs/            
  - imgs/                   
  - tools/              
  - CCTN_Demo.ipynb    
  - environment.yml    

```

## Training Tricks

### Optimizer Configuration
```python
optimizer = dict(
    type='AdamW',                # Use AdamW optimizer
    lr=5e-5,                     # Small learning rate
    weight_decay=0.0001,         # Weight decay
    eps=1e-8,                    # AdamW parameter
    betas=(0.9, 0.999),         # AdamW parameter
    paramwise_cfg=dict(
        custom_keys={
            'roi_head.bbox_head': dict(lr_mult=10),  # Higher learning rate for ROI head
            'neck': dict(lr_mult=5)                  # Medium learning rate for neck
        }))
```

### Training Strategy

If the data is not annotated with the seg gt, the mask part of the model config needs to be deleted

1. Learning Rate Warmup
   - Linear warmup strategy
   - Warmup iterations: 50 (For a dataset with 50 images this is 1 epoch)
   - Warmup ratio: 1/3

2. Evaluation and Saving Strategy
   - Evaluate model every epoch
   - Save best mAP model

3. Validated Effective Techniques:
   - Small batch training (batch_size = 1 or 2)
   - Differential learning rates for models's backbone, ROI head and neck (or you can freeze the backbone)
   - Regular checkpoint saving (every 2 epochs)

## Model Performance
The best test set results was achieved in epoch 13:
- mAP@0.5: 0.911
- mAP@0.75: 0.820
- mAP@0.5:0.95: 0.749

