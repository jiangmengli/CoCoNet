# CoCoNet

### Install the PyTorch based environment for CoCoNet. 
```bash
# Create a conda environment
conda create -n coconet python=3.7

# Activate the environment
conda activate coconet

# Install dependencies
pip install -r requirements.txt
```

### Install the datasets.
The used datasets are totally established, please follow the official instruction to install the datasets. Note that our code requires the datasets containing the "images" not "features".

## How to Run

We provide the running scripts as follows.
```bash
# Train
python train_CoCoNet.py

# Test
python LinearProbing.py
```

Readers can change hyperparameters directly in code or bash script
