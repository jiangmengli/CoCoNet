# CoCoNet

This paper ''Modeling Multiple Views via Implicitly Preserving Global Consistency and Local Complementarity'' is accepted by IEEE TKDE in 2022.

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

## Citation

If you find this repo useful for your research, please consider citing the paper
```
@article{coconet2022jml,
  author    = {Jiangmeng Li and
               Wenwen Qiang and
               Changwen Zheng and
               Bing Su and
               Farid Razzak and
               Ji-Rong Wen and
               Hui Xiong},
  title     = {Modeling Multiple Views via Implicitly Preserving Global Consistency and Local Complementarity},
  journal   = {CoRR}
  year      = {2022},
  eprinttype = {arXiv}
}
```
