---
title: Getting Started
description: getting started page
hide_table_of_contents: false
sidebar_position: 1
---
# Getting Started

## Requirements

* A computer running macOS or Linux
* At least one NVIDIA GPU
* Python version >= 3.10.12

## Installation and Setup
Clone this repository to your machine that has at least one GPU.

Install flatter by following the instructions on [Flatter Github](https://github.com/keeganryan/flatter)

#### Option 1: Install with Pip [Doesn't have fpyLLL]
Create a virtual environment via the following (first make sure you have the virtualenv): 
```
python3 -m venv ~/.lattice_venv
source ~/.lattice_env/bin/activate
pip install -r environment/requirements.txt
```
#### Option 2: Install with Conda
Install a conda environment with the command:
```
conda env create -f environment/exported_env.yml
conda activate lattice_env
```

## Quickstart
### Running the SALSA Attack
Run the following scripts:
* `preprocess.py`: Runs the preprocessing step to prepare the reduction matrix R
* ```generate_secrets.py```: Generates (RA, Rb) pairs and associated secrets.
* ```train_and_recover.py```: Runs the transformer-based secret recovery attack (encoder-only model by default)
  
Example run commands:

`python preprocess.py --N 256 --Q 3329 --dump_path /checkpoint/ewenger/data/debug --exp_name R_A_256_12_omega10_debug --num_workers 5 --reload_data /checkpoint/ewenger/dumped/orig_A/n256_logq12/tiny_A.npy --thresholds "0.72,0.725,0.9" --lll_penalty 10`

`python generate_secrets.py --processed_dump_path /checkpoint/ewenger/data/debug/R_A_256_12_omega10_debug/ --exp_name demo --dump_path /checkpoint/ewenger/data/ --secret_type binary --num_secret_seeds 10 --min_hamming 4 --max_hamming 20 --max_samples 1000000`

`CUDA_VISIBLE_DEVICES=1 python train_and_recover.py --data_path /checkpoint/ewenger/data/demo/A_b_256_12_omega4/binary_4_20/ --secret_seed 1 --hamming 20 --train_batch_size 768 --val_batch_size 1568 --n_enc_layers 4 --enc_emb_dim 512 --n_enc_heads 4 --angular_emb true --learning_rate 1e-5`

### Running the Cruel and Cool Attack
Run the following scripts:
* `preprocess.py`: Runs the preprocessing step to prepare the reduction matrix R
* ```generate_secrets.py```: Generates (RA, Rb) pairs and associated secrets.
* ```cruel.py```: Runs the Cruel and Cool attack.
  
Example run commands:

`python preprocess.py --N 256 --Q 3329 --dump_path /checkpoint/ewenger/data/debug --exp_name R_A_256_12_omega10_debug --num_workers 5 --reload_data /checkpoint/ewenger/dumped/orig_A/n256_logq12/tiny_A.npy --thresholds "0.72,0.725,0.9" --lll_penalty 10`

`python generate_secrets.py --processed_dump_path /checkpoint/ewenger/data/debug/R_A_256_12_omega10_debug/ --exp_name demo --dump_path /checkpoint/ewenger/data/ --secret_type binary --num_secret_seeds 10 --min_hamming 4 --max_hamming 20 --max_samples 1000000`

`CUDA_VISIBLE_DEVICES=1 python cruel.py --TODO`

### Running the USVP Attack
Run the following script:
* `usvp.py`: Runs the USVP attack on a randomly generated LWE matrix
  
Example run commands:

`python usvp.py --N 256 --Q 3329 --algo flatter --secret_path /checkpoint/ewenger/data/demo/A_b_256_12_omega4/binary_4_20/ --hamming 10`

### Running the MITM Attack
TODO

## Citation
TODO

## License
This code is made available under CC-by-NC, however you may have other legal obligations that govern your use of other content, such as the terms of service for third-party models.
