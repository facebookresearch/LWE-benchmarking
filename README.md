# README
 
## Requirements

* A computer running macOS or Linux
* At least one NVIDIA GPU
* Python version >= 3.10.12

## Installation and Setup
Clone this repository to your machine that has at least one GPU.

Install flatter by following the instructions on [Flatter Github](https://github.com/keeganryan/flatter). Make sure it is symlinked correctly so it runs on your machine when you type the command "flatter". 

Install a conda environment with the command:
```
conda env create -f environment/environment.yml
conda activate lattice_env
```

## Quickstart

### (TODO UPDATE) Using provided data for SALSA and CC attacks 

For all attacks, we have provided the LWE secrets used in the benchmark results of our paper: see data/benchmark_paper_data/{setting}/{secret type}_secrets_h{min}_{max}/secret.npy. We highly recommend using these secrets to test new attacks on these same settings, since this ensures comparable results. In the sections below, we provide instructions on which flags to use in the attack scripts if you want to run them on these provided secrets (rather than generating new ones on the fly). 

For the SALSA and CC attacks, the preprocessing step is time and resource-intensive. Hence, we have provided original LWE samples and preprocessed datasets for each of the 6 benchmark settings proposed in our paper. Each has been compressed. They are available at the following links:
- (Toy) $n=80$, $log_2 q = 7$: (140MB compressed --> 1 GB uncompressed): https://dl.fbaipublicfiles.com/lwe-benchmarking/80_7_omega15_lwe_data_prefix.tar.gz
- (Kyber) $n=256, k=2,log_2 q = 12$: 
- (Kyber) $n=256, k=2,log_2 q = 28$: 
- (Kyber) $n=256, k=3, log_2 q = 35$: 
- (HE) $n=1024, log_2 q = 26$ (5GB compressed --> 21GB uncompressed): https://dl.fbaipublicfiles.com/1024_26_omega10_rlwe_data_prefix.tar.gz
- (HE) $n=1024, log_2 q = 29$ (6 GB compressed --> 24 GB uncompressed): https://dl.fbaipublicfiles.com/1024_29_omega10_rlwe_data_prefix.tar.gz
- (HE) $n=1024, log_2 q = 50$ (18GB compressed --> 46GB uncompressed): https://dl.fbaipublicfiles.com/1024_50_omega10_rlwe_data_prefix.tar.gz

When downloading these, make sure you place them in a place with enough storage. You can then tar -xvf and gunzip them to restore them to original format. If you want our attack pipeline to flow smoothly, you should either place these files directly in ./data/benchmark_paper_data/{appropriate folder} or symlink them to that directory. 

To create the full set of reduced LWE (A,b) pairs using the provided secrets and preprocessed data, DO X TODO TBD.

### Running the SALSA Attack
Run the following scripts:
* `preprocess.py`: Runs the preprocessing step to prepare the reduction matrix R
* ```generate_secrets.py```: Generates (RA, Rb) pairs and associated secrets.
* ```train_and_recover.py```: Runs the transformer-based secret recovery attack (encoder-only model by default)
  
Example run commands:

`python3 preprocess.py --N 256 --Q 3329 --dump_path /checkpoint/ewenger/data/debug --exp_name R_A_256_12_omega10_debug --num_workers 5 --reload_data /checkpoint/ewenger/dumped/orig_A/n256_logq12/tiny_A.npy --thresholds "0.72,0.725,0.9" --lll_penalty 10`

`python3 generate_secrets.py --processed_dump_path /checkpoint/ewenger/data/debug/R_A_256_12_omega10_debug/ --exp_name demo --dump_path /checkpoint/ewenger/data/ --secret_type binary --num_secret_seeds 10 --min_hamming 4 --max_hamming 20 --max_samples 1000000`

`python src/salsa/train_and_recover.py --data_path /checkpoint/eshika/data/sp_paper/training_data/A_b_1024_26_omega10_rlwe/ternary_8_9/ --secret_seed 0 --rlwe 1 --task mlwe-i --angular_emb true --dxdistinguisher true --hamming 8 --cruel_bits 750 --train_batch_size 64 --val_batch_size 128 --n_enc_heads 8 --n_enc_layers 4 --enc_emb_dim 512 --base 25325715601611 --bucket_size 5065143120`

### Running the Cruel and Cool Attack
To run everything, you need
- `flatter` (only if you want to actually use that, can also reduce with BKZ)
- `fpylll`
- `pytorch` (nightly, unless you disable compilation with --compile_bf 0)
- `numpy`
- `tqdm`

Run the following scripts:
* `preprocess.py`: Runs the preprocessing step to prepare the reduction matrix R
* ```generate_secrets.py```: Generates (RA, Rb) pairs and associated secrets.
* ```cruel_cool/main.py```: Runs the Cruel and Cool attack.
  
Example run commands:

`python3 preprocess.py --N 256 --Q 3329 --dump_path /checkpoint/ewenger/data/debug --exp_name R_A_256_12_omega10_debug --num_workers 5 --reload_data /checkpoint/ewenger/dumped/orig_A/n256_logq12/tiny_A.npy --thresholds "0.72,0.725,0.9" --lll_penalty 10`

`python3 generate_secrets.py --processed_dump_path /checkpoint/ewenger/data/debug/R_A_256_12_omega10_debug/ --exp_name demo --dump_path /checkpoint/ewenger/data/ --secret_type binary --num_secret_seeds 10 --min_hamming 4 --max_hamming 20 --max_samples 1000000`

`python3 src/cruel_cool/main.py --path /checkpoint/eshika/data/sp_paper/training_data/A_b_1024_26_omega10_rlwe/ternary_8_9/ --exp_name demo --greedy_max_data 100000 --keep_n_tops 1 --batch_size 10000 --compile_bf 0 --mlwe_k 1 --secret_window -1  --full_hw 9 --secret_type ternary --bf_dim 750 --min_bf_hw 1 --max_bf_hw 5 --seed 0`

### Running the USVP Attack
First, generate a secret to use in the test attack via the command:
`python3 src/generate/generate_secrets.py --N 32 --secret_type binary --min_hamming 5 --max_hamming 10 --dump_path ./data --processed_dump_path ./ --actions only_secrets`

Run the following script:
* `usvp/usvp.py`: Runs the USVP attack on a randomly generated LWE matrix
  
Example run commands:

`python3 src/usvp/usvp.py --N 32 --Q 967 --algo BKZ2.0 --secret_path ./data/secrets/secret_N32_binary_5_10/ --hamming 6 --secret_type binary`

### Running the MITM Attack
Generate a secret, as described in the uSVP section above.

Example commands:
Then run
```python3 src/dual_hybrid_mitm/dual_hybrid_mitm.py --dump_path ./ --exp_name test_mitm --k 16 --N 32 --Q 11197 --hamming 6 --exp_id mitm_binomial_test --num_workers 10 --step reduce --tau 30 --secret_seed 2 --secret_path ./data/secrets/secret_N32_binary_5_10/```

Then run 
```python3 src/dual_hybrid_mitm/dual_hybrid_mitm.py --step mitm --secret_seed 2 --bound 100 --secret_path ./data/secrets/secret_N32_binary_5_10/ --hamming 6 --short_vectors_path ./test_mitm/mitm_binomial_test/ --secret_type binary``` (change short vectors path if you modified dump path, exp_name, or exp_id in the first command)

Whole experiment should take about 2 minutes. 

## Benchmark Results

<body>
<table class="tg" style="undefined;table-layout: fixed; width: 1427px"><colgroup>
<col style="width: 156px">
<col style="width: 178px">
<col style="width: 280px">
<col style="width: 142px">
<col style="width: 143px">
<col style="width: 221px">
<col style="width: 153px">
<col style="width: 154px">
</colgroup>
<thead>
  <tr>
    <th class="tg-v0hj" rowspan="2">Attack</th>
    <th class="tg-v0hj" rowspan="2">Results</th>
    <th class="tg-v0hj" colspan="3">Kyber MLWE Setting (n, k, log<sub>2</sub> q)</th>
    <th class="tg-v0hj" colspan="3">HE LWE Setting (n, log<sub>2</sub> q)</th>
  </tr>
  <tr>
    <th class="tg-ezme">(256, 2, 12)</th>
    <th class="tg-ezme">(256, 2, 28)</th>
    <th class="tg-ezme">(256, 3, 35)</th>
    <th class="tg-ezme">(1024, 26)</th>
    <th class="tg-ezme">(1024, 29)</th>
    <th class="tg-ezme">(1024, 50)</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-3xi5" rowspan="2">uSVP</td>
    <td class="tg-ycr8">Best h</td>
    <td class="tg-3xi5">-</td>
    <td class="tg-3xi5">-</td>
    <td class="tg-3xi5">-</td>
    <td class="tg-3xi5">-</td>
    <td class="tg-3xi5">-</td>
    <td class="tg-3xi5">-</td>
  </tr>
  <tr>
    <td class="tg-gaf0">Recover hrs (1 CPU)</td>
    <td class="tg-5986">&gt; 1100</td>
    <td class="tg-5986">&gt; 1100</td>
    <td class="tg-5986">&gt; 1300</td>
    <td class="tg-pury"><span style="font-weight:400;font-style:normal">&gt; 1300</span></td>
    <td class="tg-5986"><span style="font-weight:400;font-style:normal">&gt; 1300</span></td>
    <td class="tg-5986"><span style="font-weight:400;font-style:normal">&gt; 1300</span></td>
  </tr>
  <tr>
    <td class="tg-yj5y" rowspan="4">ML</td>
    <td class="tg-y698">Best h</td>
    <td class="tg-yj5y">9</td>
    <td class="tg-yj5y">18</td>
    <td class="tg-yj5y">16</td>
    <td class="tg-yj5y">8</td>
    <td class="tg-yj5y">10</td>
    <td class="tg-yj5y">17</td>
  </tr>
  <tr>
    <td class="tg-xq07">Preproc. hrs * CPUs</td>
    <td class="tg-wc2v">28 * 3216</td>
    <td class="tg-wc2v">11 * 3010</td>
    <td class="tg-wc2v">33 * 1843</td>
    <td class="tg-wc2v">21.5 * 1160</td>
    <td class="tg-wc2v">31.6 * 1164</td>
    <td class="tg-wc2v">23.8 * 1284</td>
  </tr>
  <tr>
    <td class="tg-y698">Recover hrs * GPUs</td>
    <td class="tg-yj5y">8 * 256</td>
    <td class="tg-yj5y">16 * 256</td>
    <td class="tg-yj5y">6 * 256</td>
    <td class="tg-yj5y">13.4 *1024</td>
    <td class="tg-yj5y">17.8 *1024</td>
    <td class="tg-yj5y">5.3 * 1024</td>
  </tr>
  <tr>
    <td class="tg-xq07">Total hrs</td>
    <td class="tg-wc2v">36</td>
    <td class="tg-wc2v">27</td>
    <td class="tg-wc2v">39</td>
    <td class="tg-wc2v">34.9</td>
    <td class="tg-wc2v">49.4</td>
    <td class="tg-wc2v">29.1</td>
  </tr>
  <tr>
    <td class="tg-3xi5" rowspan="4">CC</td>
    <td class="tg-c6of">Best h</td>
    <td class="tg-3xi5"><span style="font-weight:bold">11</span></td>
    <td class="tg-3xi5"><span style="font-weight:bold">25</span></td>
    <td class="tg-3xi5"><span style="font-weight:bold">19</span></td>
    <td class="tg-3xi5"><span style="font-weight:bold">12</span></td>
    <td class="tg-3xi5"><span style="font-weight:bold">12</span></td>
    <td class="tg-3xi5"><span style="font-weight:bold">20</span></td>
  </tr>
  <tr>
    <td class="tg-gaf0">Preproc. hrs * CPUs</td>
    <td class="tg-5986">28 * 161</td>
    <td class="tg-5986">11 * 151</td>
    <td class="tg-5986">23 * 92</td>
    <td class="tg-5986">21.5 * 58</td>
    <td class="tg-5986">31.6 * 58</td>
    <td class="tg-5986">23.8 * 64</td>
  </tr>
  <tr>
    <td class="tg-c6of">Recover hrs * GPUs</td>
    <td class="tg-3xi5">0.1 * 256</td>
    <td class="tg-3xi5">42 * 256</td>
    <td class="tg-3xi5">0.9 * 256</td>
    <td class="tg-3xi5">0.04 * 1024</td>
    <td class="tg-3xi5">0.1 * 1024</td>
    <td class="tg-3xi5">4.2 * 1024</td>
  </tr>
  <tr>
    <td class="tg-gaf0">Total hrs</td>
    <td class="tg-5986">28.1</td>
    <td class="tg-5986">53</td>
    <td class="tg-5986">34</td>
    <td class="tg-5986">21.5</td>
    <td class="tg-5986">31.7</td>
    <td class="tg-5986">28</td>
  </tr>
  <tr>
    <td class="tg-yj5y" rowspan="4">MiTM (Decision LWE)</td>
    <td class="tg-y698">Best h</td>
    <td class="tg-yj5y">4</td>
    <td class="tg-yj5y">12</td>
    <td class="tg-yj5y">14</td>
    <td class="tg-yj5y">9</td>
    <td class="tg-yj5y">9</td>
    <td class="tg-yj5y">16</td>
  </tr>
  <tr>
    <td class="tg-xq07">Preproc. hrs * CPUs</td>
    <td class="tg-wc2v">0.5 * 50</td>
    <td class="tg-wc2v">1.6 * 50</td>
    <td class="tg-wc2v">4.4 * 50</td>
    <td class="tg-wc2v">8 * 50</td>
    <td class="tg-wc2v">11.4 * 50</td>
    <td class="tg-wc2v">14.4 * 50</td>
  </tr>
  <tr>
    <td class="tg-y698">Decide hrs (1 CPU)</td>
    <td class="tg-yj5y">0.2</td>
    <td class="tg-yj5y">0.01</td>
    <td class="tg-yj5y">25</td>
    <td class="tg-yj5y">57</td>
    <td class="tg-yj5y">2</td>
    <td class="tg-yj5y">1.1</td>
  </tr>
  <tr>
    <td class="tg-xq07">Total hrs</td>
    <td class="tg-wc2v">0.7</td>
    <td class="tg-wc2v">1.61</td>
    <td class="tg-wc2v">29.4</td>
    <td class="tg-wc2v">65</td>
    <td class="tg-wc2v">13</td> 
    <td class="tg-wc2v">15.5</td>
  </tr>
</tbody>
</table>
</body>

## Citation
TODO

## License
This code is made available under CC-by-NC, however you may have other legal obligations that govern your use of other content, such as the terms of service for third-party models.
