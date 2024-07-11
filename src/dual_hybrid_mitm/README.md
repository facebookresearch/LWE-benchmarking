Commands to test this (first make sure you have a conda environment with sage installed - can use /private/home/ewenger/.conda/envs/sage for now): 

```python3 src/dual_hybrid_mitm/dual_hybrid_mitm.py --dump_path ./ --exp_name dbug_mitm --k 30 --N 45 --Q 11197 --hamming 10 --exp_id mitm_binomial_test --num_workers 10 --step reduce --tau 30 --mlwe_k 3```

Then run 
```python3 src/dual_hybrid_mitm/dual_hybrid_mitm.py --step mitm --short_vectors_path ./dbug_mitm/mitm_binomial_test/ --secret_seed 2 --bound 100 --secret_path /checkpoint/ewenger/sp_paper/usvp_secrets/usvp_N45_binomial_10_20 --hamming 10 --gamma 2```

Whole experiment should take about 2 minutes. 