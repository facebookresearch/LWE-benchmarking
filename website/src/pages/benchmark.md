---
title: Benchmark Results
description: benchmark results page
hide_table_of_contents: true
---
# Benchmark Results 
 Performance of all attacks on benchmark settings. Best Hamming weight (h) for secret recovered per setting/attack, time in hours needed to recover this secret, and machines used. Highest h per setting is **bold**. All Kyber secrets are binomial, and HE secrets are ternary. First three attacks (uSVP, ML, CC) are Search-LWE; MITM* is Decision LWE. The ML, CC, and MiTM attacks have two phases: Preprocessing (Prepoc. in table), when LWE data is reduced and/or short vectors are obtained; Recovery (Recover in table) for ML/CC, when reduced vectors are used recover secrets; and Decide for MiTM, when Decision LWE is solved using short vectors. We report time separately for each step. When steps can be parallelized, we report hours/machine and number of machines. The uSVP attack has only the "recover" phase, which cannot be parallelized. "Total hrs" is total attack time assuming full parallelization.

| **Attack**              | **Results**         | **n=256, k=2, logq=12** binomial | **n=256, k=2, logq=28** binomial | **n=256, k=3, logq=35** binomial | **n=1024, logq=26** ternary | **n=1024, logq=29** ternary | **n=1024, logq=50** ternary |
|-------------------------|---------------------|----------------------------------|----------------------------------|----------------------------------|-----------------------------|-----------------------------|-----------------------------|
| **uSVP**                | Best h              | -                                | -                                | -                                | -                           | -                           | -                           |
|                         | Recover hrs (1 CPU) | >1100                            | >1100                            | >1100                            | >1300                       | >1300                       | >1300                       |
| **ML**                  | Best h              | 9                                | 18                               | 16                               | 8                           | 10                          | 17                          |
|                         | Total hrs           | 36                               | 27                               | 39                               | 34.9                        | 49.4                        | 29.1                        |
| **CC**                  | Best h              | **11**                           | **25**                           | **19**                           | **12**                      | **12**                      | **20**                      |
|                         | Total hrs           | 28.1                             | 53                               | 34                               | 21.5                        | 31.7                        | 28                          |
| **MiTM (Decision LWE)** | Best h              | 4                                | 12                               | 14                               | 9                           | 9                           | 16                          |
|                         | Total hrs           | 0.7                              | 1.61                             | 29.4                             | 65                          | 13                          | 15.5                        |
