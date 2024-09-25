---
title: Benchmark Results
description: benchmark results page
hide_table_of_contents: true
---
# Benchmark Results 
**Performance of all attacks on benchmark settings.** Best Hamming weight (h) for secret recovered per setting/attack, time in hours needed to recover this secret, and machines used. Highest h per setting is **bold**. All Kyber secrets are binomial, and HE secrets are ternary. First three attacks (uSVP, SALSA, CC) solve Search-LWE; MITM* solves Decision LWE. "Total hrs" is total attack time assuming full parallelization.

Please see our [**paper**](https://eprint.iacr.org/2024/1229) for more detailed benchmark results and more information on how these results were obtained.

| **Attack**              | **Results**         | **n=256, k=2, logq=12**<br/>binomial | **n=256, k=2, logq=28**<br/>binomial | **n=256, k=3, logq=35**<br/>binomial | **n=1024, logq=26**<br/>ternary | **n=1024, logq=29**<br/>ternary | **n=1024, logq=50**<br/>ternary |
|-------------------------|---------------------|----------------------------------|----------------------------------|----------------------------------|-----------------------------|-----------------------------|-----------------------------|
| **uSVP**                | Best h              | -                                | -                                | -                                | -                           | -                           | -                           |
|                         | Recover hrs<br/>(1 CPU) | >1100                            | >1100                            | >1100                            | >1300                       | >1300                       | >1300                       |
| **SALSA**                  | Best h              | 9                                | 18                               | 16                               | 8                           | 10                          | 17                          |
|                         | Total hrs           | 36                               | 27                               | 39                               | 34.9                        | 49.4                        | 29.1                        |
| **CC**                  | Best h              | **11**                           | **25**                           | **19**                           | **12**                      | **12**                      | **20**                      |
|                         | Total hrs           | 28.1                             | 53                               | 34                               | 21.5                        | 31.7                        | 28                          |
| **MiTM<br/>(Decision LWE)** | Best h              | 4                                | 12                               | 14                               | 9                           | 9                           | 16                          |
|                         | Total hrs           | 0.7                              | 1.61                             | 29.4                             | 65                          | 13                          | 15.5                        |
