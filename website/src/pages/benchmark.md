---
title: Benchmark Results
description: benchmark results page
hide_table_of_contents: true
---
# Benchmark Results 
|        Attack       |       Results       | Kyber MLWE Setting (n, k, log_2 q) |                |                | HE LWE Setting (n, log_2 q) |             |             |
|:-------------------:|:-------------------:|:----------------------------------:|:--------------:|:--------------:|:---------------------------:|:-----------:|:-----------:|
|                     |                     |            (256, 2, 12)            | (256, 2, 28)   | (256, 3, 35)   |          (1024, 26)         |  (1024, 29) |  (1024, 50) |
|         uSVP        | Best h              |                  -                 |       -        |       -        |              -              |      -      |      -      |
|                     | Recover hrs (1 CPU) |              &gt; 1100             |   &gt; 1100    |   &gt; 1300    |          &gt; 1300          |  &gt; 1300  |  &gt; 1300  |
|          ML         | Best h              |                  9                 |      18        |      16        |              8              |      10     |      17     |
|                     | Total hrs           |                 36                 |      27        |      39        |             34.9            |     49.4    |     29.1    |
|          CC         | Best h              |                 11                 |      25        |      19        |              12             |      12     |      20     |
|                     | Total hrs           |                28.1                |      53        |      34        |             21.5            |     31.7    |      28     |
| MiTM (Decision LWE) | Best h              |                  4                 |      12        |      14        |              9              |      9      |      16     |
|                     | Total hrs           |                 0.7                |     1.61       |     29.4       |              65             |      13     |     15.5    |