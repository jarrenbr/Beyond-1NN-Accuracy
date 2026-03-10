## Beyond 1-NN Accuracy

We provide this repository for implementation transparency. We intend to help readers to understand low-level details, such as edge cases. 

`diagnostics.py` is the machinery that we use to output `results/all_time_series_scores.csv`. With this output (provided), `reproduce_figs.py` reproduces all figures in our paper. `diagnostics.py` does not run as-is, the substantial amount of preprocessing code is not yet ported over. However, the data we use is all public. `reproduce_figs.py` will run with Python 3.13; install the project with 

```pip install .```

Some low-level details:
- We apply the exact Mann-Whitney U test for partitions with less than 30 samples.
- For undefined MCC values, we set the score to:
  - -1 if all predictions are incorrect,
  - 1 if all predictions are correct,
  - 0 otherwise.
- We use the `cluster_qr` label assignment method for spectral clustering.
- We use the `average` linkage for agglomerative clustering.
- To promote numerical instability while calculating the harmonic p-value (HMP), each p-value is replaced by `max(p_value, 1e-12)` before aggregation.
  - This lower bound is practical as well as a safeguard. For datasets of this scale, it is not realistic to claim confidence beyond “1 in a trillion.”
