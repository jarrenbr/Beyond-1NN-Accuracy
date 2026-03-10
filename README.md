## Reference Implementation Only

We provide this repository for implementation transparency. We intend to help readers to understand low-level details, such as edge cases. It does not run as-is.

Low-level detail examples:
- We apply the exact Mann-Whitney U test for partitions with less than 30 samples.
- The handling of undefined MCC values, where MCC is set to:
  - -1 if all predictions are incorrect,
  - 1 if all predictions are correct,
  - 0 otherwise.
- We use the `cluster_qr` label assignment method for spectral clustering.
- To promote numerical instability while calculating the harmonic p-value (HMP), each p-value is replaced by `max(p_value, 1e-12)` before aggregation.
  - This lower bound is practical as well as a safeguard. For datasets of this scale, it is not realistic to claim confidence beyond “1 in a trillion.”
