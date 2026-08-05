# Benchmark Results

Aggregated from `11` runs. Sorted by `final_val_perplexity` (lower = better).

| run_id | final_val_perplexity | final_val_loss | params | attention | discovery | cap_source | cap_window | wall_clock_seconds |
|---|---|---|---|---|---|---|---|---|
| cap_native_hier_d128_seed7 | 8.84 | 2.1788 | 143,021,056 | - | - | - | 3 | 10060.3 |
| cap_native_hier_d128_seed42 | 8.91 | 2.1874 | 143,021,056 | - | - | - | 3 | 9878.3 |
| cap_native_hier_d128_seed123 | 9.12 | 2.2101 | 143,021,056 | - | - | - | 3 | 9418.7 |
| cap_native_sparse_d128_seed7 | 10.09 | 2.3113 | 368,271,616 | - | - | - | 3 | 43467.3 |
| cap_native_sparse_d128_seed123 | 10.47 | 2.3486 | 368,271,616 | - | - | - | 3 | 40013.7 |
| cap_native_sparse_d128_seed42 | 10.93 | 2.3915 | 368,271,616 | - | - | - | 3 | 68840.2 |
| kmeans_w3_seed42 | 13.72 | 2.6190 | 895,360 | standard | kmeans | local | 3 | 3126.4 |
| kmeans_w3_d64_seed123 | 32.36 | 3.4769 | 234,048 | standard | kmeans | local | 3 | 1565.7 |
| kmeans_w3_d64_seed42 | 35.45 | 3.5682 | 234,048 | standard | kmeans | local | 3 | 1521.5 |
| pure_transformer_d64_seed42 | 35.80 | 3.5780 | 229,952 | standard | nodiscovery | local | 1 | 1433.9 |
| cap_native_sparse_d64_small_seed42 | 170.57 | 5.1391 | 5,296,128 | - | - | - | 4 | 105.2 |

## Top 3 Configurations

1. **cap_native_hier_d128_seed7** - val_ppl `8.84` - attention=`None`, discovery=`None`, window=`3`, source=`None`
2. **cap_native_hier_d128_seed42** - val_ppl `8.91` - attention=`None`, discovery=`None`, window=`3`, source=`None`
3. **cap_native_hier_d128_seed123** - val_ppl `9.12` - attention=`None`, discovery=`None`, window=`3`, source=`None`

## Trajectories

11 run(s) include step-by-step trajectories in their `report.json`. Use `--plot` flag to visualize.
