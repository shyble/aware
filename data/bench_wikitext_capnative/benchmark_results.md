# Benchmark Results

Aggregated from `6` runs. Sorted by `final_val_perplexity` (lower = better).

| run_id | final_val_perplexity | final_val_loss | params | attention | discovery | cap_source | cap_window | wall_clock_seconds |
|---|---|---|---|---|---|---|---|---|
| cap_native_hier_d128_seed123 | 14.33 | 2.6623 | 143,021,056 | - | - | - | 3 | 20429.1 |
| cap_native_hier_d128_seed42 | 14.58 | 2.6795 | 143,021,056 | - | - | - | 3 | 20130.5 |
| cap_native_hier_d128_seed7 | 14.62 | 2.6823 | 143,021,056 | - | - | - | 3 | 19199.1 |
| cap_native_sparse_d128_seed7 | 15.36 | 2.7317 | 368,271,616 | - | - | - | 3 | 86439.3 |
| cap_native_sparse_d128_seed123 | 15.60 | 2.7474 | 368,271,616 | - | - | - | 3 | 90739.3 |
| cap_native_sparse_d128_seed42 | 16.01 | 2.7731 | 368,271,616 | - | - | - | 3 | 72107.8 |

## Top 3 Configurations

1. **cap_native_hier_d128_seed123** - val_ppl `14.33` - attention=`None`, discovery=`None`, window=`3`, source=`None`
2. **cap_native_hier_d128_seed42** - val_ppl `14.58` - attention=`None`, discovery=`None`, window=`3`, source=`None`
3. **cap_native_hier_d128_seed7** - val_ppl `14.62` - attention=`None`, discovery=`None`, window=`3`, source=`None`

## Trajectories

6 run(s) include step-by-step trajectories in their `report.json`. Use `--plot` flag to visualize.
