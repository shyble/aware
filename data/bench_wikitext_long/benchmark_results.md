# Benchmark Results

Aggregated from `2` runs. Sorted by `final_val_perplexity` (lower = better).

| run_id | final_val_perplexity | final_val_loss | params | attention | discovery | cap_source | cap_window | wall_clock_seconds |
|---|---|---|---|---|---|---|---|---|
| kmeans_w3_seed42 | 20.26 | 3.0086 | 895,360 | standard | kmeans | local | 3 | 9441.4 |
| pure_transformer_seed42 | 45.65 | 3.8211 | 853,120 | standard | nodiscovery | local | 1 | 8089.3 |

## Top 3 Configurations

1. **kmeans_w3_seed42** - val_ppl `20.26` - attention=`standard`, discovery=`kmeans`, window=`3`, source=`local`
2. **pure_transformer_seed42** - val_ppl `45.65` - attention=`standard`, discovery=`nodiscovery`, window=`1`, source=`local`

## Trajectories

2 run(s) include step-by-step trajectories in their `report.json`. Use `--plot` flag to visualize.
