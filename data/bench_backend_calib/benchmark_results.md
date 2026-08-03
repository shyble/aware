# Benchmark Results

Aggregated from `3` runs. Sorted by `final_val_perplexity` (lower = better).

| run_id | final_val_perplexity | final_val_loss | params | attention | discovery | cap_source | cap_window | wall_clock_seconds |
|---|---|---|---|---|---|---|---|---|
| pure_transformer_seed7 | 50.95 | 3.9309 | 853,120 | standard | nodiscovery | local | 1 | 137.8 |
| pure_transformer_seed123 | 51.04 | 3.9326 | 853,120 | standard | nodiscovery | local | 1 | 137.7 |
| pure_transformer_seed42 | 51.68 | 3.9450 | 853,120 | standard | nodiscovery | local | 1 | 137.0 |

## Top 3 Configurations

1. **pure_transformer_seed7** - val_ppl `50.95` - attention=`standard`, discovery=`nodiscovery`, window=`1`, source=`local`
2. **pure_transformer_seed123** - val_ppl `51.04` - attention=`standard`, discovery=`nodiscovery`, window=`1`, source=`local`
3. **pure_transformer_seed42** - val_ppl `51.68` - attention=`standard`, discovery=`nodiscovery`, window=`1`, source=`local`

## Trajectories

3 run(s) include step-by-step trajectories in their `report.json`. Use `--plot` flag to visualize.
