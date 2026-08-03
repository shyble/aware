# Benchmark Results

Aggregated from `9` runs. Sorted by `final_val_perplexity` (lower = better).

| run_id | final_val_perplexity | final_val_loss | params | attention | discovery | cap_source | cap_window | wall_clock_seconds |
|---|---|---|---|---|---|---|---|---|
| dense_30m_seed7 | 33.10 | 3.4994 | 25,436,672 | standard | nodiscovery | local | 1 | 2645.6 |
| dense_30m_seed42 | 33.28 | 3.5050 | 25,436,672 | standard | nodiscovery | local | 1 | 2649.8 |
| dense_30m_seed123 | 33.55 | 3.5130 | 25,436,672 | standard | nodiscovery | local | 1 | 2649.4 |
| dense_10m_seed123 | 35.89 | 3.5806 | 10,818,432 | standard | nodiscovery | local | 1 | 1393.6 |
| dense_10m_seed42 | 36.05 | 3.5849 | 10,818,432 | standard | nodiscovery | local | 1 | 1396.6 |
| dense_10m_seed7 | 36.38 | 3.5940 | 10,818,432 | standard | nodiscovery | local | 1 | 1392.5 |
| dense_3m_seed7 | 40.43 | 3.6995 | 3,279,104 | standard | nodiscovery | local | 1 | 587.4 |
| dense_3m_seed123 | 40.56 | 3.7027 | 3,279,104 | standard | nodiscovery | local | 1 | 584.4 |
| dense_3m_seed42 | 40.75 | 3.7074 | 3,279,104 | standard | nodiscovery | local | 1 | 575.6 |

## Top 3 Configurations

1. **dense_30m_seed7** - val_ppl `33.10` - attention=`standard`, discovery=`nodiscovery`, window=`1`, source=`local`
2. **dense_30m_seed42** - val_ppl `33.28` - attention=`standard`, discovery=`nodiscovery`, window=`1`, source=`local`
3. **dense_30m_seed123** - val_ppl `33.55` - attention=`standard`, discovery=`nodiscovery`, window=`1`, source=`local`

## Trajectories

9 run(s) include step-by-step trajectories in their `report.json`. Use `--plot` flag to visualize.
