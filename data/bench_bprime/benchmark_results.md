# Benchmark Results

Aggregated from `27` runs. Sorted by `final_val_perplexity` (lower = better).

| run_id | final_val_perplexity | final_val_loss | params | attention | discovery | cap_source | cap_window | wall_clock_seconds |
|---|---|---|---|---|---|---|---|---|
| pure_transformer_seed123 | 26.90 | 3.2920 | 853,120 | standard | nodiscovery | local | 1 | 2680.5 |
| cappair_multihead_seed123 | 26.96 | 3.2942 | 4,916,352 | cap_pair | nodiscovery | local | 1 | 4648.2 |
| cappair_multihead_seed7 | 27.15 | 3.3013 | 4,916,352 | cap_pair | nodiscovery | local | 1 | 4673.7 |
| cappair_multihead_kmeans_seed123 | 27.30 | 3.3068 | 4,916,352 | cap_pair | nodiscovery | local | 1 | 4513.7 |
| cappair_multihead_matched_seed123 | 27.35 | 3.3088 | 1,770,624 | cap_pair | nodiscovery | local | 1 | 3686.3 |
| cappair_seed123 | 27.39 | 3.3104 | 1,770,624 | cap_pair | nodiscovery | local | 1 | 3439.4 |
| cappair_seed42 | 27.54 | 3.3156 | 1,770,624 | cap_pair | nodiscovery | local | 1 | 3521.7 |
| cappair_multihead_matched_seed42 | 27.59 | 3.3173 | 1,770,624 | cap_pair | nodiscovery | local | 1 | 3644.2 |
| cappair_multihead_kmeans_seed7 | 27.65 | 3.3197 | 4,916,352 | cap_pair | nodiscovery | local | 1 | 4520.1 |
| cappair_multihead_matched_seed7 | 27.68 | 3.3206 | 1,770,624 | cap_pair | nodiscovery | local | 1 | 3644.2 |
| cappair_multihead_seed42 | 27.76 | 3.3235 | 4,916,352 | cap_pair | nodiscovery | local | 1 | 4637.4 |
| cappair_multihead_kmeans_seed42 | 27.80 | 3.3249 | 4,916,352 | cap_pair | nodiscovery | local | 1 | 4531.5 |
| cappair_seed7 | 27.82 | 3.3256 | 1,770,624 | cap_pair | nodiscovery | local | 1 | 3418.5 |
| pure_transformer_seed7 | 28.54 | 3.3513 | 853,120 | standard | nodiscovery | local | 1 | 2686.5 |
| pure_transformer_seed42 | 28.92 | 3.3645 | 853,120 | standard | nodiscovery | local | 1 | 2655.5 |
| pure_transformer_1head_seed123 | 29.08 | 3.3701 | 853,120 | standard | nodiscovery | local | 1 | 2817.1 |
| pure_transformer_1head_seed42 | 29.93 | 3.3988 | 853,120 | standard | nodiscovery | local | 1 | 2821.3 |
| pure_transformer_1head_seed7 | 30.01 | 3.4014 | 853,120 | standard | nodiscovery | local | 1 | 2795.4 |
| capmem_seed123 | 31.88 | 3.4621 | 722,048 | cap_memory | nodiscovery | local | 1 | 2397.4 |
| capmem_multihead_discovered_seed123 | 32.04 | 3.4669 | 722,048 | cap_memory | nodiscovery | local | 1 | 2647.1 |
| capmem_multihead_seed123 | 32.05 | 3.4673 | 722,048 | cap_memory | nodiscovery | local | 1 | 2644.6 |
| capmem_multihead_seed7 | 32.18 | 3.4714 | 722,048 | cap_memory | nodiscovery | local | 1 | 2636.8 |
| capmem_multihead_discovered_seed7 | 32.19 | 3.4717 | 722,048 | cap_memory | nodiscovery | local | 1 | 2640.4 |
| capmem_seed7 | 32.40 | 3.4783 | 722,048 | cap_memory | nodiscovery | local | 1 | 2432.1 |
| capmem_multihead_discovered_seed42 | 32.61 | 3.4847 | 722,048 | cap_memory | nodiscovery | local | 1 | 2645.2 |
| capmem_multihead_seed42 | 32.62 | 3.4848 | 722,048 | cap_memory | nodiscovery | local | 1 | 2626.7 |
| capmem_seed42 | 32.63 | 3.4852 | 722,048 | cap_memory | nodiscovery | local | 1 | 2383.5 |

## Top 3 Configurations

1. **pure_transformer_seed123** - val_ppl `26.90` - attention=`standard`, discovery=`nodiscovery`, window=`1`, source=`local`
2. **cappair_multihead_seed123** - val_ppl `26.96` - attention=`cap_pair`, discovery=`nodiscovery`, window=`1`, source=`local`
3. **cappair_multihead_seed7** - val_ppl `27.15` - attention=`cap_pair`, discovery=`nodiscovery`, window=`1`, source=`local`

## Trajectories

27 run(s) include step-by-step trajectories in their `report.json`. Use `--plot` flag to visualize.
