# Benchmark Results

Aggregated from `49` runs. Sorted by `final_val_perplexity` (lower = better).

| run_id | final_val_perplexity | final_val_loss | params | attention | discovery | cap_source | cap_window | wall_clock_seconds |
|---|---|---|---|---|---|---|---|---|
| kmeans_w2_seed42 | 28.91 | 3.3642 | 895,360 | standard | kmeans | local | 2 | 2986.8 |
| kmeans_w2_seed200 | 28.93 | 3.3650 | 895,360 | standard | kmeans | local | 2 | 3052.2 |
| kmeans_w2_seed123 | 29.09 | 3.3705 | 895,360 | standard | kmeans | local | 2 | 2979.9 |
| disc_hybrid_w3_seed123 | 29.35 | 3.3791 | 895,360 | standard | hybrid | local | 3 | 3043.5 |
| kmeans_w2_seed7 | 29.51 | 3.3846 | 895,360 | standard | kmeans | local | 2 | 3035.4 |
| kmeans_w3_seed123 | 29.87 | 3.3970 | 895,360 | standard | kmeans | local | 3 | 3068.7 |
| kmeans_w3_seed200 | 30.03 | 3.4024 | 895,360 | standard | kmeans | local | 3 | 3066.7 |
| kmeans_w3_seed7 | 30.06 | 3.4031 | 895,360 | standard | kmeans | local | 3 | 3045.7 |
| kmeans_w2_seed100 | 30.09 | 3.4041 | 895,360 | standard | kmeans | local | 2 | 3055.8 |
| kmeans_w3_seed42 | 30.55 | 3.4193 | 895,360 | standard | kmeans | local | 3 | 3147.3 |
| disc_hybrid_w3_seed42 | 30.79 | 3.4271 | 895,360 | standard | hybrid | local | 3 | 3047.0 |
| disc_random_w3_seed123 | 30.87 | 3.4298 | 895,360 | standard | random | local | 3 | 3068.4 |
| disc_hybrid_w3_seed7 | 31.19 | 3.4401 | 895,360 | standard | hybrid | local | 3 | 3086.1 |
| kmeans_w3_seed100 | 31.34 | 3.4448 | 895,360 | standard | kmeans | local | 3 | 3039.4 |
| disc_random_w3_seed42 | 32.83 | 3.4914 | 895,360 | standard | random | local | 3 | 3009.5 |
| disc_random_w3_seed7 | 33.22 | 3.5030 | 895,360 | standard | random | local | 3 | 3093.9 |
| kmeans_w4_seed42 | 33.40 | 3.5086 | 895,360 | standard | kmeans | local | 4 | 3105.8 |
| disc_nodiscovery_w3_seed123 | 33.61 | 3.5148 | 895,360 | standard | nodiscovery | local | 3 | 2985.6 |
| disc_nodiscovery_w3_seed7 | 34.04 | 3.5276 | 895,360 | standard | nodiscovery | local | 3 | 2931.4 |
| kmeans_w4_seed7 | 34.50 | 3.5410 | 895,360 | standard | kmeans | local | 4 | 3118.2 |
| kmeans_w4_seed123 | 34.50 | 3.5411 | 895,360 | standard | kmeans | local | 4 | 3100.2 |
| disc_nodiscovery_w3_seed42 | 34.55 | 3.5425 | 895,360 | standard | nodiscovery | local | 3 | 2992.1 |
| kmeans_w5_seed42 | 41.82 | 3.7333 | 895,360 | standard | kmeans | local | 5 | 3117.8 |
| kmeans_w5_seed123 | 42.82 | 3.7571 | 895,360 | standard | kmeans | local | 5 | 3160.6 |
| kmeans_w5_seed7 | 43.17 | 3.7651 | 895,360 | standard | kmeans | local | 5 | 3158.5 |
| cappair_multihead_seed123 | 50.71 | 3.9261 | 4,916,352 | cap_pair | nodiscovery | local | 1 | 253.8 |
| cappair_multihead_seed7 | 50.90 | 3.9299 | 4,916,352 | cap_pair | nodiscovery | local | 1 | 254.3 |
| cappair_multihead_matched_seed7 | 50.96 | 3.9310 | 1,770,624 | cap_pair | nodiscovery | local | 1 | 166.1 |
| cappair_multihead_kmeans_seed123 | 50.97 | 3.9311 | 4,916,352 | cap_pair | nodiscovery | local | 1 | 250.4 |
| cappair_multihead_kmeans_seed7 | 51.10 | 3.9337 | 4,916,352 | cap_pair | nodiscovery | local | 1 | 251.2 |
| cappair_multihead_matched_seed123 | 51.15 | 3.9348 | 1,770,624 | cap_pair | nodiscovery | local | 1 | 165.5 |
| pure_transformer_seed42 | 51.22 | 3.9361 | 853,120 | standard | nodiscovery | local | 1 | 2630.0 |
| cappair_multihead_seed42 | 51.31 | 3.9380 | 4,916,352 | cap_pair | nodiscovery | local | 1 | 253.5 |
| pure_transformer_seed123 | 51.32 | 3.9380 | 853,120 | standard | nodiscovery | local | 1 | 2642.2 |
| cappair_multihead_matched_seed42 | 51.56 | 3.9428 | 1,770,624 | cap_pair | nodiscovery | local | 1 | 164.8 |
| cappair_multihead_kmeans_seed42 | 51.69 | 3.9453 | 4,916,352 | cap_pair | nodiscovery | local | 1 | 249.8 |
| pure_transformer_seed7 | 51.71 | 3.9456 | 853,120 | standard | nodiscovery | local | 1 | 2642.9 |
| kmeans_w1_seed123 | 51.96 | 3.9504 | 895,360 | standard | kmeans | local | 1 | 2816.4 |
| kmeans_w1_seed7 | 52.55 | 3.9617 | 895,360 | standard | kmeans | local | 1 | 2825.3 |
| capmem_multihead_seed123 | 53.02 | 3.9706 | 722,048 | cap_memory | nodiscovery | local | 1 | 120.2 |
| capmem_multihead_discovered_seed123 | 53.07 | 3.9717 | 722,048 | cap_memory | nodiscovery | local | 1 | 118.9 |
| capmem_multihead_seed7 | 53.08 | 3.9717 | 722,048 | cap_memory | nodiscovery | local | 1 | 120.9 |
| capmem_multihead_discovered_seed7 | 53.34 | 3.9766 | 722,048 | cap_memory | nodiscovery | local | 1 | 119.3 |
| kmeans_w1_seed42 | 53.70 | 3.9834 | 895,360 | standard | kmeans | local | 1 | 2995.2 |
| capmem_multihead_seed42 | 53.97 | 3.9884 | 722,048 | cap_memory | nodiscovery | local | 1 | 119.0 |
| capmem_multihead_discovered_seed42 | 54.11 | 3.9910 | 722,048 | cap_memory | nodiscovery | local | 1 | 120.1 |
| kmeans_w8_seed7 | 68.39 | 4.2253 | 895,360 | standard | kmeans | local | 8 | 3329.1 |
| kmeans_w8_seed42 | 71.09 | 4.2639 | 895,360 | standard | kmeans | local | 8 | 3351.2 |
| kmeans_w8_seed123 | 73.71 | 4.3001 | 895,360 | standard | kmeans | local | 8 | 3309.4 |

## Top 3 Configurations

1. **kmeans_w2_seed42** - val_ppl `28.91` - attention=`standard`, discovery=`kmeans`, window=`2`, source=`local`
2. **kmeans_w2_seed200** - val_ppl `28.93` - attention=`standard`, discovery=`kmeans`, window=`2`, source=`local`
3. **kmeans_w2_seed123** - val_ppl `29.09` - attention=`standard`, discovery=`kmeans`, window=`2`, source=`local`

## Trajectories

49 run(s) include step-by-step trajectories in their `report.json`. Use `--plot` flag to visualize.
