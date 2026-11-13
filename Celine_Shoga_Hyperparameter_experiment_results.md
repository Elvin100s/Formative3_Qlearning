# Celine_Shoga: Hyperparameter Experiment Results

Environment: `ALE/Boxing-v5`. Each run: 200,000 steps.

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay (steps) | Avg reward (last 20 eps) | Best episode | Observed behaviour |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 100,000 | -17.4 | 13 | TODO |
| 2 | 0.0001 | 0.95 | 32 | 1.0 | 0.05 | 100,000 | 5.7 | 41 | TODO |
| 3 | 0.0001 | 0.9 | 32 | 1.0 | 0.05 | 100,000 | 3.9 | 22 | TODO |
| 4 | 0.0001 | 0.8 | 32 | 1.0 | 0.05 | 100,000 | 4.7 | 33 | TODO |
| 5 | 0.0001 | 0.999 | 32 | 1.0 | 0.05 | 100,000 | -16.3 | 10 | TODO |
| 6 | 0.0001 | 0.995 | 32 | 1.0 | 0.05 | 100,000 | -16.1 | 10 | TODO |
| 7 | 0.0002 | 0.99 | 32 | 1.0 | 0.05 | 100,000 | -16.6 | 10 | TODO |
| 8 | 0.0001 | 0.99 | 32 | 0.3 | 0.05 | 100,000 | -6.7 | 10 | TODO |
| 9 | 0.0001 | 0.99 | 32 | 1.0 | 0.01 | 50,000 | -16.9 | 7 | TODO |
| 10 | 0.0001 | 0.99 | 16 | 1.0 | 0.05 | 100,000 | -22.5 | 12 | TODO |

**Best:** Experiment 2 (5.7).  **Worst:** Experiment 10 (-22.5).
