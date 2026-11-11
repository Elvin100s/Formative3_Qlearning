# Elvin Cyubahiro: Hyperparameter Experiment Results

**Member:** Elvin Cyubahiro · **Environment:** `ALE/Boxing-v5` · **Each run:** 200,000 steps, same seed

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay (steps) | Start: first 20 rounds | Low point (20-round avg) | End: last 20 rounds | Wins in last 20 | Observed behaviour |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 100,000 | -0.6 | -14.2 at 119k | -11.4 | 1/20 | TODO |
| 2 | 0.00005 | 0.99 | 32 | 1.0 | 0.05 | 100,000 | +0.6 | -24.4 at 137k | -13.6 | 1/20 | TODO |
| 3 | 0.00025 | 0.99 | 32 | 1.0 | 0.05 | 100,000 | +0.8 | -24.9 at 135k | -15.8 | 0/20 | TODO |
| 4 | 0.0005 | 0.99 | 32 | 1.0 | 0.05 | 100,000 | -0.3 | -27.6 at 130k | -19.1 | 0/20 | TODO |
| 5 | 0.001 | 0.99 | 32 | 1.0 | 0.05 | 100,000 | -0.1 | -37.4 at 138k | -31.1 | 0/20 | TODO |
| 6 | 0.00001 | 0.99 | 32 | 1.0 | 0.05 | 100,000 | +0.2 | -33.0 at 138k | -25.6 | 0/20 | TODO |
| 7 | 0.0001 | 0.98 | 32 | 1.0 | 0.05 | 100,000 | -0.3 | -13.7 at 119k | -4.2 | 5/20 | TODO |
| 8 | 0.0001 | 0.99 | 64 | 1.0 | 0.05 | 100,000 | +1.3 | -5.2 at 124k | +4.8 | 13/20 | TODO |
| 9 | 0.0001 | 0.99 | 32 | 0.8 | 0.05 | 100,000 | -1.9 | -15.4 at 133k | -10.8 | 1/20 | TODO |
| 10 | 0.0001 | 0.99 | 32 | 1.0 | 0.02 | 150,000 | -1.2 | -14.3 at 165k | -12.4 | 2/20 | TODO |

Best: Experiment 8 (end +4.8).  Worst: Experiment 5 (end -31.1).

**How to read the reward columns.** Reward per round = my points minus the opponent's points, so 0 is a draw. **Start** = first 20 rounds (mostly random play). **Low point** = worst 20-round average during training. **End** = last 20 rounds. **Wins** = rounds above 0 among the last 20.
