**Summary.** FedAvg closes 90% (MSE) / 87% (CI) of the gap between local-only and centralized training at convergence, and 73–74% at equal training budget. Local-only models fail on other clients' drugs: on the global test set every local model has R² < 0. The centralized reference reaches MSE 0.272 / CI 0.880.

| Method | Training budget | MSE | Pearson | CI | R2 |
|---|---|---|---|---|---|
| Centralized | 68 epochs (best 53) | 0.272 | 0.815 | 0.880 | 0.66 |
| FedAvg, equal budget (best of rounds 1-50) | 50 rounds x 2 local epochs (selected round 50) | 0.428 | 0.682 | 0.818 | 0.46 |
| FedAvg, best | 137 rounds x 2 local epochs (best round 127) | 0.333 | 0.766 | 0.851 | 0.58 |
| Local-only (mean of 5 clients) | up to 100 epochs per client | 0.873 | 0.312 | 0.647 | -0.09 |

| FedAvg | Gap closed, MSE | Gap closed, CI |
|---|---|---|
| Equal budget | 74% | 73% |
| Best | 90% | 87% |

| Client | Train pairs | Local-only (own) | FedAvg, equal budget | FedAvg, best | Centralized |
|---|---|---|---|---|---|
| 0 | 4,022 | 0.298 / 0.840 | 0.390 / 0.846 | 0.278 / 0.886 | 0.234 / 0.898 |
| 1 | 4,022 | 0.329 / 0.849 | 0.386 / 0.805 | 0.324 / 0.842 | 0.247 / 0.875 |
| 2 | 4,022 | 0.333 / 0.748 | 0.393 / 0.745 | 0.286 / 0.822 | 0.228 / 0.840 |
| 3 | 4,022 | 0.343 / 0.858 | 0.449 / 0.848 | 0.332 / 0.864 | 0.257 / 0.885 |
| 4 | 4,950 | 0.410 / 0.859 | 0.507 / 0.837 | 0.423 / 0.856 | 0.368 / 0.874 |
| Mean | | 0.343 / 0.831 | 0.425 / 0.816 | 0.329 / 0.854 | 0.267 / 0.874 |

| Local-only model | Best epoch | Epochs run | Own test MSE / CI | Global test MSE / CI |
|---|---|---|---|---|
| Client 0 | 100 | 100 (limit) | 0.298 / 0.840 | 0.866 / 0.626 |
| Client 1 | 53 | 68 | 0.329 / 0.849 | 0.884 / 0.595 |
| Client 2 | 69 | 84 | 0.333 / 0.748 | 0.897 / 0.615 |
| Client 3 | 73 | 88 | 0.343 / 0.858 | 0.836 / 0.683 |
| Client 4 | 64 | 79 | 0.410 / 0.859 | 0.882 / 0.718 |
