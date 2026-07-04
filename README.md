# Federated Learning for Drug-Target Interaction Prediction

## Architecture Implementation (Work in Progress)
This project implements the **architecture** for federated learning (FedAvg) for drug-target interaction prediction using the DAVIS dataset. The current implementation includes:

- ✅ Data loading and preprocessing (DAVIS dataset)
- ✅ Non-IID client partitioning (drug-based)
- ✅ Centralized baseline training
- ✅ FL Client/Server class structure
- ✅ Model weight management infrastructure

### Under Development

The **full FedAvg communication rounds** (clients sending weights to server, server aggregating, distributing back) are currently being implemented. The current code simulates independent client training to demonstrate the performance gap between centralized and isolated training.


### 📊 Current Results

### Centralized Baseline (10 epochs)
- **Concordance Index:** 0.731
- **Test MSE:** 0.615
- **Pearson Correlation:** 0.440

### Client Performance (Independent Training)
| Client | Samples | Unique Drugs | MSE | Pearson | C-Index |
|--------|---------|--------------|-----|---------|---------|
| Client 0 | 5,746 | 13 | 0.822 | 0.212 | 0.630 |
| Client 1 | 5,746 | 13 | 0.774 | 0.365 | 0.693 |
| Client 2 | 5,746 | 13 | 0.474 | 0.200 | 0.596 |
| Client 3 | 5,746 | 13 | 0.721 | 0.342 | 0.669 |
| Client 4 | 7,072 | 16 | 0.894 | 0.369 | 0.698 |

*These results demonstrate the performance variation across non-IID clients, highlighting the need for federated aggregation.*


## Visualizations

| Centralized Training | Federated vs Centralized |
|:--------------------:|:------------------------:|
| ![Centralized Loss](results/centralized_loss_curve.png) | ![Federated Comparison](results/federated_comparison.png) |

| Client Data Distribution | Performance Summary |
|:------------------------:|:-------------------:|
| ![Client Distribution](results/client_data_distribution.png) | ![Performance Table](results/performance_table.png) |

### Key Observations

1. **Centralized performance gap**: Centralized model (C-Index: 0.731) outperforms individual clients (C-Index: 0.596–0.698), demonstrating the value of data sharing.
2. **Non-IID heterogeneity**: Client 2 shows the lowest C-Index (0.596) with only 0.200 Pearson correlation, highlighting the challenge of skewed drug distributions.
3. **Data imbalance**: Client 4 has the most samples (7,072) and unique drugs (16), correlating with the highest C-Index among clients (0.698).


### Project Structure
```
federated-deeppurpose/
├── data/                              # DAVIS dataset
├── experiments/                       # Centralized & FL runs
│   ├── run_centralized.py             # Baseline training
│   └── run_federated.py               # FL simulation
├── federated/                         # FL components
│   ├── client.py                      # Client class with weight management
│   ├── server.py                      # Server class with FedAvg aggregation
│   └── fedavg.py                      # Training loop structure
├── utils/                             # Utilities
│   ├── load_davis.py                  # Data loading
│   └── data_split.py                  # Non-IID client creation
├── results/                           # Visualizations
│   ├── centralized_loss_curve.png
│   ├── client_data_distribution.png
│   ├── combined_summary.png
│   ├── federated_comparison.png
│   └── performance_table.png
└── README.md
```

## 📝 Next Steps

- [ ] Implement full FedAvg communication rounds with weight aggregation
- [ ] Add cross-client evaluation metrics
- [ ] Compare FL vs. local-only and centralized performance
- [ ] Prepare manuscript for publication (code will be released upon submission)


## Technologies
- Python 3.8+
- DeepPurpose (TensorFlow/Keras backend)
- NumPy, Pandas, Matplotlib
- scikit-learn

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run centralized baseline
python experiments/run_centralized.py

# Run federated learning simulation (work in progress)
python experiments/run_federated.py

# Generate all plots
python experiments/generate_plots.py
``` 
