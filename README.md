# Federated Learning Framework for Drug–Target Interaction Prediction using DeepPurpose

## Overview

Drug–Target Interaction (DTI) prediction is one of the fundamental tasks in computational drug discovery and drug repurposing. Recent deep learning frameworks such as **DeepPurpose** have demonstrated strong predictive performance by learning representations of drugs and target proteins from large-scale datasets. However, these methods assume that all training data are centrally available, which is often unrealistic in biomedical research due to privacy regulations, institutional policies, and proprietary pharmaceutical data.

This project investigates how **Federated Learning (FL)** can be integrated with the DeepPurpose framework to enable collaborative training across multiple institutions while keeping local datasets private. The long-term objective is to develop a privacy-preserving federated framework for drug-target interaction prediction and drug repurposing under heterogeneous (Non-IID) data distributions.

---

## Current Project Status

**Project Stage:** Active Research & Software Development

This repository is currently under active development and serves as the software foundation for ongoing research in Federated Learning for drug-target interaction prediction.

### Currently Implemented

- ✅ DAVIS dataset preprocessing and loading
- ✅ Drug-based Non-IID client partitioning
- ✅ Centralized DeepPurpose baseline
- ✅ Federated Client architecture
- ✅ Federated Server architecture
- ✅ Model parameter management
- ✅ Baseline experiments and visualization pipeline

### Currently Under Development

The complete Federated Learning optimization pipeline is being implemented.

The current version already contains the software infrastructure required for Federated Learning, including the client-server architecture and model parameter management. However, the complete **FedAvg communication protocol** (global model synchronization, weighted aggregation, and iterative communication rounds) is still under development.

At this stage, the repository provides **independent local client training** on heterogeneous datasets. These experiments are intended as **baseline experiments** for comparison with the future Federated Learning implementation and should not be interpreted as final Federated Learning results.

---

## Research Motivation

Although DeepPurpose provides powerful deep learning models for drug-target interaction prediction, it assumes centralized access to biomedical datasets.

In practice, biomedical data are naturally distributed across different organizations such as

- hospitals,
- pharmaceutical companies,
- research laboratories,
- medical centers.

Because these institutions often cannot share sensitive data directly, Federated Learning offers an attractive solution by allowing collaborative model training without exchanging raw data.

This project aims to bridge **DeepPurpose** and **Federated Learning** by developing a privacy-preserving framework for distributed drug-target interaction prediction.

---

## Current Baseline Experimental Results

### Centralized Baseline

Training was performed on the complete DAVIS dataset using the DeepPurpose MPNN-CNN architecture.

| Metric | Value |
|---------|------:|
| Test MSE | 0.615 |
| Pearson Correlation | 0.440 |
| Concordance Index | 0.731 |

### Independent Client Training (Non-IID)

To investigate the effect of heterogeneous data distributions, the DAVIS dataset was partitioned into five drug-based Non-IID clients. Each client was trained independently without model aggregation.

| Client | Samples | Unique Drugs | MSE | Pearson | C-Index |
|--------|---------|--------------|-----|---------|---------|
| Client 0 | 5,746 | 13 | 0.822 | 0.212 | 0.630 |
| Client 1 | 5,746 | 13 | 0.774 | 0.365 | 0.693 |
| Client 2 | 5,746 | 13 | 0.474 | 0.200 | 0.596 |
| Client 3 | 5,746 | 13 | 0.721 | 0.342 | 0.669 |
| Client 4 | 7,072 | 16 | 0.894 | 0.369 | 0.698 |

These baseline experiments demonstrate the impact of Non-IID data distributions on local model performance and provide a reference for evaluating future Federated Learning algorithms.

---

## Experimental Observations

The preliminary experiments reveal several important characteristics of distributed drug-target interaction prediction.

- The centralized model consistently outperforms independently trained local models, illustrating the performance loss caused by isolated learning.

- Considerable variation exists across clients due to heterogeneous drug distributions, confirming the challenges introduced by Non-IID data.

- Client performance appears to be influenced by both dataset size and drug diversity, motivating the need for collaborative Federated Learning optimization.

These observations establish the motivation for implementing Federated Learning algorithms such as FedAvg and evaluating their ability to recover centralized performance while preserving data privacy.

---

## Repository Structure

```text
federated-deeppurpose/

├── data/                 # DAVIS dataset
├── experiments/          # Centralized and FL experiments
├── federated/            # Client, Server and aggregation modules
├── utils/                # Dataset loading and preprocessing
├── results/              # Experimental figures
└── README.md
```

---

## Current Development Roadmap

### Phase 1 (Completed)

- ✅ DAVIS preprocessing
- ✅ DeepPurpose centralized baseline
- ✅ Non-IID client partitioning
- ✅ Client-Server software architecture
- ✅ Parameter management

### Phase 2 (In Progress)

- ⏳ Complete FedAvg implementation
- ⏳ Global model synchronization
- ⏳ Communication rounds
- ⏳ Federated evaluation pipeline

### Phase 3 (Planned)

- Federated optimization under Non-IID settings
- FedProx implementation
- Personalized Federated Learning
- Extensive benchmarking
- Research paper preparation

---

## Technologies

- Python
- DeepPurpose
- TensorFlow / Keras
- NumPy
- Pandas
- scikit-learn
- Matplotlib

---

## Running the Project

```bash
pip install -r requirements.txt

python experiments/run_centralized.py

python experiments/run_federated.py
```

---

## Disclaimer

This repository is an active research project. Both the software architecture and the experimental pipeline are continuously evolving as part of ongoing work toward a complete Federated Learning framework for drug-target interaction prediction.
