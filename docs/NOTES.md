# Project notes

Short explanations of each change: what changed, why, and how it works.

---

## FedAvg implementation (`experiments/run_federated.py`)

### What changed
`run_federated.py` used to train five separate client models with no communication (that baseline
now lives in `run_local.py`). It now runs real Federated Averaging (FedAvg): the clients train one
shared model together without ever sharing their data. The code is split into:

- `federated/client.py`: one client and its local training
- `federated/server.py`: the global weights and the averaging step
- `federated/fedavg.py`: the round loop, evaluation, checkpoints and CSV logs

### How it works

**Setup.** Each of the 5 clients holds only its own drugs' training pairs (about 4,000 pairs each,
4,950 for client 4). One MPNN-CNN model is created with a fixed seed. Its weights are the first
*global model*, so every client starts from exactly the same point.

**The round loop.** One round = one exchange between server and clients:

1. **Broadcast:** the server sends the current global weights to every client.
2. **Local training:** each client loads those weights and trains on its own data for
   `--local-epochs` epochs (default 2), then sends its updated weights back. Only weights travel,
   never data.
3. **Aggregation:** the server averages the 5 sets of weights into a new global model.
4. **Evaluation:** the new global model is scored on the validation and test sets.

This repeats for `--rounds` rounds (default 30). All clients take part in every round.

**Local training details.** Each client gets a fresh Adam optimizer every round. Clients therefore keep
no memory between rounds; everything they learned is carried by the weights they send back. This is
standard FedAvg. The order in which a client sees its data is shuffled with a seed built from the
run seed, round and client, so a run is reproducible.

**Sample-weighted aggregation.** The new global weights are a weighted average of the client weights,
with each client counted in proportion to its number of training pairs:

    w_global = sum over clients k of (n_k / n_total) * w_k

Client 4 has 4,950 of the 21,038 training pairs, so its weights count 23.5%. Each other client
counts 19.1%. A plain average would give a client with little data as much say as one with a lot;
weighting by size is part of the FedAvg definition.

**Federated evaluation.** After every round the global model predicts on:
- the **global validation set**: the union of all clients' validation pairs (3,007). Its MSE picks
  the best round.
- the **global test set**: the union of all clients' test pairs (6,011). It is logged every round to
  draw learning curves but is never used to choose a model, which keeps the final test score honest.

Every score is also broken down per client (each client's own drugs), which shows whether the shared
model works for every client or only for some.

In this simulation the server scores pooled predictions directly. In a real deployment each client
would score the global model locally and report back. MSE could be combined exactly from those
reports, but CI and Pearson need all predictions together, so pooled scoring gives the cleanest numbers.

### Outputs and resuming
After every round the run saves `last.pt` (global weights, best model so far, random-number state)
and adds a row to `metrics.csv`, plus per-client rows to `client_metrics.csv`. `best.pt` holds the
round with the lowest validation MSE. If Colab disconnects, rerunning with `--resume` and the same
`--output-dir` continues from the last finished round. A resumed run gives the same results as an
uninterrupted one (checked on CPU). At the end, `final_metrics.json` stores the best round's
validation and test scores, overall and per client.

### How it compares
All three methods use the same data split, so their test scores can be compared directly:
- **Centralized** (`run_centralized.py`): one model sees all training data. This is the upper bound.
- **Local-only** (`run_local.py`): each client trains alone and never communicates. This is the lower bound.
- **FedAvg** (`run_federated.py`): clients share weights, not data. It is expected to land in between.
