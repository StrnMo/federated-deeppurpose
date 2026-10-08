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

---

## Colab notebook: reliable clone and install (`notebooks/colab_run.ipynb`)

### What went wrong
On Colab the clone step failed, so `/content/federated-deeppurpose` never existed. The notebook kept
going anyway: `!git clone` and `!pip install` print an error but do not stop the notebook. pip then
could not find `requirements_colab.txt`, nothing was installed, and the first visible error came much
later: `No module named 'lifelines'` when importing DeepPurpose. That message pointed at the wrong problem.

### What changed
- **Fixed paths.** `REPO_DIR = /content/federated-deeppurpose` and the requirements path are set
  once in the config cell. Every later cell uses these absolute paths, including the training command.
- **Clone cell stops on failure.** It first checks that `REPO_URL` has been filled in. It then clones
  the repo, or updates it if it is already there, and raises a clear error if git fails or
  `requirements_colab.txt` is missing. The error names the usual causes: a wrong URL, a branch that
  doesn't exist on GitHub, or a private repo without a token. Any token in the URL is masked in printed
  output. Finally it `%cd`s into the repo and prints the repo folder, current branch and commit.
- **Install cell stops on failure.** pip runs from Python and the notebook raises an error if pip
  fails, instead of carrying on silently.
- **Git never waits for a login.** Git runs with terminal prompts and credential helpers turned off,
  so a wrong or private URL fails within seconds instead of hanging on a password prompt.
- **Train cell stops on failure.** Training runs with the notebook's own Python (`sys.executable`, not
  whichever `python` comes first on the PATH), shows its progress live, and raises an error if the
  script crashes. Rerunning the cell resumes from the last checkpoint.
- **Import check.** Before importing DeepPurpose, the notebook imports every package it needs one by
  one and lists any that fail, so a missing package is named directly.
- **`requirements_colab.txt` is complete.** It lists every package DeepPurpose 0.1.5 imports for
  MPNN-CNN: rdkit, descriptastorus, pandas-flavor, xarray, lifelines, prettytable, subword-nmt and
  wget, plus torch, numpy, pandas, scipy, scikit-learn, matplotlib, tensorboard, tqdm and requests.

### How Colab's PyTorch stays untouched
Before installing, the notebook writes Colab's installed versions of torch, numpy, pandas, scipy,
scikit-learn, matplotlib and tensorboard to a *constraints file*. pip must keep those exact versions,
so installing the requirements can add missing packages but never replace Colab's GPU build of torch.
DeepPurpose is installed with `--no-deps`, because its own dependency list would pull in other torch versions.

### NumPy 2
Colab uses NumPy 2. DeepPurpose's code uses none of the NumPy names that NumPy 2 removed
(`np.float`, `np.int`, `np.unicode_`, ...), so it runs unchanged.

---

## Fix: FedAvg predictions swinging up and down (Adam reset + output-bias init)

### The bug
In the first Colab test (5% subset, 30 rounds), validation MSE alternated between about 1 and 3 from
round to round. Logging the mean prediction showed why: after each round **all** predictions moved
up or down together by 1–3 pKd (val MSE ≈ bias² + 0.6), on the training set as well as on validation.

The cause was that every client started a **fresh Adam optimizer each round**. Adam's very first step
moves every weight by a full learning-rate step, however small the gradient is. With all weights moving at once, the
model's output jumps by a large constant and overshoots the label mean. Next round, the same thing
happens in the other direction.

The test that proved it: FedAvg with a single client holding all the data. Averaging one client
changes nothing, so this is just centralised training with Adam restarted every round. It swung
just as much, while centralised training settled within 2 epochs. So aggregation, sample weighting
and the non-IID split were not the cause.

### The fix
1. **Clients keep their Adam state between rounds.** Each client stores its optimizer state after
   local training and reloads it next round, so Adam's step sizes stay calibrated. The states are
   saved in `last.pt`, so resuming continues exactly. `--reset-client-optimizer` restores the old
   behaviour for comparison.
2. **Output bias starts at the mean training label** (all methods). An untrained model predicts
   about 0, but labels are around 5.4 pKd, so early training was spent just moving the output up.
   Now the model starts at the mean and can spend training on the differences between pairs.
   - Centralised: mean of all training labels.
   - Local-only: mean of that client's training labels.
   - FedAvg: each client reports its sample count and label mean, and the server combines them as
     `sum(n_k * mean_k) / sum(n_k)`. That equals the mean of all training labels, but no
     individual labels leave a client.

   `--init-bias none` turns this off.

### Note
Keeping Adam state on the client is a common FedAvg variant: the original FedAvg used plain SGD,
which has no optimizer state to reset. The client's Adam statistics come from its previous local
model, not the newly received global one. In practice this works well, and it is far better than
resetting.
