"""
DeepPurpose MPNN-CNN featurisation of the evaluation protocol.

Each unique drug and protein is encoded once. DTIDataset returns the same items
as DeepPurpose's data_process_loader, but caches the CNN protein one-hot matrix
per unique sequence instead of rebuilding it for every sample in every epoch.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils import data

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DeepPurpose.utils import encode_drug, encode_protein, protein_2_embed, mpnn_collate_func
from utils.protocol import load_protocol, subsample_protocol, SPLITS

DRUG_ENCODING = "MPNN"
TARGET_ENCODING = "CNN"

# Target Sequence -> float32 one-hot matrix (26 x 1000), shared by all datasets in the process
_PROTEIN_CACHE = {}


def featurize_protocol(num_clients=5, seed=42, subsample=None):
    """
    Load the protocol and add DeepPurpose encodings.

    Returns the same layout as load_protocol(); every DataFrame has a 0..n-1 index and
    columns idx, SMILES, Target Sequence, Label, client, drug_encoding, target_encoding.
    """
    proto = load_protocol(num_clients=num_clients, seed=seed)
    if subsample:
        proto = subsample_protocol(proto, subsample, seed)

    parts = [d[s].assign(split=s) for d in proto["clients"] for s in SPLITS]
    full = pd.concat(parts, ignore_index=True).rename(columns={"Drug": "SMILES", "Target": "Target Sequence"})
    full = encode_drug(full, DRUG_ENCODING)
    full = encode_protein(full, TARGET_ENCODING)

    def select(mask):
        return full[mask].drop(columns="split").reset_index(drop=True)

    clients = [{s: select((full["client"] == c) & (full["split"] == s)) for s in SPLITS}
               for c in range(num_clients)]
    global_sets = {s: select(full["split"] == s) for s in SPLITS}
    return {"clients": clients, "global": global_sets}


def _protein_matrix(seq, encoded):
    m = _PROTEIN_CACHE.get(seq)
    if m is None:
        # 0/1 values are exact in float32; the training loop casts to float32 anyway
        m = protein_2_embed(encoded).astype(np.float32)
        _PROTEIN_CACHE[seq] = m
    return m


class DTIDataset(data.Dataset):
    """(drug MPNN features, protein one-hot, label) items for MPNN-CNN."""

    def __init__(self, df, cache_proteins=True):
        self.drug = df["drug_encoding"].tolist()
        self.labels = df["Label"].to_numpy()
        self.client = df["client"].to_numpy()
        self.cache_proteins = cache_proteins
        if cache_proteins:
            self.protein = [_protein_matrix(s, e) for s, e in zip(df["Target Sequence"], df["target_encoding"])]
        else:
            self.protein = df["target_encoding"].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        v_p = self.protein[i] if self.cache_proteins else protein_2_embed(self.protein[i])
        return self.drug[i], v_p, self.labels[i]


def make_loader(dataset, batch_size=128, shuffle=False, seed=0, num_workers=0):
    """DataLoader with MPNN collation. Shuffle order depends only on `seed`, so resumed runs repeat it."""
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator,
                           num_workers=num_workers, drop_last=False, collate_fn=mpnn_collate_func)
