"""
Federated learning client: holds local training data and trains the global model locally.
"""

import torch

from utils.features import DTIDataset, make_loader
from utils.engine import train_one_epoch
from utils.trainer import cpu_state


class FLClient:
    def __init__(self, client_id, train_df):
        self.client_id = client_id
        self.dataset = DTIDataset(train_df)

    @property
    def num_samples(self):
        return len(self.dataset)

    def train(self, model, global_weights, local_epochs, lr, batch_size, device, seed, num_workers=0):
        """
        Load the global weights into `model`, train for `local_epochs` with a fresh Adam optimizer
        (clients keep no state between rounds), and return (local weights, last-epoch training MSE).
        """
        model.load_state_dict(global_weights)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss = float("nan")
        for e in range(local_epochs):
            loader = make_loader(self.dataset, batch_size, shuffle=True, seed=seed + e, num_workers=num_workers)
            loss = train_one_epoch(model, loader, opt, device)
        return cpu_state(model), loss
