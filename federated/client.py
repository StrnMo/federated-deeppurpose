"""
Federated learning client: holds local training data and trains the global model locally.
"""

import copy
import torch

from utils.features import DTIDataset, make_loader
from utils.engine import train_one_epoch
from utils.trainer import cpu_state


class FLClient:
    def __init__(self, client_id, train_df, reset_optimizer=False):
        self.client_id = client_id
        self.dataset = DTIDataset(train_df)
        # Adam state carried from round to round (None until the first round)
        self.reset_optimizer = reset_optimizer
        self.optimizer_state = None

    @property
    def num_samples(self):
        return len(self.dataset)

    @property
    def label_mean(self):
        """Mean training label: the only label statistic shared with the server (for bias init)."""
        return float(self.dataset.labels.mean())

    def train(self, model, global_weights, local_epochs, lr, batch_size, device, seed, num_workers=0):
        """
        Load the global weights into `model`, train for `local_epochs` and return
        (local weights, last-epoch training MSE).

        The client's Adam state is kept between rounds. A fresh Adam takes a full-size first step on
        every weight, which shifted all predictions up or down each round (see docs/NOTES.md).
        With reset_optimizer=True a fresh Adam is used every round instead.
        """
        model.load_state_dict(global_weights)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        if self.optimizer_state is not None and not self.reset_optimizer:
            opt.load_state_dict(self.optimizer_state)
        loss = float("nan")
        for e in range(local_epochs):
            loader = make_loader(self.dataset, batch_size, shuffle=True, seed=seed + e, num_workers=num_workers)
            loss = train_one_epoch(model, loader, opt, device)
        if not self.reset_optimizer:
            self.optimizer_state = copy.deepcopy(opt.state_dict())
        return cpu_state(model), loss
