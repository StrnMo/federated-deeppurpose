"""
Federated learning server: holds the global weights and aggregates client updates with FedAvg.
"""

import torch


def fedavg_aggregate(client_weights, num_samples):
    """Sample-weighted average of client state_dicts: sum_k (n_k / n) * w_k."""
    total = float(sum(num_samples))
    coeffs = [n / total for n in num_samples]
    avg = {}
    for key, ref in client_weights[0].items():
        if torch.is_floating_point(ref):
            avg[key] = sum(c * w[key].double() for c, w in zip(coeffs, client_weights)).to(ref.dtype)
        else:
            # integer buffers (none in MPNN-CNN) are taken from the first client
            avg[key] = ref.clone()
    return avg


class FLServer:
    def __init__(self, global_weights):
        self.global_weights = global_weights

    def get_global_weights(self):
        return self.global_weights

    def aggregate(self, client_weights, num_samples):
        self.global_weights = fedavg_aggregate(client_weights, num_samples)
        return self.global_weights
