"""
Federated learning server: aggregates client weights.
"""

import numpy as np

class FLServer:
    def __init__(self, global_model):
        self.global_model = global_model
        self.global_weights = global_model.model.get_weights()
        self.round_history = []
        
    def aggregate_fedavg(self, client_weights):
        """
        FedAvg: weighted average of client weights.
        All clients have equal weight.
        """
        if not client_weights:
            return self.global_weights
        
        num_clients = len(client_weights)
        avg_weights = []
        
        # Average each layer
        for layer_idx in range(len(client_weights[0])):
            # Convert to float32 for averaging
            layer_avg = np.zeros_like(client_weights[0][layer_idx], dtype=np.float32)
            for client_w in client_weights:
                layer_avg += client_w[layer_idx].astype(np.float32)
            layer_avg /= num_clients
            avg_weights.append(layer_avg)
        
        # Update global model
        self.global_model.model.set_weights(avg_weights)
        self.global_weights = avg_weights
        
        return avg_weights
    
    def get_global_weights(self):
        return self.global_weights