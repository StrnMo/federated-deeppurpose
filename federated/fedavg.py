"""
Main FedAvg training loop.
"""

import copy
import numpy as np
from federated.client import FLClient
from federated.server import FLServer

def run_fedavg(clients_data, global_model, num_rounds=10, local_epochs=5):
    """
    Execute FedAvg algorithm.
    
    Args:
        clients_data: list of (X_drug, X_target, y) tuples for each client
        global_model: initialized DeepPurpose model
        num_rounds: number of communication rounds
        local_epochs: number of local training epochs per round
    
    Returns:
        trained global model, history
    """
    print("=" * 50)
    print("Starting Federated Learning (FedAvg)")
    print("=" * 50)
    print(f"Number of clients: {len(clients_data)}")
    print(f"Number of rounds: {num_rounds}")
    print(f"Local epochs per round: {local_epochs}")
    print("=" * 50)
    
    # Initialize clients
    clients = []
    for i, (X_drug, X_target, y) in enumerate(clients_data):
        client = FLClient(i, X_drug, X_target, y)
        clients.append(client)
    
    # Initialize server
    server = FLServer(global_model)
    
    history = {
        'rounds': [],
        'client_weights': []
    }
    
    for round_idx in range(num_rounds):
        print(f"\n--- Round {round_idx + 1}/{num_rounds} ---")
        
        # Distribute global weights to all clients
        global_weights = server.get_global_weights()
        client_weights = []
        
        for client in clients:
            client.set_weights(copy.deepcopy(global_weights))
            updated_weights = client.train(local_epochs=local_epochs)
            client_weights.append(updated_weights)
        
        # Aggregate weights
        server.aggregate_fedavg(client_weights)
        
        history['rounds'].append(round_idx + 1)
        history['client_weights'].append(len(client_weights))
        
        print(f"  Round {round_idx + 1} completed")
    
    print("\n" + "=" * 50)
    print("Federated Learning completed!")
    print("=" * 50)
    
    return global_model, history