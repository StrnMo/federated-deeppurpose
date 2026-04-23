"""
Create non-IID client splits for federated learning.
"""

import numpy as np
from utils.load_davis import load_davis

def create_non_iid_clients(num_clients=5):
    """Split data into non-IID clients (each gets different drugs)."""
    
    print("\nLoading data...")
    drugs, targets, y = load_davis()
    
    # Get unique drugs
    unique_drugs = np.unique(drugs)
    print(f"Total unique drugs: {len(unique_drugs)}")
    
    # Shuffle and assign drugs to clients
    np.random.seed(42)
    shuffled_drugs = np.random.permutation(unique_drugs)
    
    drugs_per_client = len(unique_drugs) // num_clients
    
    clients_data = []
    for i in range(num_clients):
        start = i * drugs_per_client
        end = (i + 1) * drugs_per_client if i < num_clients - 1 else len(unique_drugs)
        client_drugs = set(shuffled_drugs[start:end])
        
        mask = np.isin(drugs, list(client_drugs))
        clients_data.append((drugs[mask], targets[mask], y[mask]))
        print(f"Client {i}: {np.sum(mask)} samples, {len(client_drugs)} unique drugs")
    
    return clients_data

def get_full_data():
    """Return full dataset for centralized baseline."""
    return load_davis()

if __name__ == "__main__":
    print("=" * 50)
    print("Creating Non-IID Client Splits")
    print("=" * 50)
    clients = create_non_iid_clients(num_clients=5)
    print(f"\nCreated {len(clients)} clients successfully!")
    print("=" * 50)