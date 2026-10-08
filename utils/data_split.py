"""
Create non-IID client splits for federated learning.
"""

import numpy as np
from utils.load_davis import load_davis

def assign_drugs_to_clients(drugs, num_clients=5, seed=42):
    """Shuffle unique drugs and assign disjoint drug sets to clients.

    Returns a list of sets, one per client. The last client takes the remainder.
    """
    unique_drugs = np.unique(drugs)
    # RandomState(seed).permutation matches the legacy np.random.seed(seed) + permutation
    shuffled_drugs = np.random.RandomState(seed).permutation(unique_drugs)

    drugs_per_client = len(unique_drugs) // num_clients

    client_drug_sets = []
    for i in range(num_clients):
        start = i * drugs_per_client
        end = (i + 1) * drugs_per_client if i < num_clients - 1 else len(unique_drugs)
        client_drug_sets.append(set(shuffled_drugs[start:end]))

    return client_drug_sets

def create_non_iid_clients(num_clients=5):
    """Split data into non-IID clients (each gets different drugs)."""

    print("\nLoading data...")
    drugs, targets, y = load_davis()

    print(f"Total unique drugs: {len(np.unique(drugs))}")

    clients_data = []
    for i, client_drugs in enumerate(assign_drugs_to_clients(drugs, num_clients)):
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