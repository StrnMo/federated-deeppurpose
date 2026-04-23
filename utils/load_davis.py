"""
Load DAVIS dataset from local ZIP file
"""

import zipfile
import numpy as np
import os
import ast

def load_davis(zip_path="data/DAVIS.zip"):
    """Load DAVIS dataset from local ZIP file."""
    
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"File not found: {zip_path}")
    
    print(f"Loading DAVIS from: {zip_path}")
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Read affinity.txt
        with zf.open('DAVIS/affinity.txt') as f:
            content = f.read().decode('utf-8')
            affinities = []
            for line in content.strip().split('\n'):
                if line.strip():
                    numbers = [float(x) for x in line.strip().split()]
                    affinities.extend(numbers)
        
        # Read SMILES.txt
        with zf.open('DAVIS/SMILES.txt') as f:
            smiles_content = f.read().decode('utf-8')
            smiles_dict = ast.literal_eval(smiles_content)
        
        # Read target_seq.txt
        with zf.open('DAVIS/target_seq.txt') as f:
            target_content = f.read().decode('utf-8')
            target_dict = ast.literal_eval(target_content)
    
    # Get IDs
    compound_ids = list(smiles_dict.keys())
    target_ids = list(target_dict.keys())
    
    # Convert flat affinities list to matrix
    n_compounds = len(compound_ids)
    n_targets = len(target_ids)
    expected_size = n_compounds * n_targets
    
    if len(affinities) == expected_size:
        print(f"Reshaping affinities to {n_compounds}×{n_targets} matrix")
        affinity_matrix = np.array(affinities).reshape(n_compounds, n_targets)
    else:
        raise ValueError(f"Expected {expected_size} affinities, got {len(affinities)}")
    
    # Create flat arrays (CORRECTED)
    drugs = []
    targets = []
    y = []
    
    for i, comp_id in enumerate(compound_ids):
        for j, target_id in enumerate(target_ids):
            drugs.append(smiles_dict[comp_id])
            targets.append(target_dict[target_id])
            y.append(affinity_matrix[i, j])
    
    # Convert to numpy arrays (OUTSIDE the loops - CORRECTED)
    drugs = np.array(drugs)
    targets = np.array(targets)
    y = np.array(y, dtype=np.float32)
    
    print(f"\nLoaded {len(y)} drug-target pairs")
    print(f"Unique drugs: {len(np.unique(drugs))}")
    print(f"Unique targets: {len(np.unique(targets))}")
    print(f"Affinity - Min: {y.min():.2f}, Max: {y.max():.2f}, Mean: {y.mean():.2f}")
    
    return drugs, targets, y

if __name__ == "__main__":
    print("=" * 50)
    drugs, targets, y = load_davis()
    print("=" * 50)
    print("SUCCESS! Loader works.")
    print(f"\nSample drug: {drugs[0][:80]}...")
    print(f"\nSample target (first 80 chars): {targets[0][:80]}...")
    print(f"\nSample affinity: {y[0]}")
    print(f"\nArray shapes:")
    print(f"  Drugs: {drugs.shape}")
    print(f"  Targets: {targets.shape}")
    print(f"  Affinities: {y.shape}")
    print("=" * 50)