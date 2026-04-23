"""
Centralized baseline training on full DAVIS dataset.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DeepPurpose import DTI as models
from DeepPurpose.utils import generate_config
from DeepPurpose.dataset import data_process
from utils.load_davis import load_davis

def transform_affinity(y, method='pIC50'):
    """Transform affinity values for better training."""
    y = np.array(y)
    
    if method == 'pIC50':
        # Convert nM to M, then to pIC50
        y_transformed = -np.log10(y * 1e-9)
        print(f"Transformed to pIC50: range [{y_transformed.min():.2f}, {y_transformed.max():.2f}]")
        print(f"pIC50 mean: {y_transformed.mean():.2f}, std: {y_transformed.std():.2f}")
    else:
        y_transformed = y
    
    return y_transformed

def run_centralized_train(train_epochs=10):
    """Run centralized training with specified number of epochs."""
    print("=" * 50)
    print(f"Centralized Baseline Training ({train_epochs} Epochs)")
    print("=" * 50)
    
    # Load data
    print("\n1. Loading DAVIS dataset...")
    X_drug, X_target, y = load_davis()
    
    # Check raw affinity statistics
    print(f"\nRaw affinity stats:")
    print(f"  Range: [{y.min():.2f}, {y.max():.2f}]")
    print(f"  Mean: {y.mean():.2f}")
    
    # Transform affinities
    print("\n2. Transforming affinity values to pIC50...")
    y = transform_affinity(y, method='pIC50')
    
    # Convert to lists
    X_drug = X_drug.tolist()
    X_target = X_target.tolist()
    y = y.tolist()
    
    print(f"\n3. Splitting into train/val/test (70/10/20)...")
    train, val, test = data_process(
        X_drug, X_target, y,
        drug_encoding='MPNN',
        target_encoding='CNN',
        split_method='random',
        frac=[0.7, 0.1, 0.2],
        random_seed=42
    )
    
    # Configure model
    print(f"\n4. Configuring MPNN-CNN model ({train_epochs} epochs)...")
    
    config = generate_config(
        drug_encoding='MPNN',
        target_encoding='CNN',
        train_epoch=train_epochs,
        LR=0.001,
        batch_size=128,
        mpnn_hidden_size=128,
        mpnn_depth=3,
        cnn_target_filters=[32, 64, 96],
        cnn_target_kernels=[4, 8, 12]
    )
    
    # Train
    print(f"\n5. Training centralized model ({train_epochs} epochs)...")
    print("   This may take 10-30 minutes...")
    
    model = models.model_initialize(**config)
    model.train(train, val, test)
    
    print("\n" + "=" * 50)
    print(f"✅ Training completed! ({train_epochs} epochs)")
    print("=" * 50)
    
    return model

if __name__ == "__main__":
    print("🔬 Centralized Baseline Training")
    print("=" * 50)
    print("Options:")
    print("  1. Test mode (1 epoch, quick check)")
    print("  2. Full training (10 epochs, recommended)")
    print("  3. Extended training (20 epochs, better results)")
    print("=" * 50)
    
    choice = input("Enter choice (1/2/3): ").strip()
    
    if choice == '1':
        run_centralized_train(train_epochs=1)
    elif choice == '2':
        run_centralized_train(train_epochs=10)
    elif choice == '3':
        run_centralized_train(train_epochs=20)
    else:
        print("Invalid choice. Running test mode (1 epoch)...")
        run_centralized_train(train_epochs=1)