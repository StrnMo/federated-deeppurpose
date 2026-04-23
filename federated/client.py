"""
Federated learning client: trains on local data.
"""

import copy
from DeepPurpose import DTI as models
from DeepPurpose.utils import generate_config
from DeepPurpose.dataset import data_process

class FLClient:
    def __init__(self, client_id, X_drug, X_target, y, drug_encoding='MPNN', target_encoding='CNN'):
        self.client_id = client_id
        self.X_drug = X_drug
        self.X_target = X_target
        self.y = y
        self.drug_encoding = drug_encoding
        self.target_encoding = target_encoding
        self.model = None
        self.training_history = []
        
    def _init_model(self):
        """Initialize a fresh model for this client."""
        config = generate_config(
            drug_encoding=self.drug_encoding,
            target_encoding=self.target_encoding,
            train_epoch=1,  # Will train for multiple epochs in each round
            LR=0.001,
            batch_size=128,
            mpnn_hidden_size=128,
            mpnn_depth=3,
            cnn_target_filters=[32, 64, 96],
            cnn_target_kernels=[4, 8, 12]
        )
        self.model = models.model_initialize(**config)
    
    def set_weights(self, weights):
        """Set model weights from global model."""
        if self.model is None:
            self._init_model()
        # DeepPurpose uses model.model for the underlying Keras model
        self.model.model.set_weights(weights)
    
    def get_weights(self):
        """Get current model weights."""
        if self.model is None:
            return None
        return self.model.model.get_weights()
    
    def train(self, local_epochs=5):
        """Train locally for specified epochs."""
        if self.model is None:
            self._init_model()
        
        # Prepare data
        X_drug = self.X_drug.tolist() if hasattr(self.X_drug, 'tolist') else self.X_drug
        X_target = self.X_target.tolist() if hasattr(self.X_target, 'tolist') else self.X_target
        y = self.y.tolist() if hasattr(self.y, 'tolist') else self.y
        
        # Split into train/val
        train, val, _ = data_process(
            X_drug, X_target, y,
            drug_encoding=self.drug_encoding,
            target_encoding=self.target_encoding,
            split_method='random',
            frac=[0.8, 0.2, 0.0],
            random_seed=42
        )
        
        # Train for local_epochs
        print(f"    Client {self.client_id}: Training for {local_epochs} epochs...")
        
        # DeepPurpose's train method trains for the number of epochs in config
        original_epochs = self.model.config['train_epoch']
        self.model.config['train_epoch'] = local_epochs
        
        # Train the model
        self.model.train(train, val, None, verbose=False)
        
        # Restore original config
        self.model.config['train_epoch'] = original_epochs
        
        return self.get_weights()