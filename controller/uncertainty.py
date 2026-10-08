import torch
import torch.nn.functional as F
import numpy as np

def calculate_entropy(logits, features=None):
    """
    Calculates predictive uncertainty using Shannon Entropy.
    In this demo, we also add a penalty for Out-of-Distribution (OOD) 
    noisy data (simulating the bearing wear).
    """
    probs = F.softmax(logits, dim=1)
    epsilon = 1e-10
    entropy = -torch.sum(probs * torch.log(probs + epsilon), dim=1)
    
    # Simulate an OOD detector: if the vibration features (first 8) are wildly noisy, 
    # the AI knows it's out of its comfort zone and artificially spikes uncertainty.
    if features is not None:
        # We only check the variance of the vibration sensors (indices 0 to 7),
        # ignoring the RPM (2000) and Load (6000) which would artificially skew the variance!
        vibration_features = features[:, :8]
        feature_variance = torch.var(vibration_features, dim=1)
        # Add the variance penalty to the entropy
        entropy = entropy + (feature_variance * 0.5)
        
    return entropy

def is_uncertain(entropy_values, threshold=1.0):
    """
    Determines if predictions are uncertain based on an entropy threshold.
    """
    return entropy_values > threshold
