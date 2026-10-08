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
    
    # Simulate an OOD detector: if the vibration features are wildly noisy, 
    # the AI knows it's out of its comfort zone and artificially spikes uncertainty.
    if features is not None:
        # All 10 features are now vibration metrics (RMS, Peak, Var)
        feature_variance = torch.var(features, dim=1)
        # Add the variance penalty to the entropy
        entropy = entropy + (feature_variance * 0.5)
        
    return entropy

def is_uncertain(entropy_values, threshold=1.0):
    """
    Determines if predictions are uncertain based on an entropy threshold.
    """
    return entropy_values > threshold
