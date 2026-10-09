import torch
import torch.nn.functional as F
import numpy as np

def calculate_entropy(logits, features=None, temperature=1.0):
    """
    Calculates predictive uncertainty using Temperature-Scaled Shannon Entropy (Guo et al., 2017)
    combined with Distance-Aware OOD scoring (van Amersfoort et al., 2020).
    
    Parameters:
    -----------
    logits : torch.Tensor
        Raw output logits from the model [N, C].
    features : torch.Tensor, optional
        Input feature representations [N, D] used to detect distributional shift.
    temperature : float, default=1.0
        Post-hoc calibration temperature (T > 1 softens overconfident probabilities).
        
    Returns:
    --------
    torch.Tensor : Composite uncertainty score for each sample.
    """
    # 1. Temperature-scaled probabilities for calibrated softmax (Guo et al., ICML 2017)
    scaled_logits = logits / max(temperature, 1e-4)
    probs = F.softmax(scaled_logits, dim=1)
    epsilon = 1e-10
    entropy = -torch.sum(probs * torch.log(probs + epsilon), dim=1)
    
    # 2. Distance-Aware OOD penalty (approximating Mahalanobis / Epistemic variance)
    # When mechanical vibration shifts to heavy-tailed non-Gaussian regimes,
    # the feature dispersion spikes, indicating epistemic drift.
    if features is not None:
        feature_variance = torch.var(features, dim=1)
        entropy = entropy + (feature_variance * 0.5)
        
    return entropy

def calculate_conformal_score(logits, temperature=1.0):
    """
    Computes the Conformal Prediction non-conformity score (Vovk et al. 2005, Angelopoulos & Bates 2021).
    Score s(x) = 1 - max_k p_k(x) (least confidence non-conformity).
    
    Samples where s(x) > q_hat (conformal quantile) cannot be guaranteed at (1 - alpha)
    confidence level and are offloaded to the Cloud.
    """
    scaled_logits = logits / max(temperature, 1e-4)
    probs = F.softmax(scaled_logits, dim=1)
    max_probs, _ = torch.max(probs, dim=1)
    non_conformity = 1.0 - max_probs
    return non_conformity

def is_uncertain(uncertainty_values, threshold=1.0):
    """
    Rejection / Offloading function g(x):
    Determines if predictions exceed the risk-controlled threshold.
    Returns boolean tensor where True = offload to Cloud, False = accept Edge inference.
    """
    return uncertainty_values > threshold

