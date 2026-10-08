import torch
import torch.nn as nn

class EdgeModel(nn.Module):
    """
    Lightweight neural network representing a model deployed on a resource-constrained edge device.
    Uses fewer layers and parameters.
    """
    def __init__(self, input_dim=10, num_classes=3):
        super(EdgeModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        out = self.fc2(x)
        return out

def get_quantized_edge_model():
    """
    Simulates loading a quantized edge model. 
    In a real scenario, this would use torch.quantization.
    """
    model = EdgeModel()
    # Mocking quantization for the sake of the structural demo
    model.eval()
    return model
