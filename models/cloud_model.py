import torch
import torch.nn as nn

class CloudModel(nn.Module):
    """
    Heavy, accurate neural network representing a model deployed in the cloud.
    Uses more layers and parameters for higher accuracy.
    """
    def __init__(self, input_dim=10, num_classes=3):
        super(CloudModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        out = self.fc3(x)
        return out

def get_cloud_model():
    model = CloudModel()
    model.eval()
    return model
