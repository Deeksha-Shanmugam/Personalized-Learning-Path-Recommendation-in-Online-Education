
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),  # Wider first hidden layer
            nn.ReLU(),
            nn.Dropout(0.3),             # Add dropout for regularization
            nn.Linear(64, 32),           # Second hidden layer
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 4)             # Output layer (4 classes)
        )

    def forward(self, x):
        return self.fc(x)



'''
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 4),  # 4 classes for final_result
        )

    def forward(self, x):
        return self.fc(x)

'''
