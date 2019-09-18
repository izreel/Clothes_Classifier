import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class ClothesClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(784, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, 32)
        self.linear5 = nn.Linear(32, 16)
        
        self.dropout = nn.Dropout(p= 0.2)
        self.output = nn.Linear(16, 10)

    def forward(self, x):
        x = x.view(x.shape[0], -1)

        x = self.dropout(F.relu(self.linear1(x)))
        x = self.dropout(F.relu(self.linear2(x)))
        x = self.dropout(F.relu(self.linear3(x)))
        x = self.dropout(F.relu(self.linear4(x)))
        x = self.dropout(F.relu(self.linear5(x)))
        
        #creates class probabilities. To get scores, take this line out
        x = F.log_softmax(self.output(x), dim= 1)

        return x
        