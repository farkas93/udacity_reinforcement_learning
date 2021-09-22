import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=48, kernel_size=11)
        self.maxpool = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=48, out_channels=128, kernel_size=5)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3)
        self.conv5 = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3)

        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 500)
        self.fc4 = nn.Linear(500, action_size)
        

    def forward(self, state):
        """Build a network that maps state -> action values."""
        # TODO: make state suitable
        state = state.transpose(1,3)
        x = self.conv1(state)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = torch.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class DuelingQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(state_size, fc1_units)
        self.action_advantages = nn.Linear(fc1_units, action_size)
        self.state_value = nn.Linear(fc1_units, 1)

        self.out_layer = nn.Linear(action_size+1, action_size)
        

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x_1 = F.relu(self.fc1(state))
        x_2 = F.relu(self.fc2(state))
        x_1 = F.relu(self.action_advantages(x_1))
        x_2 = F.relu(self.state_value(x_2))
        x = torch.cat([x_1, x_2], dim=1)
        return self.out_layer(x)        
