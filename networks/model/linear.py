import torch
import torch.nn as nn
import torch.optim as optim


class LinearNet(nn.Module):
  def __init__(self, input_dim, hidden_dim1, output_dim, dropout):
    super(LinearNet, self).__init__()
    self.fc1 = nn.Linear(input_dim, hidden_dim1)
    self.fc2 = nn.Linear(hidden_dim1, output_dim)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(dropout)

  def forward(self,
      speech: torch.Tensor,
      video: torch.Tensor,
      speech_lens: torch.Tensor,
    ):
    # time dim mean
    x = speech.mean(dim=1)
    # x = speech.max(dim=1)[0]
    x = self.relu(self.fc1(x))
    x = self.dropout(x)
    x = self.fc2(x)
    return x


class LinearNet_1(nn.Module):
  def __init__(self, input_dim, hidden_dim1, output_dim, dropout):
    super(LinearNet_1, self).__init__()
    self.fc1 = nn.Linear(input_dim, hidden_dim1)
    self.fc2 = nn.Linear(hidden_dim1, output_dim)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(dropout)

  def forward(self,
      speech: torch.Tensor,
      video: torch.Tensor,
      speech_lens: torch.Tensor,
    ):
    x = speech # ivector
    x = self.relu(self.fc1(x))
    x = self.dropout(x)
    x = self.fc2(x)
    return x


# Define the neural network class
class LinearNet2(nn.Module):
  def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, num_frms, dropout):
    super(LinearNet2, self).__init__()
    # self.fc1 = nn.Linear(input_dim*num_frms, hidden_dim1)
    self.fc1 = nn.Linear(input_dim, hidden_dim1)
    self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
    self.fc3 = nn.Linear(hidden_dim2, output_dim)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(dropout)

  def forward(self,
      speech: torch.Tensor,
      video: torch.Tensor,
      speech_lens: torch.Tensor
    ):
    x = speech.view(speech.size(0), -1) # flatten
    x = x.view(x.size(0), -1)
    x = self.relu(self.fc1(x))
    x = self.dropout(x)
    x = self.relu(self.fc2(x))
    x = self.dropout(x)
    x = self.fc3(x)
    return x