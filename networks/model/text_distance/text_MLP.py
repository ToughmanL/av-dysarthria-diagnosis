import torch
import torch.nn as nn


class MLPNet(nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim, droprate=0.3):
    super(MLPNet,self).__init__()
    self.fc1 = nn.Linear(input_dim, hidden_dim*2)
    self.fc2 = nn.Linear(hidden_dim*2, hidden_dim // 2)
    self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
    self.relu = nn.ReLU()
    self.softmax = nn.Softmax(dim=1)
    self.dropout = nn.Dropout(droprate)
  
  def forward(self, 
        speech: torch.Tensor,
        video: torch.Tensor,
        speech_lens: torch.Tensor
    ):
    x = speech.mean(dim=1)
    x = x.view(x.size(0), -1)
    x = self.softmax(x)
    x = self.relu(self.fc1(x))
    x = self.dropout(x)
    x = self.relu(self.fc2(x))
    x = self.dropout(x)
    x = self.fc3(x)
    return x


class DistanceNet(nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim, droprate=0.0):
    super(DistanceNet,self).__init__()
    self.fc1 = nn.Linear(input_dim, hidden_dim*2)
    # self.fc2 = nn.Linear(hidden_dim*2, hidden_dim)
    self.fc3 = nn.Linear(hidden_dim*2, output_dim)
    self.softmax = nn.Softmax(dim=2)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(droprate)
  
  # normalize the input by each line
  def normalize(self, x):
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True)
    return (x - mean) / std

  # GOP
  def gop_minus(self, speech, text):
    _, indeces = torch.max(text, dim=2, keepdim=True)
    max_value = torch.gather(speech, 2, indeces)
    result = speech - max_value
    return result

  def gop_log(self, speech, text):
    _, indeces = torch.max(text, dim=2, keepdim=True)
    max_value = torch.gather(speech, 2, indeces)
    result = torch.log(speech / max_value)
    return result
    

  def forward(self, 
        speech: torch.Tensor,
        video: torch.Tensor,
        speech_lens: torch.Tensor
    ):
    speech = self.softmax(speech)
    # text = text.mean(dim=1)
    # speech = speech.mean(dim=1)
    # out = text + speech
    out = self.gop_log(speech, text)
    out = out.mean(dim=1)
    out = self.dropout(self.relu(self.fc1(out)))
    # out = self.dropout(self.relu(self.fc2(out)))
    out = self.fc3(out)
    return out


class AttensionNet(nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim, droprate=0.3):
    super(AttensionNet,self).__init__()
    # Attention
    self.att = nn.MultiheadAttention(input_dim, 1, dropout=0.2, batch_first=True)
    self.fc1 = nn.Linear(input_dim, hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, output_dim)
    self.avg_pool = nn.AdaptiveAvgPool1d(1)
    self.softmax = nn.Softmax(dim=1)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(droprate)
  
  def forward(self, 
        speech: torch.Tensor,
        video: torch.Tensor,
        speech_lens: torch.Tensor
    ):
    out, _ = self.att(text, speech, speech)
    out = self.avg_pool(out.permute(0, 2, 1)).squeeze(-1)
    out = self.relu(self.fc1(out))
    out = self.dropout(out)
    out = self.fc2(out)
    return out


if __name__ == '__main__':
  speech = torch.randn(64, 12, 512)
  video = torch.randn(64, 12, 512)
  text = torch.randn(64, 12, 512)
  AN = DistanceNet(512, 64, 2)
  out = AN(speech, video, text, None, None, None)
  print(out.size())