import torch
from transformers import Wav2Vec2FeatureExtractor, HubertModel, Wav2Vec2Model


class Wav2vecNet(torch.nn.Module):
  def __init__(self, input_dim, hidden_dim1, output_dim, dropout=0.3):
    super(Wav2vecNet, self).__init__()
    self.model = Wav2Vec2Model.from_pretrained("exp/wav2vec_hubert/wav2vec2-base/chinese-wav2vec2-base")
    self.linear1 = torch.nn.Linear(input_dim, hidden_dim1)
    self.fc = torch.nn.Linear(hidden_dim1, output_dim)
    self.batchnorm = torch.nn.BatchNorm1d(hidden_dim1)
    self.relu = torch.nn.ReLU()
    self.dropout = torch.nn.Dropout(dropout)

  def forward(self,
        speech: torch.Tensor,
        video: torch.Tensor,
        text: torch.Tensor,
        speech_lens: torch.Tensor,
        text_lens: torch.Tensor,
        label: torch.Tensor
    ):
    hidden_states = self.model(speech, output_hidden_states=True).hidden_states
    hidden_state = hidden_states[12]
    x = torch.mean(hidden_state, dim=1)
    x = self.relu(self.linear1(x))
    x = self.batchnorm(x)
    x = self.dropout(x)
    x = self.fc(x)
    return x
    
