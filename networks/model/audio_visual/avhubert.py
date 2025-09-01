import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class AHubertLSTM(torch.nn.Module):
    def __init__(self, input_dim=768, audio_dim=64, out_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=audio_dim*2, num_layers=2, batch_first=True)
        self.fc_a = nn.Linear(audio_dim*2, audio_dim)
        self.fc = nn.Linear(audio_dim, out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, 
        speech: torch.Tensor,
        video: torch.Tensor,
        speech_lens: torch.Tensor
    ):
        speech_lens = speech_lens.cpu().tolist()
        packed_input = pack_padded_sequence(speech, speech_lens, batch_first=True, enforce_sorted=False)
        packed_out, (hidden, cell_state) = self.lstm(packed_input)
        output = self.fc_a(hidden[-1,:,:])
        output = self.dropout(self.relu(output))
        output = self.fc(output)
        return output


class AVHubertLSTM(torch.nn.Module):
    def __init__(self, a_input_dim=768, v_input_dim=768, audio_dim=64, video_dim=64, out_dim=2):
        super().__init__()
        self.lstm_a = nn.LSTM(input_size=a_input_dim, hidden_size=audio_dim*2, num_layers=2, batch_first=True)
        self.lstm_v = nn.LSTM(input_size=v_input_dim, hidden_size=video_dim*2, num_layers=2, batch_first=True)
        self.fc_a = nn.Linear(audio_dim*2, audio_dim)
        self.fc_v = nn.Linear(video_dim*2, video_dim)
        self.fc_av = nn.Linear(audio_dim+video_dim, out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, 
        speech: torch.Tensor,
        video: torch.Tensor,
        speech_lens: torch.Tensor,
        video_lens: torch.Tensor
    ):
        speech_lens = speech_lens.cpu().tolist()
        a_packed_input = pack_padded_sequence(speech, speech_lens, batch_first=True, enforce_sorted=False)
        a_packed_out, (a_hidden, cell_state) = self.lstm_a(a_packed_input)
        a_output = self.fc_a(a_hidden[-1,:,:])
        a_output = self.dropout(self.relu(a_output))
        
        video_lens = video_lens.cpu().tolist()
        v_packed_input = pack_padded_sequence(video, video_lens, batch_first=True, enforce_sorted=False)
        v_packed_out, (v_hidden, cell_state) = self.lstm_v(v_packed_input)
        v_output = self.fc_v(v_hidden[-1,:,:])
        v_output = self.dropout(self.relu(v_output))

        av_out = torch.cat((a_output, v_output), 1)
        av_out = self.fc_av(av_out)
        return av_out


class AVHubertMLP(torch.nn.Module):
    def __init__(self, a_input_dim=768, v_input_dim=768, audio_dim=64, video_dim=64, out_dim=2):
        super().__init__()
        self.fc_a = nn.Linear(a_input_dim, audio_dim)
        self.fc_v = nn.Linear(v_input_dim, video_dim)
        self.fc_av = nn.Linear(audio_dim+video_dim, out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.02)

    def forward(self, 
        speech: torch.Tensor,
        video: torch.Tensor,
        speech_lens: torch.Tensor
    ):
        speech, speech_index = torch.max(speech, 1)
        a_output = self.fc_a(speech)
        a_output = self.dropout(self.relu(a_output))
        
        video, video_index = torch.max(video, 1)
        v_output = self.fc_v(video)
        v_output = self.dropout(self.relu(v_output))

        av_out = torch.cat((a_output, v_output), 1)
        av_out = self.fc_av(av_out)
        return av_out


class AHubertMLP(torch.nn.Module):
    def __init__(self, input_dim=768, audio_dim=64, out_dim=2):
        super().__init__()
        self.fc_1 = nn.Linear(input_dim, audio_dim)
        self.fc_2 = nn.Linear(audio_dim, out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, 
        speech: torch.Tensor,
        video: torch.Tensor,
        speech_lens: torch.Tensor
    ):
        # speech, index = torch.max(speech, 1)
        speech = torch.mean(speech, 1)
        output = self.fc_1(speech)
        output = self.dropout(self.relu(output))
        output = self.fc_2(output)
        return output


class VHubertMLP(torch.nn.Module):
    def __init__(self, input_dim=768, audio_dim=64, out_dim=2):
        super().__init__()
        self.fc_1 = nn.Linear(input_dim, audio_dim)
        self.fc_2 = nn.Linear(audio_dim, out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, 
        speech: torch.Tensor,
        video: torch.Tensor,
        speech_lens: torch.Tensor
    ):
        video = torch.mean(video, 1)
        output = self.fc_1(video)
        output = self.dropout(self.relu(output))
        output = self.fc_2(output)
        return output