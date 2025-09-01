import torch


class Identity(torch.nn.Module):
    def forward(self, input):
        return input


class SegmentConsensus(torch.nn.Module):

    def __init__(self, consensus_type, dim=1):
        super(SegmentConsensus, self).__init__()
        self.consensus_type = consensus_type
        self.dim = dim
        self.shape = None

    def forward(self, input_tensor):
        self.shape = input_tensor.shape
        if self.consensus_type == 'avg':
            output = input_tensor.mean(dim=self.dim, keepdim=True)
        elif self.consensus_type == 'sum':
            output = input_tensor.sum(dim=self.dim, keepdim=True)
        elif self.consensus_type == 'identity':
            output = input_tensor
        else:
            output = None

        return output


class ConsensusModule(torch.nn.Module):

    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim

    def forward(self, input):
        return SegmentConsensus(self.consensus_type, self.dim)(input)


class ModelFusion(torch.nn.Module):
    def __init__(self, fusion_type, audio_dim, dim=1):
        super(ModelFusion, self).__init__()
        self.fusion_type = fusion_type
        self.dim = dim
        self.crossatt = torch.nn.MultiheadAttention(audio_dim, 2, dropout=0.03)
    
    def forward(self, audio_data, video_data):
        if self.fusion_type == 'concat':
            output = torch.cat((audio_data, video_data), dim=self.dim)
        elif self.fusion_type == 'sum':
            output = audio_data + video_data
        elif self.fusion_type == 'mul':
            output = audio_data * video_data
        elif self.fusion_type == 'crossatt':
            audio_data = torch.unsqueeze(audio_data, 0)
            video_data = torch.unsqueeze(video_data, 0)
            output, _ = self.crossatt(audio_data, video_data, video_data)
            output = torch.squeeze(output, 0)
        else:
            output = None
        return output
    
    def output_dim(self):
        return self.dim