# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Di Wu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from ESPnet(https://github.com/espnet/espnet)

from typing import Dict, List, Optional, Tuple
import torch
from networks.model.transformer.encoder import TransformerEncoder, ConformerEncoder
from networks.model.audio_resnet.resnet import ResNet
from networks.model.audio_resnet.res2net import Res2Net
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class EncLSTM(torch.nn.Module):
    """CTC-attention hybrid Encoder-Decoder model"""
    def __init__(
        self,
        model_name: str,
        encoder: torch.nn.Module,
        vocab_size: int,
        drop_rate: 0.25
    ):
        super().__init__()
        if 'former' in model_name:
            model_type = "former"
        elif 'res' in model_name:
            model_type = "resnet"
        elif 'lstm' in model_name:
            model_type = "lstm"
        elif 'gop' in model_name:
            model_type = "gop"
        else:
            model_type = "other"
        self.model_type = model_type

        # note that eos is the same as sos (equivalent ID)
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size

        self.encoder = encoder
        
        self.lstm = torch.nn.LSTM(input_size=encoder.output_size(), hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
        self.linear = torch.nn.Linear(128, vocab_size)
        self.relu = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(drop_rate)

    @torch.jit.ignore(drop=True)
    def forward(
        self,
        speech: torch.Tensor,
        video: torch.Tensor,
        speech_lens: torch.Tensor,
    ) -> Dict[str, Optional[torch.Tensor]]:
        # Check that batch_size is unified
        assert (speech.shape[0] == speech_lens.shape[0] == label.shape[0]), (speech.shape, speech_lens.shape,label.shape)
        # 1. Encoder
        if self.model_type == "former":
            encoder_output = self.encoder(speech, speech_lens)
            if len(encoder_output) == 2:
                encoder_out, encoder_mask = encoder_output
                encoder_length = encoder_mask.sum(dim=2).squeeze(1).cpu()
                packed_input = pack_padded_sequence(encoder_out, encoder_length, batch_first=True, enforce_sorted=False)
                packed_output, (hidden_state, cell_state) = self.lstm(encoder_out)
                encoder_out = hidden_state[-1,:,:]
            elif len(encoder_output) == 3:
                encoder_out, _, _ = encoder_output
                encoder_out = encoder_out.mean(dim=1)
        elif self.model_type == "resnet" or self.model_type == "other":
            encoder_out = self.encoder(speech, speech_lens)
        elif self.model_type == "lstm":
            encoder_out = self.encoder(speech, speech_lens.cpu().type(torch.int64))
        # elif self.model_type == "gop":
        #     encoder_out = self.encoder(text, text_lens) # gop

        encoder_out = self.relu(encoder_out)
        encoder_out = self.drop(encoder_out)
        encoder_out = self.linear(encoder_out)
        return encoder_out

    def _forward_encoder(
        self,
        speech: torch.Tensor,
        speech_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Let's assume B = batch_size
        encoder_output = self.encoder(speech, speech_lens)
        if len(encoder_output) == 2:
            encoder_out, encoder_mask = encoder_output
            encoder_mask = encoder_mask.squeeze(1).unsqueeze(-1)
            encoder_out = encoder_out * encoder_mask
        elif len(encoder_output) == 3:
            encoder_out, _, _ = encoder_output
        encoder_out = encoder_output
        encoder_out = encoder_out.mean(dim=1)
        return encoder_out

    def decode(
        self,
        speech: torch.Tensor,
        speech_lens: torch.Tensor
    ):
        assert speech.shape[0] == speech_lens.shape[0]
        encoder_out = self._forward_encoder(
            speech, speech_lens)
        encoder_out = self.linear(encoder_out)
        results = torch.sigmoid(encoder_out)
        return results

if __name__ == "__main__":
    from networks.model.transformer.encoder import TransformerEncoder
    encoder = TransformerEncoder(80, 512, 3, 2, 1024, 0.1)
    model = DDLModel(40, encoder)
    speech = torch.randn(2, 100, 80)
    speech_lens = torch.tensor([100, 90])
    label = torch.randint(0, 2, (2, 40)).float()
    results = model.forward(speech)