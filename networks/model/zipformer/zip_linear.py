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
from networks.model.ops.mask import k2_make_pad_mask
from networks.model.zipformer.encoder import Zipformer2
from networks.model.zipformer.scaling import ScheduledFloat

class ZipLinear(torch.nn.Module):
    """CTC-attention hybrid Encoder-Decoder model"""
    def __init__(
        self,
        encoder_dim: 384,
        encoder: Zipformer2,
        vocab_size: int,
    ):
        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.encoder = encoder
        self.linear = torch.nn.Linear(encoder_dim, vocab_size)
        self.relu = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(0.2)

    @torch.jit.ignore(drop=True)
    def forward(
        self,
        speech: torch.Tensor,
        video: torch.Tensor,
        speech_lengths: torch.Tensor
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Frontend + Encoder + Calc loss
        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
        """

        # Check that batch_size is unified
        # assert (speech.shape[0] == speech_lengths.shape[0]), (speech.shape, speech_lengths.shape)
        # 1. Encoder
        encoder_out, encoder_out_lens = self.encoder(speech, speech_lengths)
        encoder_out = encoder_out.permute(1, 0, 2)

        encoder_out = encoder_out.mean(dim=1)
        encoder_out = self.relu(encoder_out)
        encoder_out = self.linear(encoder_out)
        return encoder_out

    def _forward_encoder(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Let's assume B = batch_size
        encoder_output = self.encoder(speech, speech_lengths)
        if len(encoder_output) == 2:
            encoder_out, encoder_mask = encoder_output
            encoder_mask = encoder_mask.squeeze(1).unsqueeze(-1)
            encoder_out = encoder_out * encoder_mask
        else:
            encoder_out, _, _ = encoder_output
        encoder_out = encoder_out.mean(dim=1)
        return encoder_out

    def decode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor
    ):
        assert speech.shape[0] == speech_lengths.shape[0]
        encoder_out = self._forward_encoder(
            speech, speech_lengths)
        encoder_out = self.linear(encoder_out)
        results = torch.sigmoid(encoder_out)
        return results
