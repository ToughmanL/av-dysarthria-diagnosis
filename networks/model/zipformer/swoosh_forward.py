#!/usr/bin/env python3
#
# Copyright      2023  Xiaomi Corporation (authors: Fangjun Kuang, Zengwei Yao)
#
# See ../../../LICENSE for clarification regarding multiple authors
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

# To run this single test, use
#
#  ctest --verbose -R swoosh_forward_deriv_test_py

import torch


# simple version of SwooshL that does not redefine the backprop, used in
# ActivationDropoutAndLinearFunction.
def SwooshLForward(x: torch.Tensor):
    x_offset = x - 4.0
    log_sum = (1.0 + x_offset.exp()).log().to(x.dtype)
    log_sum = torch.where(log_sum == float("inf"), x_offset, log_sum)
    return log_sum - 0.08 * x - 0.035


def SwooshLForwardAndDeriv(x: torch.Tensor):
    x_offset = x - 4
    # swooshl(x) = log(1 + exp(x-4)) - 0.08 * x - 0.035
    # x_deriv = -0.08 + exp(x-4) / (1 + exp(x-4))
    #         = -0.08 + (1 -  1 / (1 + exp(x-4)))
    #         = 0.92 - 1 / (1 + exp(x-4))
    # note: 1 + exp(x_offset) might be infinity,
    # but 1 / (1 + exp(x_offset)) will be 0 in that case.
    # This is partly why we rearranged the expression above,
    # to avoid infinity / infinity = nan.
    denom = 1 + x_offset.exp()
    inv_denom = (1.0 / denom).to(x.dtype)
    deriv = 0.92 - inv_denom
    log_denom = denom.log().to(x.dtype)
    log_denom = torch.where(log_denom == float("inf"), x_offset, log_denom)
    y = log_denom - 0.08 * x - 0.035
    return y, deriv


# simple version of SwooshR that does not redefine the backprop, used in
# ActivationDropoutAndLinearFunction.
def SwooshRForward(x: torch.Tensor):
    x_offset = x - 1.0
    log_sum = (1.0 + x_offset.exp()).log().to(x.dtype)
    log_sum = torch.where(log_sum == float("inf"), x_offset, log_sum)
    return log_sum - 0.08 * x - 0.313261687


def SwooshRForwardAndDeriv(x: torch.Tensor):
    x_offset = x - 1
    # swooshr(x) = log(1 + exp(x-1)) - 0.08 * x - 0.313261687
    # x_deriv = -0.08 + exp(x-1) / (1 + exp(x-1))
    #         = -0.08 + (1 -  1 / (1 + exp(x-1)))
    #         = 0.92 - 1 / (1 + exp(x-1))
    # note: 1 + exp(x_offset) might be infinity,
    # but 1 / (1 + exp(x_offset)) will be 0 in that case.
    # This is partly why we rearranged the expression above,
    # to avoid infinity / infinity = nan.
    denom = 1 + x_offset.exp()
    inv_denom = (1.0 / denom).to(x.dtype)
    deriv = 0.92 - inv_denom
    log_denom = denom.log().to(x.dtype)
    log_denom = torch.where(log_denom == float("inf"), x_offset, log_denom)
    y = log_denom - 0.08 * x - 0.313261687
    return y, deriv

