# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
Vendored from NVIDIA cosmos-predict2 (Apache 2.0), trimmed.
'''

from typing import Optional

import torch.nn as nn

from .video2world_dit import MinimalV1LVGDiT


class Mlp(nn.Module):
    def __init__(
            self,
            in_features: int,
            hidden_features: Optional[int] = None,
            out_features: Optional[int] = None,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0.0,
        ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.activation = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ActionConditionedMinimalV1LVGDiT(MinimalV1LVGDiT):
    # The action-conditioned forward is not vendored; AnyViewDiT (action_dim=0) overrides forward.
    def __init__(self, *args, **kwargs):
        assert 'action_dim' in kwargs, "action_dim must be provided"
        action_dim = kwargs['action_dim']
        del kwargs['action_dim']
        super().__init__(*args, **kwargs)

        self.action_dim = action_dim

        # Workaround: When action_dim == 0, avoid creating action embedders to prevent optimizer state mismatches upon training resume.
        # Without this check, optimizer state dicts may not match if the model is resumed with different action_dim values,
        # leading to errors or unexpected behavior. See issue tracker for details if available.
        if self.action_dim > 0:
            self.action_embedder_B_D = Mlp(
                in_features=self.action_dim,
                hidden_features=self.model_channels * 4,
                out_features=self.model_channels,
                act_layer=lambda: nn.GELU(approximate="tanh"),
                drop=0,
            )
            self.action_embedder_B_3D = Mlp(
                in_features=self.action_dim,
                hidden_features=self.model_channels * 4,
                out_features=self.model_channels * 3,
                act_layer=lambda: nn.GELU(approximate="tanh"),
                drop=0,
            )
