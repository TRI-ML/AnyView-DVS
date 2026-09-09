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

# NOTE: plain-torch subset of cosmos-predict2 needed by the AnyView DiT, tokenizer, and
# sampler. No transformer_engine, flash_attn, diffusers, or loguru: attention runs through
# torch.nn.functional.scaled_dot_product_attention. State-dict keys match the upstream modules.
