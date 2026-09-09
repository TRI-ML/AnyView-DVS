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

from typing import Tuple

import torch

from .batch_ops import batch_mul


def phi1(t: torch.Tensor) -> torch.Tensor:
    """
    Compute the first order phi function: (exp(t) - 1) / t.

    Args:
        t: Input tensor.

    Returns:
        Tensor: Result of phi1 function.
    """
    input_dtype = t.dtype
    t = t.to(dtype=torch.float64)
    return (torch.expm1(t) / t).to(dtype=input_dtype)


def phi2(t: torch.Tensor) -> torch.Tensor:
    """
    Compute the second order phi function: (phi1(t) - 1) / t.

    Args:
        t: Input tensor.

    Returns:
        Tensor: Result of phi2 function.
    """
    input_dtype = t.dtype
    t = t.to(dtype=torch.float64)
    return ((phi1(t) - 1.0) / t).to(dtype=input_dtype)


def res_x0_rk2_step(
    x_s: torch.Tensor,
    t: torch.Tensor,
    s: torch.Tensor,
    x0_s: torch.Tensor,
    s1: torch.Tensor,
    x0_s1: torch.Tensor,
) -> torch.Tensor:
    """
    Perform a residual-based 2nd order Runge-Kutta step.

    Args:
        x_s: Current state tensor.
        t: Target time tensor.
        s: Current time tensor.
        x0_s: Prediction at current time.
        s1: Intermediate time tensor.
        x0_s1: Prediction at intermediate time.

    Returns:
        Tensor: Updated state tensor.

    Raises:
        AssertionError: If step size is too small.
    """
    s = -torch.log(s)
    t = -torch.log(t)
    m = -torch.log(s1)

    dt = t - s
    assert not torch.any(torch.isclose(dt, torch.zeros_like(dt), atol=1e-6)), "Step size is too small"
    assert not torch.any(torch.isclose(m - s, torch.zeros_like(dt), atol=1e-6)), "Step size is too small"

    c2 = (m - s) / dt
    phi1_val, phi2_val = phi1(-dt), phi2(-dt)

    # Handle edge case where t = s = m
    b1 = torch.nan_to_num(phi1_val - 1.0 / c2 * phi2_val, nan=0.0)
    b2 = torch.nan_to_num(1.0 / c2 * phi2_val, nan=0.0)

    return batch_mul(torch.exp(-dt), x_s) + batch_mul(dt, batch_mul(b1, x0_s) + batch_mul(b2, x0_s1))


def reg_x0_euler_step(
    x_s: torch.Tensor,
    s: torch.Tensor,
    t: torch.Tensor,
    x0_s: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform a regularized Euler step based on x0 prediction.

    Args:
        x_s: Current state tensor.
        s: Current time tensor.
        t: Target time tensor.
        x0_s: Prediction at current time.

    Returns:
        Tuple[Tensor, Tensor]: Updated state tensor and current prediction.
    """
    coef_x0 = (s - t) / s
    coef_xs = t / s
    return batch_mul(coef_x0, x0_s) + batch_mul(coef_xs, x_s), x0_s
