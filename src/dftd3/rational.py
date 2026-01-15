# This file is part of tad-dftd3.
# SPDX-Identifier: Apache-2.0
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
r"""
Rational (Becke-Johnson) damping function
=========================================

This module defines the rational damping function, also known as Becke-Johnson
damping.

.. math::

    f^n_{\text{damp}}\left(R_0^{\text{AB}}\right) =
    \dfrac{R^n_{\text{AB}}}{R^n_{\text{AB}} +
    \left( a_1 R_0^{\text{AB}} + a_2 \right)^n}
"""

import torch


from . import defaults


def rational_damping(
    order: int,
    distances: torch.tensor,
    qq: torch.tensor,
    param: dict[str, torch.tensor],
) -> torch.tensor:
    """
    Compute the Becke–Johnson (rational) damping factors for pairwise dispersion interactions.
    
    Calculates damping values for a given dispersion order using the formula
    1 / (distances**order + (a1 * sqrt(qq) + a2)**order). The `param` mapping may
    provide tensors for keys 'a1' and 'a2'; when absent, module defaults are used
    with the same device and dtype as `distances`.
    
    Parameters
    ----------
    order : int
        Dispersion interaction order (e.g., 6, 8).
    distances : torch.tensor
        Pairwise distances between atoms.
    qq : torch.tensor
        Quotient C8/C6 for each pair.
    param : dict[str, torch.tensor]
        Damping parameters; may include 'a1' and 'a2' tensors.
    
    Returns
    -------
    torch.tensor
        Damping factors with the same device and dtype as `distances`.
    """
    dd = {"device": distances.device, "dtype": distances.dtype}

    a1 = param.get("a1", torch.tensor(defaults.A1, **dd))
    a2 = param.get("a2", torch.tensor(defaults.A2, **dd))
    return 1.0 / (distances.pow(order) + (a1 * torch.sqrt(qq) + a2).pow(order))