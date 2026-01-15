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

import torch

from .data.reference_cn import reference_cn

def gaussian_weight(dcn: torch.tensor, factor: float = 4.0) -> torch.tensor:
    """
    Compute Gaussian-style weights from differences in coordination numbers.
    
    Parameters:
        dcn (torch.tensor): Difference between reference and target coordination numbers.
        factor (float): Positive scalar controlling the Gaussian width; larger values produce narrower weights.
    
    Returns:
        torch.tensor: Weight values given by exp(-factor * dcn**2).
    """

    return torch.exp(-factor * dcn.pow(2))



#Uses gaussian weighting function
def weight_references(
    numbers: torch.tensor,
    cn: torch.tensor,
) -> torch.tensor:
    """
    Compute normalized Gaussian weights for reference coordination numbers for each atom.
    
    For each atom (indexed by `numbers`), the function retrieves reference coordination numbers and computes a Gaussian weight exp(-4 * dcn^2) as a function of the difference between each reference CN and the provided `cn`. Computation is performed in double precision to avoid underflow; results are normalized across reference entries and cast back to the input dtype. Entries with no valid references (refcn < 0) return zeros. If normalization would fail (sum of weights is zero) or produce overflow, the weight for the largest reference CN is set to 1 and all others to 0. Final weights are clamped to a minimum of 1e-10.
    
    Parameters:
        numbers (torch.tensor): Atomic numbers indexing the reference CN table.
        cn (torch.tensor): Coordination numbers of the atoms; shape must broadcast with the reference CNs retrieved for `numbers`.
    
    Returns:
        torch.tensor: Normalized weights per reference entry for each atom, with the same leading shape as `cn` and a trailing dimension matching the number of references for each atomic number.
    """
    refcn = reference_cn()[numbers]
    mask = refcn >= 0

    zero = torch.tensor(0.0, device=cn.device, dtype=cn.dtype)
    zero_double = torch.tensor(0.0, device=cn.device, dtype=torch.double)
    one = torch.tensor(1.0, device=cn.device, dtype=cn.dtype)

    # Due to the exponentiation, `norms` and `weights` may become very small.
    # This may cause problems for the division by `norms`. It may occur that
    # `weights` and `norms` are equal, in which case the result should be
    # exactly one. This might, however, not be the case and ultimately cause
    # larger deviations in the final values.
    #
    # This must be done in the D4 variant because the weighting functions
    # contains higher powers, which lead to values down to 1e-300.
    # Since there are also cases in D3, we have to evaluate this portion
    # in double precision to retain the correct results and avoid nan's.
    dcn = (refcn - cn.unsqueeze(-1)).type(torch.double)
    weights = torch.where(
        mask,
        gaussian_weight(dcn,),
        zero_double,  # not eps!
    )

    # Previously, a small value was added to `norms` to prevent division by zero
    # (`norms = torch.add(torch.sum(weights, dim=-1), 1e-20)`). However, even
    # such small values can lead to relatively large deviations because the
    # small value is not added to the weights, and hence, the case where
    # `weights` and `norms` are equal does not yield one anymore. In fact, the
    # test suite fails because some elements deviate up to around 1e-4.
    # We solve this by running in double precision, adding a very small number
    # and using multiple masks.

    small = torch.tensor(1e-300, device=cn.device, dtype=torch.double)

    # normalize weights
    norm = torch.where(
        mask,
        torch.sum(weights, dim=-1, keepdim=True),
        small,  # double!
    )

    # back to real dtype
    #gw_temp = storch.divide(weights, norm, eps=small).type(cn.dtype)
    gw_temp = torch.divide(weights, norm,).clamp_min(1e-10).type(cn.dtype)

    # If the tensor is not a grad tracking tensor, we can check for NaN's
    # if not is_functorch_tensor(gw_temp):
    #     assert torch.isnan(gw_temp).sum() == 0

    # The following section handles cases with large CNs that lead to zeros in
    # after the exponential in the weighting function. If this happens all
    # weights become zero, which is not desired. Instead, we set the weight of
    # the largest reference number to one.
    # This case can occur if the CN of the current (actual) system is too far
    # away from the largest CN of the reference systems. An example would be an
    # atom within a fullerene (La3N@C80).

    # maximum reference CN for each atom
    maxcn = torch.max(refcn, dim=-1, keepdim=True)[0]

    # Here, we catch the potential NaN's from `gw_temp`. We cannot use `gw_temp`
    # directly, because we have to use safe divide to not get NaN's in the
    # backward. But `norm == 0` is equivalent. Additionally, we catch very
    # large values occuring because of division by small values.
    exceptional = (norm == 0) | (gw_temp > torch.finfo(cn.dtype).max)

    gw = torch.where(
        exceptional,
        torch.where(refcn == maxcn, one, zero),
        gw_temp,
    )

    return torch.where(mask, gw, zero)