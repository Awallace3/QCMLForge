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
    Calculate weight of indivdual reference system.

    Parameters
    ----------
    dcn : torch.tensor
        Difference of coordination numbers.
    factor : float
        Factor to calculate weight.

    Returns
    -------
    torch.tensor
        Weight of individual reference system.
    """

    return torch.exp(-factor * dcn.pow(2))



#Uses gaussian weighting function
def weight_references(
    numbers: torch.tensor,
    cn: torch.tensor,
) -> torch.tensor:
    """
    Calculate the weights of the reference system.

    Parameters
    ----------
    numbers : torch.tensor
        The atomic numbers of the atoms in the system.
    cn : torch.tensor
        Coordination numbers for all atoms in the system.

    Returns
    -------
    torch.tensor
        Weights of all reference systems
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
