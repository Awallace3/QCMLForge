"""
Main init for the AP-Net package
"""

__version__ = "0.0.1"
__author__ = "Austin M. Wallace; Zachary M. Glick"
__credits__ = "Georgia Institute of Technology"

import warnings

import torch


def _install_safe_torch_compile() -> None:
    """Fall back to eager mode if ``torch.compile`` is unavailable/broken.

    Some PyTorch/Inductor builds can import ``torch`` successfully but fail as
    soon as ``torch.compile`` is applied during module import.  Several AP-Net
    modules use ``@torch.compile`` decorators, so catch those failures and keep
    the package importable by returning the original callable/module.
    """
    if getattr(torch.compile, "_apnet_safe_compile", False):
        return

    original_compile = torch.compile

    def safe_compile(model=None, *args, **kwargs):
        def compile_or_eager(obj):
            try:
                return original_compile(obj, *args, **kwargs)
            except (ImportError, RuntimeError, AttributeError) as exc:
                warnings.warn(
                    "torch.compile failed during AP-Net setup; falling back to "
                    f"eager execution. Original error: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return obj

        if model is None:
            return compile_or_eager
        return compile_or_eager(model)

    safe_compile._apnet_safe_compile = True
    safe_compile._apnet_original_compile = original_compile
    torch.compile = safe_compile


_install_safe_torch_compile()

from . import atomic_datasets
from . import pairwise_datasets
from .util import load_dimer_dataset, load_monomer_dataset, load_atomic_module_graph_dataset
from . import torch_util
from .AtomPairwiseModels.apnet2 import APNet2Model
from . import AtomPairwiseModels
from . import AtomModels
from .pretrained_models import atom_model_predict
from . import classical_induction
from . import pt_datasets
from .model_print import get_model_info, print_model_tree, ModelInfo
