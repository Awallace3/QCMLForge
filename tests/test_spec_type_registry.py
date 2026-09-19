"""A spec_type is only usable when every copy of its registration agrees.

Each dataset class carries its own accepted-spec assert list, its own
``raw_file_names`` map and its own split-spec frozenset.  A spec added to one
copy and not another fails at dataset construction with an AssertionError that
names neither the spec nor the class, so pin the agreement here instead.
"""

import pytest

from apnet_pt.pairwise_datasets import (
    APNET2_SPLIT_SPEC_TYPES,
    apnet2_module_dataset,
)
from apnet_pt.pt_datasets.ap2_fused_ds import (
    AP2_FUSED_SPLIT_SPEC_TYPES,
    ap2_fused_module_dataset,
    ap2_fused_module_dataset_lmdb,
)
from apnet_pt.pt_datasets.ap3_fused_ds import (
    AP3_FUSED_SPLIT_SPEC_TYPES,
    ap3_fused_module_dataset,
    ap3_fused_module_dataset_lmdb,
)

# spec_type -> the raw train/test pair every dataset class must resolve it to.
SHARED_RAW_FILE_NAMES = {
    2: ["1600K_train_dimers-fixed.pkl", "1600K_test_dimers-fixed.pkl"],
    5: ["t_train.pkl", "t_test.pkl"],
    6: ["t_train10k.pkl", "t_test2k.pkl"],
    7: ["t_train_100.pkl", "t_test_20.pkl"],
    9: ["t_train_19.pkl", "t_test_19.pkl"],
    11: [
        "splinter_omol25_sapt0indu_v1_train.pkl",
        "splinter_omol25_sapt0indu_v1_test.pkl",
    ],
}

DATASET_CLASSES = [
    apnet2_module_dataset,
    ap2_fused_module_dataset,
    ap2_fused_module_dataset_lmdb,
    ap3_fused_module_dataset,
    ap3_fused_module_dataset_lmdb,
]

SPLIT_SPEC_SETS = {
    "apnet2": APNET2_SPLIT_SPEC_TYPES,
    "ap2_fused": AP2_FUSED_SPLIT_SPEC_TYPES,
    "ap3_fused": AP3_FUSED_SPLIT_SPEC_TYPES,
}


def raw_file_names_for(cls, spec_type):
    """Read the raw-file map without building a dataset.

    ``__init__`` wants a root, an atom model and real files on disk; the map
    itself only reads ``self.spec_type``.
    """
    obj = object.__new__(cls)
    obj.spec_type = spec_type
    return list(cls.raw_file_names.fget(obj))


@pytest.mark.parametrize("cls", DATASET_CLASSES, ids=lambda c: c.__name__)
@pytest.mark.parametrize("spec_type", sorted(SHARED_RAW_FILE_NAMES))
def test_raw_file_names_agree_across_dataset_classes(cls, spec_type):
    assert raw_file_names_for(cls, spec_type) == SHARED_RAW_FILE_NAMES[spec_type]


@pytest.mark.parametrize("name", sorted(SPLIT_SPEC_SETS))
@pytest.mark.parametrize("spec_type", sorted(SHARED_RAW_FILE_NAMES))
def test_split_spec_sets_cover_every_train_test_spec(name, spec_type):
    """A spec with separate train/test raw files must select split files.

    Without this the class reads both raw files into one store and the test
    split silently becomes training data.
    """
    assert spec_type in SPLIT_SPEC_SETS[name]
