"""The public plan validator resolves the parser's shared unset sentinel."""

import pytest

from apnet_pt.training.mace_ap3d3_factory import validate_mace_cli_args
from tests.test_mace_ap3d3_cli import _base_cli


@pytest.mark.parametrize("batch_size,expected", [(None, 16), (1, 1), (32, 32)])
def test_mace_validator_resolves_batch_size_without_main_mutation(
    tmp_path, batch_size, expected
):
    args, _ = _base_cli(tmp_path)
    args.batch_size = batch_size
    plan = validate_mace_cli_args(args)
    assert plan.batch_size == expected
    assert args.batch_size == batch_size


@pytest.mark.parametrize("batch_size", [0, -1])
def test_mace_validator_still_rejects_nonpositive_explicit_batches(
    tmp_path, batch_size
):
    args, _ = _base_cli(tmp_path)
    args.batch_size = batch_size
    with pytest.raises(ValueError, match="batch_size"):
        validate_mace_cli_args(args)
