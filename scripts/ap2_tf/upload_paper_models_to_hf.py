"""Publish the converted AP-Net2 paper checkpoints to the QCMLForge HF repo.

The ten checkpoints in ``models/ap2_tf_paper/`` are the TensorFlow ensemble from
``zachglick/apnet`` converted by ``convert_tf_to_pt.py`` and locked against
recorded TensorFlow output by ``tests/test_ap2_tf_parity.py``. Uploading them
makes ``weights="ap2_tf_paper"`` work for users who never clone the repository.

The remote layout is not spelled out here: it is read from
``apnet_pt.hf_pretrained.APNET2_WEIGHT_SETS``, so the upload cannot drift from
what the loader asks for. Each uploaded file is read back and compared by
sha256 unless ``--no-verify`` is passed.

Usage:

    python upload_paper_models_to_hf.py --dry-run
    python upload_paper_models_to_hf.py
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from apnet_pt.hf_pretrained import (  # noqa: E402
    HF_REPO_ID,
    apnet2_weight_paths,
    apnet2_weight_set_size,
)


def sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def planned_uploads(weights: str, models_dir: Path) -> list[tuple[Path, str]]:
    """Pair every local checkpoint with the repo path the loader expects.

    The local tree mirrors the remote one, so the basename of each registry
    path is enough to locate the file; a mismatch means the rename that
    produced ``models/ap2_tf_paper`` and the registry have diverged.
    """
    uploads = []
    for model_id in range(apnet2_weight_set_size(weights)):
        for rel_path in apnet2_weight_paths(model_id, weights).values():
            local_path = models_dir / Path(rel_path).relative_to(weights)
            if not local_path.is_file():
                raise FileNotFoundError(
                    f"{local_path} is missing but the registry maps "
                    f"{weights}/{model_id} to '{rel_path}'"
                )
            uploads.append((local_path, rel_path))
    return uploads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="ap2_tf_paper")
    parser.add_argument("--repo-id", default=HF_REPO_ID)
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Local directory holding the weight set (default models/<weights>).",
    )
    parser.add_argument(
        "--commit-message",
        default=None,
        help="Commit message for the upload (default mentions the weight set).",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the post-upload sha256 read-back.",
    )
    args = parser.parse_args()

    models_dir = args.models_dir or REPO_ROOT / "models" / args.weights
    uploads = planned_uploads(args.weights, models_dir)

    print(f"repo: {args.repo_id}")
    total = 0
    for local_path, rel_path in uploads:
        size = local_path.stat().st_size
        total += size
        print(f"  {local_path.relative_to(REPO_ROOT)} -> {rel_path} "
              f"({size / 1e6:.1f} MB, sha256 {sha256(local_path)[:12]})")
    print(f"{len(uploads)} files, {total / 1e6:.1f} MB")

    if args.dry_run:
        print("dry run: nothing uploaded")
        return

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    commit_message = args.commit_message or (
        f"Add the {args.weights} APNet2 checkpoints"
    )
    api.create_commit(
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=commit_message,
        operations=[
            _addition(local_path, rel_path) for local_path, rel_path in uploads
        ],
    )
    print(f"uploaded {len(uploads)} files")

    if args.no_verify:
        return

    for local_path, rel_path in uploads:
        remote_path = hf_hub_download(
            repo_id=args.repo_id,
            filename=rel_path,
            force_download=True,
        )
        local_digest = sha256(local_path)
        remote_digest = sha256(remote_path)
        if local_digest != remote_digest:
            raise SystemExit(
                f"sha256 mismatch for '{rel_path}': "
                f"local {local_digest} != remote {remote_digest}"
            )
        print(f"  verified {rel_path}")
    print("all files verified")


def _addition(local_path: Path, rel_path: str):
    from huggingface_hub import CommitOperationAdd

    return CommitOperationAdd(
        path_in_repo=rel_path, path_or_fileobj=str(local_path)
    )


if __name__ == "__main__":
    main()
