import os
import sys
from importlib import resources
import logging

HF_REPO_ID = "awallace3/qcmlforge"
_DOWNLOAD_APPROVED = None
LOGGER = logging.getLogger(__name__)


def _hf_hub_download(rel_path: str, local_files_only: bool) -> str:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to load pretrained models. "
            "Install qcmlforge dependencies or `pip install huggingface_hub`."
        ) from exc

    return hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=rel_path,
        local_files_only=local_files_only,
    )


def _packaged_model_path(rel_path: str) -> str | None:
    model_path = resources.files("apnet_pt").joinpath("models", *rel_path.split("/"))
    return str(model_path) if model_path.is_file() else None


def _allow_model_download(missing_paths: list[str]) -> bool:
    global _DOWNLOAD_APPROVED

    if _DOWNLOAD_APPROVED is not None:
        return _DOWNLOAD_APPROVED

    env_value = os.getenv("QCMLFORGE_AUTO_DOWNLOAD_PRETRAINED", "").strip().lower()
    if env_value in {"1", "true", "yes", "y"}:
        _DOWNLOAD_APPROVED = True
        LOGGER.info(
            "QCMLFORGE_AUTO_DOWNLOAD_PRETRAINED enabled pretrained downloads from %s",
            HF_REPO_ID,
        )
        return True
    if env_value in {"0", "false", "no", "n"}:
        _DOWNLOAD_APPROVED = False
        LOGGER.info(
            "QCMLFORGE_AUTO_DOWNLOAD_PRETRAINED disabled pretrained downloads from %s",
            HF_REPO_ID,
        )
        return False

    if not sys.stdin.isatty():
        _DOWNLOAD_APPROVED = False
        return False

    preview = ", ".join(missing_paths[:3])
    if len(missing_paths) > 3:
        preview += ", ..."
    try:
        answer = (
            input(
                "Pretrained model weights are not available locally and need to be "
                f"downloaded from https://huggingface.co/{HF_REPO_ID} "
                f"(missing: {preview}). Download now? [y/N]: "
            )
            .strip()
            .lower()
        )
    except (EOFError, KeyboardInterrupt):
        _DOWNLOAD_APPROVED = False
        return False
    _DOWNLOAD_APPROVED = answer in {"y", "yes"}
    return _DOWNLOAD_APPROVED


def resolve_pretrained_paths(rel_paths: list[str]) -> dict[str, str]:
    """
    Resolve pretrained artifact paths for one or more model files.

    Parameters
    ----------
    rel_paths : list[str]
        Relative paths inside the QCMLForge Hugging Face repository.

    Returns
    -------
    dict[str, str]
        Mapping from each requested relative path to a local filesystem path.

    Notes
    -----
    Resolution checks the local Hugging Face cache first, optionally downloads
    missing artifacts, and falls back to packaged files when they exist.
    Interactive downloads are controlled by
    ``QCMLFORGE_AUTO_DOWNLOAD_PRETRAINED``.
    """
    resolved = {}
    missing = []

    for rel_path in rel_paths:
        try:
            resolved[rel_path] = _hf_hub_download(rel_path, local_files_only=True)
        except ImportError:
            raise
        except Exception:
            missing.append(rel_path)

    if not missing:
        return resolved

    if not _allow_model_download(missing):
        for rel_path in missing:
            fallback = _packaged_model_path(rel_path)
            if fallback is not None:
                resolved[rel_path] = fallback
                continue
            raise RuntimeError(
                "Missing pretrained model in local cache. "
                "Set QCMLFORGE_AUTO_DOWNLOAD_PRETRAINED=1 to auto-download, "
                f"or run interactively and accept download for '{rel_path}'."
            )
        return resolved

    for rel_path in missing:
        try:
            resolved[rel_path] = _hf_hub_download(rel_path, local_files_only=False)
        except ImportError:
            raise
        except Exception as exc:
            fallback = _packaged_model_path(rel_path)
            if fallback is not None:
                resolved[rel_path] = fallback
                continue
            raise RuntimeError(
                f"Unable to load pretrained model '{rel_path}' from "
                f"https://huggingface.co/{HF_REPO_ID}."
            ) from exc

    return resolved


def resolve_pretrained_path(rel_path: str) -> str:
    """
    Resolve a single pretrained artifact path.

    Parameters
    ----------
    rel_path : str
        Relative path inside the QCMLForge Hugging Face repository.

    Returns
    -------
    str
        Local filesystem path for the requested artifact.

    Notes
    -----
    This is a thin wrapper around ``resolve_pretrained_paths``.
    """
    return resolve_pretrained_paths([rel_path])[rel_path]
