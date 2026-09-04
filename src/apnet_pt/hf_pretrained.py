import operator
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


DEFAULT_APNET2_WEIGHTS = "qcmlforge"

#: Named APNet2 weight sets: Hugging Face-relative path templates for the atom
#: (multipole) and pair models plus ensemble sizes. ``ap2_tf_paper`` is the
#: ensemble published with the paper (``zachglick/apnet``), converted from
#: TensorFlow; see docs/apnet2-tensorflow-weights.md.
APNET2_WEIGHT_SETS = {
    "qcmlforge": {
        "atom": "am_ensemble/am_{model_id}.pt",
        "pair": "ap2_ensemble/ap2_{model_id}.pt",
        "n_models": 5,
        "n_atom_models": 10,
    },
    "ap2_tf_paper": {
        "atom": "ap2_tf_paper/atom_models/atom{model_id}.pt",
        "pair": "ap2_tf_paper/pair_models/pair{model_id}.pt",
        "n_models": 5,
    },
}


def apnet2_weight_sets() -> list[str]:
    """Named APNet2 weight sets accepted by ``weights=``."""
    return list(APNET2_WEIGHT_SETS)


def _apnet2_weight_set(weights: str) -> dict:
    try:
        return APNET2_WEIGHT_SETS[weights]
    except KeyError:
        raise ValueError(
            f"Unknown APNet2 weight set {weights!r}. "
            f"Available: {apnet2_weight_sets()}"
        ) from None


def _checked_model_id(model_id, weights: str, kind: str) -> tuple[dict, int]:
    """Validate ``model_id`` against a weight set's size for ``kind``."""
    weight_set = _apnet2_weight_set(weights)
    n_models = int(
        weight_set.get("n_atom_models", weight_set["n_models"])
        if kind == "atom"
        else weight_set["n_models"]
    )
    try:
        model_id = operator.index(model_id)
    except TypeError:
        raise TypeError(
            f"model_id must be an integer, got {type(model_id).__name__}"
        ) from None
    if not 0 <= model_id < n_models:
        label = "atom model_id" if kind == "atom" else "model_id"
        raise ValueError(
            f"{label} must be in [0, {n_models - 1}] for weights={weights!r}, "
            f"got {model_id}"
        )
    return weight_set, model_id


def apnet2_weight_set_size(weights: str = DEFAULT_APNET2_WEIGHTS) -> int:
    """Number of pair-model ensemble members in a named weight set."""
    return int(_apnet2_weight_set(weights)["n_models"])


def apnet2_atom_weight_path(
    model_id: int, weights: str = DEFAULT_APNET2_WEIGHTS
) -> str:
    """Repository-relative atom checkpoint path for a named weight set."""
    weight_set, model_id = _checked_model_id(model_id, weights, "atom")
    return weight_set["atom"].format(model_id=model_id)


def apnet2_weight_paths(
    model_id: int, weights: str = DEFAULT_APNET2_WEIGHTS
) -> dict[str, str]:
    """Repository-relative ``{"atom", "pair"}`` paths for one ensemble member.

    The two entries belong together: each pair model was trained against its
    own atom model.
    """
    weight_set, model_id = _checked_model_id(model_id, weights, "pair")
    return {
        kind: weight_set[kind].format(model_id=model_id)
        for kind in ("atom", "pair")
    }


def resolve_apnet2_weights(
    model_id: int, weights: str = DEFAULT_APNET2_WEIGHTS
) -> dict[str, str]:
    """Local ``{"atom", "pair"}`` checkpoint paths, downloading if needed."""
    rel_paths = apnet2_weight_paths(model_id, weights)
    resolved = resolve_pretrained_paths(list(rel_paths.values()))
    return {kind: resolved[rel] for kind, rel in rel_paths.items()}
