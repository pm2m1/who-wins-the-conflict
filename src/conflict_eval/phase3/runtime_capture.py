"""Runtime, hardware and artifact-digest capture for Phase 3C (§36).

§36 requires the pre-run freeze manifest to record "Environment and
hardware capture" and a SHA256 for every empirical artifact. This module
produces both, and nothing here loads a model or touches a dataset.

Two deliberate properties:

- **Digests use the single project pattern.** `sha256_file` produces the
  lowercase 64-hex form that `calibration_provenance.SHA256_PATTERN`
  validates, so a digest written here is by construction acceptable to the
  manifest validator.
- **Capture never invents.** If a value cannot be observed (no CUDA, a
  package absent) the field is recorded as the literal string
  ``"unavailable"`` rather than guessed or omitted. `validate_manifest`
  rejects `None` placeholders, so an unfilled capture cannot pass as a
  real one -- but an honestly-recorded "unavailable" is visible in the
  artifact rather than silently blank.
"""

from __future__ import annotations

import hashlib
import platform
import sys
from pathlib import Path
from typing import Any

#: Packages whose exact versions materially affect generation/scoring.
CAPTURED_PACKAGES: tuple[str, ...] = (
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "tokenizers",
    "numpy",
)

UNAVAILABLE = "unavailable"


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """SHA256 of a file's bytes, lowercase hex.

    Streamed, so a multi-gigabyte artifact does not have to be resident.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    """SHA256 of text encoded UTF-8.

    Newlines are NOT normalized: the digest describes the exact bytes on
    disk, which is what a replicator will re-hash.
    """
    return sha256_bytes(text.encode("utf-8"))


def _package_version(name: str) -> str:
    # A package simply not being installed is a normal, recordable state --
    # not an error to propagate out of a provenance capture.
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return UNAVAILABLE


def capture_environment() -> dict[str, Any]:
    """Software environment (§36).

    Every value is observed. Nothing is defaulted to a plausible-looking
    version string.
    """
    capture: dict[str, Any] = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable": sys.executable,
    }
    for package in CAPTURED_PACKAGES:
        capture[package] = _package_version(package)
    capture["cuda"] = _cuda_version()
    return capture


def _cuda_version() -> str:
    try:
        import torch
    except ImportError:
        return UNAVAILABLE
    return getattr(getattr(torch, "version", None), "cuda", None) or UNAVAILABLE


def capture_hardware() -> dict[str, Any]:
    """GPU/hardware capture (§36).

    Reads `torch.cuda` if it is importable and initialized. On a host with
    no GPU every field is `"unavailable"`, which is the honest record --
    and `assert_cuda_available` is what actually prevents a screening run
    from proceeding there.
    """
    capture: dict[str, Any] = {
        "machine": platform.machine(),
        "processor": platform.processor() or UNAVAILABLE,
        "gpu_name": UNAVAILABLE,
        "gpu_count": 0,
        "vram": UNAVAILABLE,
        "cuda_available": False,
    }
    try:
        import torch
    except ImportError:
        return capture
    # torch.cuda raises RuntimeError/AssertionError on a driverless or
    # misconfigured host. That is a legitimate "no GPU here" answer for a
    # capture, so it is recorded as such rather than raised -- the hard
    # refusal to run without CUDA is `assert_cuda_available`, not this.
    try:
        if torch.cuda.is_available():
            capture["cuda_available"] = True
            capture["gpu_count"] = torch.cuda.device_count()
            capture["gpu_name"] = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory
            capture["vram"] = f"{total / (1024 ** 3):.1f}GiB"
    except (RuntimeError, AssertionError, OSError):
        return capture
    return capture


class RuntimeRequirementError(RuntimeError):
    """Raised when the frozen Phase 3 runtime requirements are unmet."""


def assert_cuda_available() -> None:
    """Refuse to screen on a host without CUDA.

    The frozen design specifies unquantized float16 execution on a
    24 GB-class GPU (§7, §35). Silently falling back to CPU would produce
    records that are not reproducible against that specification, so this
    fails loudly instead.
    """
    hardware = capture_hardware()
    if not hardware["cuda_available"]:
        raise RuntimeRequirementError(
            "CUDA is not available. Phase 3 baseline screening runs unquantized "
            "float16 on a GPU (§7, §35); running on CPU would silently produce "
            "records that do not match the frozen runtime specification."
        )


def assert_runtime_matches(dtype: str, quantization: str | None) -> None:
    """Refuse any runtime other than the frozen unquantized float16 one."""
    if str(dtype).lower() not in ("float16", "fp16", "torch.float16"):
        raise RuntimeRequirementError(
            f"Phase 3 requires float16, got dtype={dtype!r} (§7, §36). The frozen "
            "design pins precision; it is not a tuning knob."
        )
    if quantization is not None and str(quantization).lower() not in ("none", "false"):
        raise RuntimeRequirementError(
            f"Phase 3 runs UNQUANTIZED, got quantization={quantization!r} (§7, §36)."
        )


def runtime_provenance(dtype: str = "float16", quantization: str = "none") -> dict[str, Any]:
    """The full runtime block written beside every screening artifact."""
    return {
        "dtype": dtype,
        "quantization": quantization,
        "environment": capture_environment(),
        "hardware": capture_hardware(),
    }
