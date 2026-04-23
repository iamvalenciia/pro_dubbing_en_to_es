"""Centralized path handling for dual storage mode (network + local NVMe)."""

import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _default_network_dir() -> str:
    if os.name == "nt":
        return _REPO_ROOT
    return "/runpod-volume"


def _default_local_dir() -> str:
    if os.name == "nt":
        return os.path.join(_REPO_ROOT, "temp_workspace", "qdp_data")
    return "/workspace/qdp_data"


NETWORK_DIR = os.environ.get("QDP_NETWORK_DIR", _default_network_dir())
LOCAL_DIR = os.environ.get("QDP_LOCAL_DIR", _default_local_dir())

if os.name == "nt" and "QDP_NETWORK_DIR" not in os.environ:
    NETWORK_MODELS = os.path.join(_REPO_ROOT, "models")
    NETWORK_USER_INPUT = os.path.join(_REPO_ROOT, "input")
    NETWORK_OUTPUT = os.path.join(_REPO_ROOT, "output")
else:
    NETWORK_MODELS = os.path.join(NETWORK_DIR, "models")
    NETWORK_USER_INPUT = os.path.join(NETWORK_DIR, "user_input")
    NETWORK_OUTPUT = os.path.join(NETWORK_DIR, "output_final")

LOCAL_INPUT = os.path.join(LOCAL_DIR, "input")
LOCAL_TEMP = os.path.join(LOCAL_DIR, "temp_processing")
LOCAL_OUTPUT = os.path.join(LOCAL_DIR, "output")
LOCAL_LOGS = os.path.join(LOCAL_DIR, "logs")
LOCAL_CACHE = os.path.join(LOCAL_DIR, "torch_cache")


def ensure_local_dirs() -> None:
    """Create local and network working directories and redirect caches locally."""
    os.makedirs(NETWORK_DIR, exist_ok=True)
    os.makedirs(NETWORK_MODELS, exist_ok=True)
    os.makedirs(NETWORK_USER_INPUT, exist_ok=True)
    os.makedirs(NETWORK_OUTPUT, exist_ok=True)

    os.makedirs(LOCAL_DIR, exist_ok=True)
    os.makedirs(LOCAL_INPUT, exist_ok=True)
    os.makedirs(LOCAL_TEMP, exist_ok=True)
    os.makedirs(LOCAL_OUTPUT, exist_ok=True)
    os.makedirs(LOCAL_LOGS, exist_ok=True)
    os.makedirs(LOCAL_CACHE, exist_ok=True)

    # Keep model/cache writes off network volume for better runtime speed.
    os.environ.setdefault("TORCH_HOME", LOCAL_CACHE)
    os.environ.setdefault("HF_HOME", LOCAL_CACHE)
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(LOCAL_CACHE, "hub"))
