"""
Common utilities for nanochat_mlx.
Ported from nanochat/common.py - removes all PyTorch/CUDA/DDP dependencies.
"""

import os
import re
import logging
import urllib.request
from filelock import FileLock

import mlx.core as mx


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log messages."""
    COLORS = {
        'DEBUG': '\033[36m',
        'INFO': '\033[32m',
        'WARNING': '\033[33m',
        'ERROR': '\033[31m',
        'CRITICAL': '\033[35m',
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'

    def format(self, record):
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"
        message = super().format(record)
        if levelname == 'INFO':
            message = re.sub(r'(\d+\.?\d*\s*(?:GB|MB|%|docs))', rf'{self.BOLD}\1{self.RESET}', message)
            message = re.sub(r'(Shard \d+)', rf'{self.COLORS["INFO"]}{self.BOLD}\1{self.RESET}', message)
        return message


def setup_default_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.basicConfig(level=logging.INFO, handlers=[handler])


setup_default_logging()
logger = logging.getLogger(__name__)


def get_base_dir():
    """Get the base directory for nanochat data/checkpoints."""
    if os.environ.get("NANOCHAT_BASE_DIR"):
        nanochat_dir = os.environ.get("NANOCHAT_BASE_DIR")
    else:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache")
        nanochat_dir = os.path.join(cache_dir, "nanochat")
    os.makedirs(nanochat_dir, exist_ok=True)
    return nanochat_dir


def download_file_with_lock(url, filename, postprocess_fn=None):
    """Downloads a file from a URL to a local path in the base directory."""
    base_dir = get_base_dir()
    file_path = os.path.join(base_dir, filename)
    lock_path = file_path + ".lock"

    if os.path.exists(file_path):
        return file_path

    with FileLock(lock_path):
        if os.path.exists(file_path):
            return file_path

        print(f"Downloading {url}...")
        with urllib.request.urlopen(url) as response:
            content = response.read()

        with open(file_path, 'wb') as f:
            f.write(content)
        print(f"Downloaded to {file_path}")

        if postprocess_fn is not None:
            postprocess_fn(file_path)

    return file_path


def print0(s="", **kwargs):
    """Print function that only prints on rank 0. For MLX (single device), always prints."""
    print(s, **kwargs)


def print_banner():
    banner = """
                                                       █████                █████
                                                      ░░███                ░░███
     ████████    ██████   ████████    ██████   ██████  ░███████    ██████  ███████
    ░░███░░███  ░░░░░███ ░░███░░███  ███░░███ ███░░███ ░███░░███  ░░░░░███░░░███░
     ░███ ░███   ███████  ░███ ░███ ░███ ░███░███ ░░░  ░███ ░███   ███████  ░███
     ░███ ░███  ███░░███  ░███ ░███ ░███ ░███░███  ███ ░███ ░███  ███░░███  ░███ ███
     ████ █████░░████████ ████ █████░░██████ ░░██████  ████ █████░░███████  ░░█████
    ░░░░ ░░░░░  ░░░░░░░░ ░░░░ ░░░░░  ░░░░░░   ░░░░░░  ░░░░ ░░░░░  ░░░░░░░░   ░░░░░

    MLX Edition - For Apple Silicon
    """
    print0(banner)


def mlx_init(seed=42):
    """Initialize MLX random seed. No DDP, no device management needed."""
    mx.random.seed(seed)
    print0(f"MLX initialized with seed {seed}")
    print0(f"MLX default device: {mx.default_device()}")


def get_dist_info():
    """Return dist info compatible with original API. Always single device for MLX."""
    return False, 0, 0, 1


class DummyWandb:
    """Useful if we wish to not use wandb but have all the same signatures."""
    def __init__(self):
        pass
    def log(self, *args, **kwargs):
        pass
    def finish(self):
        pass


def get_memory_usage():
    """Get current memory usage in bytes using psutil."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss
    except ImportError:
        return 0


def get_peak_flops(device_name: str) -> float:
    """
    Return estimated peak BF16 FLOPS for Apple Silicon chips.
    These are rough estimates based on published specs.
    """
    name = device_name.lower()
    _PEAK_FLOPS_TABLE = (
        # M4 Ultra
        (["m4", "ultra"], 28.0e12),
        # M4 Max
        (["m4", "max"], 6.8e12),
        # M4 Pro
        (["m4", "pro"], 4.0e12),
        # M4 (base)
        (["m4"], 3.0e12),
        # M3 Ultra
        (["m3", "ultra"], 22.0e12),
        # M3 Max
        (["m3", "max"], 6.0e12),
        # M3 Pro
        (["m3", "pro"], 3.5e12),
        # M3 (base)
        (["m3"], 2.5e12),
        # M2 Ultra
        (["m2", "ultra"], 27.2e12),
        # M2 Max
        (["m2", "max"], 13.6e12),
        # M2 Pro
        (["m2", "pro"], 6.8e12),
        # M2 (base)
        (["m2"], 3.6e12),
        # M1 Ultra
        (["m1", "ultra"], 21.0e12),
        # M1 Max
        (["m1", "max"], 10.4e12),
        # M1 Pro
        (["m1", "pro"], 5.2e12),
        # M1 (base)
        (["m1"], 2.6e12),
    )
    for patterns, flops in _PEAK_FLOPS_TABLE:
        if all(p in name for p in patterns):
            return flops

    logger.warning(f"Peak flops undefined for: {device_name}, MFU will show as 0%")
    return float('inf')
