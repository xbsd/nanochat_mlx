"""
The base/pretraining dataset is a set of parquet files.
This file contains utilities for:
- iterating over the parquet files and yielding documents from it
- download the files on demand if they are not on disk

For details of how the dataset was prepared, see `repackage_data_reference.py`.
"""

import os
import argparse
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import pyarrow.parquet as pq
from tqdm import tqdm

from nanochat_mlx.common import get_base_dir

# -----
# The specifics of the current pretraining dataset

# The URL on the internet where the data is hosted and downloaded from on demand
BASE_URL = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
MAX_SHARD = 1822  # the last datashard is shard_01822.parquet
index_to_filename = lambda index: f"shard_{index:05d}.parquet"  # format of the filenames
base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "base_data")
os.makedirs(DATA_DIR, exist_ok=True)

# -----
# These functions are useful utilities to other modules, can/should be imported

def list_parquet_files(data_dir=None):
    """Looks into a data dir and returns full paths to all parquet files."""
    data_dir = DATA_DIR if data_dir is None else data_dir
    parquet_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])
    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    return parquet_paths

def parquets_iter_batched(split, start=0, step=1):
    """
    Iterate through the dataset, in batches of underlying row_groups for efficiency.
    - split can be "train" or "val". the last parquet file will be val.
    - start/step are useful for skipping rows. e.g. start=0, step=1 for single device.
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files()
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column('text').to_pylist()
            yield texts

# -----
# Tokenizer artifacts needed before training can begin

TOKENIZER_BASE_URL = "https://huggingface.co/karpathy/nanochat-d32/resolve/main"
TOKENIZER_DIR = os.path.join(base_dir, "tokenizer")
TOKENIZER_FILES = {
    "tokenizer.pkl": "33f28610ffd37a57d6631f8d7bd91929bd877ae3f4a87dcbdff00b07f6bd7cc3",
    "token_bytes.pt": "e280877820a90174f3b47bf797b67b9026cd859b7d6d5b7f78e64bcdaca126b4",
}


def _sha256(path):
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def download_tokenizer():
    """Download tokenizer artifacts if missing or corrupted (checksum mismatch)."""
    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    needed = []
    for filename, expected_sha in TOKENIZER_FILES.items():
        filepath = os.path.join(TOKENIZER_DIR, filename)
        if os.path.exists(filepath) and _sha256(filepath) == expected_sha:
            continue
        needed.append(filename)

    if not needed:
        return

    print("Downloading tokenizer files...")
    for filename in needed:
        filepath = os.path.join(TOKENIZER_DIR, filename)
        expected_sha = TOKENIZER_FILES[filename]
        if os.path.exists(filepath):
            print(f"  {filename}: checksum mismatch, re-downloading...")
        else:
            print(f"  Downloading {filename}...")
        url = f"{TOKENIZER_BASE_URL}/{filename}"
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                temp_path = filepath + ".tmp"
                with open(temp_path, 'wb') as f:
                    f.write(response.content)
                actual_sha = _sha256(temp_path)
                if actual_sha != expected_sha:
                    os.remove(temp_path)
                    raise RuntimeError(
                        f"Checksum mismatch for {filename}: "
                        f"expected {expected_sha[:16]}..., got {actual_sha[:16]}..."
                    )
                os.rename(temp_path, filepath)
                print(f"  Saved {filename} ({os.path.getsize(filepath) // 1024}KB)")
                break
            except (requests.RequestException, IOError) as e:
                if attempt < max_attempts:
                    print(f"  Attempt {attempt}/{max_attempts} failed: {e}, retrying...")
                    time.sleep(2 ** attempt)
                else:
                    raise RuntimeError(f"Failed to download {filename} from {url}") from e
    print(f"Tokenizer ready at {TOKENIZER_DIR}")


# -----
def download_single_file(index):
    """Downloads a single file index, with progress bar and backoff."""

    # Construct the local filepath for this file and skip if it already exists
    filename = index_to_filename(index)
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        return True

    # Construct the remote URL for this file
    url = f"{BASE_URL}/{filename}"

    # Download with retries
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))
            temp_path = filepath + ".tmp"
            with open(temp_path, 'wb') as f, \
                 tqdm(total=total_size, unit="B", unit_scale=True,
                      desc=filename, leave=False) as pbar:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            # Move temp file to final location
            os.rename(temp_path, filepath)
            return True

        except (requests.RequestException, IOError) as e:
            # Clean up any partial files
            for path in [filepath + ".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
            if attempt < max_attempts:
                time.sleep(2 ** attempt)
            else:
                tqdm.write(f"Failed to download {filename} after {max_attempts} attempts")
                return False

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download FineWeb-Edu 100BT dataset shards")
    parser.add_argument("-n", "--num-files", type=int, default=-1,
                        help="Number of shards to download (default: -1), -1 = disable")
    parser.add_argument("-w", "--num-workers", type=int, default=4,
                        help="Number of parallel download workers (default: 4)")
    args = parser.parse_args()

    # Always ensure tokenizer is available first
    download_tokenizer()
    print()

    num = MAX_SHARD + 1 if args.num_files == -1 else min(args.num_files, MAX_SHARD + 1)
    ids_to_download = list(range(num))
    print(f"Downloading up to {len(ids_to_download)} shards using {args.num_workers} workers...")
    print(f"Target directory: {DATA_DIR}")
    print()
    results = []
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(download_single_file, i): i for i in ids_to_download}
        with tqdm(total=len(futures), desc="Shards", unit="file") as pbar:
            for future in as_completed(futures):
                results.append(future.result())
                pbar.update(1)

    # Report results
    successful = sum(1 for success in results if success)
    print(f"Done! {successful}/{len(ids_to_download)} shards ready in {DATA_DIR}")
