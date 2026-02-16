"""
Dataloader for pretraining, ported to MLX.

BOS-aligned bestfit:
 - Every row starts with BOS token
 - Documents packed using best-fit algorithm to minimize cropping
 - When no document fits remaining space, crops a document to fill exactly
 - 100% utilization (no padding), ~35% tokens cropped at T=2048

Compared to naive cropping, BOS-aligned loses ~35% of tokens to cropping,
but ensures that every token can attend back to the BOS token and sees the
full context of the document.

MLX port notes:
 - No DDP/multi-GPU: single device, no rank-based sharding
 - No pin_memory or GPU staging buffers (MLX unified memory)
 - Uses mx.int32 for token IDs (vocab fits in int32)
 - No .to(device) calls needed
"""

import mlx.core as mx
import pyarrow.parquet as pq

from nanochat_mlx.dataset import list_parquet_files


def _document_batches(split, resume_state_dict, tokenizer_batch_size):
    """
    Infinite iterator over document batches (list of text strings) from parquet files.

    Single-device version: no DDP rank sharding, reads every row group sequentially.

    Each yield is (text_batch, (pq_idx, rg_idx, epoch)) where text_batch is a list
    of document strings, indices track position for resumption, and epoch counts
    how many times we've cycled through the dataset (starts at 1).
    """
    parquet_paths = list_parquet_files()
    assert len(parquet_paths) != 0, "No dataset parquet files found, did you run dataset.py?"
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]

    resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
    resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
    resume_epoch = resume_state_dict.get("epoch", 1) if resume_state_dict is not None else 1
    first_pass = True
    epoch = resume_epoch

    while True:  # iterate infinitely (multi-epoch)
        pq_idx = resume_pq_idx if first_pass else 0
        while pq_idx < len(parquet_paths):
            filepath = parquet_paths[pq_idx]
            pf = pq.ParquetFile(filepath)
            # Start from resume point if resuming on same file, otherwise from 0
            if first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                rg_idx = resume_rg_idx + 1  # advance by 1 so we don't repeat data
                if rg_idx >= pf.num_row_groups:
                    pq_idx += 1
                    continue
                resume_rg_idx = None  # only do this once
            else:
                rg_idx = 0
            while rg_idx < pf.num_row_groups:
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i + tokenizer_batch_size], (pq_idx, rg_idx, epoch)
                rg_idx += 1
            pq_idx += 1
        first_pass = False
        epoch += 1


def tokenizing_data_loader_with_state_bos_bestfit(
    tokenizer, B, T, split,
    tokenizer_threads=4, tokenizer_batch_size=128,
    resume_state_dict=None,
    buffer_size=1000,
):
    """
    BOS-aligned dataloader with Best-Fit Cropping, ported to MLX.

    Reduces token waste compared to simple greedy cropping by searching a buffer
    for documents that fit well, while maintaining 100% utilization (no padding).

    Algorithm for each row:
    1. From buffered docs, pick the LARGEST doc that fits entirely
    2. Repeat until no doc fits
    3. When nothing fits, crop a doc to fill remaining space exactly

    Key properties:
    - Every row starts with BOS
    - 100% utilization (no padding, every token is trained on)
    - Approximately 35% of all tokens are discarded due to cropping

    Yields:
        inputs: mx.array of shape (B, T), dtype mx.int32
        targets: mx.array of shape (B, T), dtype mx.int32
        state_dict: dict with pq_idx, rg_idx, epoch for resumption
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    row_capacity = T + 1  # +1 because inputs=row[:-1], targets=row[1:]
    batches = _document_batches(split, resume_state_dict, tokenizer_batch_size)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    pq_idx, rg_idx, epoch = 0, 0, 1

    def refill_buffer():
        nonlocal pq_idx, rg_idx, epoch
        doc_batch, (pq_idx, rg_idx, epoch) = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
        for tokens in token_lists:
            doc_buffer.append(tokens)

    # Pre-allocate row buffer as a list of lists (pure Python, then convert to mx at yield)
    # MLX unified memory means no staging buffers or HtoD transfers needed
    rows = [[0] * row_capacity for _ in range(B)]

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                # Ensure buffer has documents
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                # Find largest doc that fits entirely
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    # Place the best-fitting document
                    doc = doc_buffer.pop(best_idx)
                    doc_len = len(doc)
                    rows[row_idx][pos:pos + doc_len] = doc
                    pos += doc_len
                else:
                    # No doc fits - crop shortest in buffer to fill remaining and minimize waste
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    rows[row_idx][pos:pos + remaining] = doc[:remaining]
                    pos += remaining

        # Convert to mx.array: rows is (B, T+1), then slice into inputs and targets
        # Using mx.int32 since the vocabulary fits comfortably in int32 range
        batch_array = mx.array(rows, dtype=mx.int32)
        inputs = batch_array[:, :-1]   # (B, T)
        targets = batch_array[:, 1:]   # (B, T)

        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}

        yield inputs, targets, state_dict


def tokenizing_data_loader_bos_bestfit(*args, **kwargs):
    """Helper that omits state_dict from yields."""
    for inputs, targets, state_dict in tokenizing_data_loader_with_state_bos_bestfit(*args, **kwargs):
        yield inputs, targets
