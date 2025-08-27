# Copyright 2025 The HuggingFace Team.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

from typing import Callable, Iterable, Optional, Any
import torch

from ..accelerator import Accelerator


def collect_epoch_for_metrics(
    accelerator: Accelerator,
    step_iter: Iterable,
    step_fn: Callable[[Any], torch.Tensor],
    *,
    cat_dim: int = 0,
    expected_len: Optional[int] = None,
    stream_to: Optional[Callable[[torch.Tensor, int], None]] = None,
):
    """
    Stream per-step outputs across processes (via ``gather_for_metrics``) and return an epoch-level tensor.

    - Calls :meth:`Accelerator.gather_for_metrics` each step, so tail duplicates introduced by even batches
      are removed automatically.
    - On the main process, accumulates CPU chunks and concatenates along ``cat_dim`` at the end.
    - If ``expected_len`` is provided, the result is sliced to this length as a final safety guard.
    - If ``stream_to`` is provided, it is called as ``stream_to(chunk, step_idx)`` on the main process for each
      gathered chunk, and the function returns a small summary dict instead of a big tensor.

    Parameters
    ----------
    accelerator: Accelerator
        The accelerator instance driving the run.
    step_iter: Iterable
        An iterable over the evaluation/inference steps (usually your prepared dataloader).
    step_fn: Callable[[batch], torch.Tensor]
        A function mapping a batch to a tensor of shape ``[b, ...]``.
    cat_dim: int, optional (default=0)
        Dimension along which to concatenate gathered chunks at the end.
    expected_len: int, optional
        If provided (e.g. ``len(dataset)``), the final tensor will be sliced to this length.
    stream_to: Callable[[torch.Tensor, int], None], optional
        If provided, called with each gathered CPU tensor and the current step index on the main process.
        In this mode the function returns a summary dict ``{"steps": k, "count": n}``.

    Returns
    -------
    torch.Tensor | dict | None
        On the main process: the concatenated tensor (or a summary dict if ``stream_to`` is used).
        On non-main processes: ``None``.
    """
    if not isinstance(accelerator, Accelerator):
        raise TypeError("`accelerator` must be an instance of Accelerator.")

    chunks = [] if (accelerator.is_main_process and stream_to is None) else None
    total = 0

    for step_idx, batch in enumerate(step_iter):
        out = step_fn(batch)
        if not isinstance(out, torch.Tensor):
            raise TypeError(
                "collect_epoch_for_metrics (minimal version) supports tensors only. "
                "If you need objects, please gather objects per-step and stream manually."
            )

        # Gather across processes and drop duplicated tail samples.
        gathered = accelerator.gather_for_metrics(out)

        # Only main process accumulates or streams.
        if not accelerator.is_main_process:
            continue

        gathered_cpu = gathered.detach().cpu()
        if stream_to is None:
            chunks.append(gathered_cpu)
        else:
            stream_to(gathered_cpu, step_idx)

        total += gathered_cpu.shape[0]

    # Non-main returns None to keep semantics similar to other helpers.
    if not accelerator.is_main_process:
        return None

    if stream_to is not None:
        steps = (step_idx + 1) if "step_idx" in locals() else 0
        return {"steps": steps, "count": total}

    if not chunks:
        return torch.empty((0,), dtype=torch.float32)

    result = torch.cat(chunks, dim=cat_dim)
    if expected_len is not None:
        result = result[: expected_len]
    return result
