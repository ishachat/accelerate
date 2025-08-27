# tests/test_utils_collect_epoch.py
import torch
from torch.utils.data import DataLoader, TensorDataset

from accelerate import Accelerator
from accelerate.utils.collect import collect_epoch_for_metrics


def test_collect_epoch_basic_cpu():
    N = 100
    x = torch.randn(N, 4)
    ds = TensorDataset(x)
    dl = DataLoader(ds, batch_size=16)

    acc = Accelerator()  # single process on CPU
    dl = acc.prepare(dl)

    def step_fn(batch):
        (bx,) = batch
        return bx  # tensor [b, 4]

    out = collect_epoch_for_metrics(acc, dl, step_fn, expected_len=N)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (N, 4)
    assert torch.allclose(out, x)


def test_collect_epoch_trims_to_expected_len():
    # Simulate final step not aligning exactly with N
    N = 50
    x = torch.arange(N).float().unsqueeze(1)  # [N,1]
    ds = TensorDataset(x)
    dl = DataLoader(ds, batch_size=13)  # steps: 13, 13, 13, 11

    acc = Accelerator()
    dl = acc.prepare(dl)

    out = acc.collect_epoch_for_metrics(dl, lambda b: b[0], cat_dim=0, expected_len=N)
    assert out.shape == (N, 1)
    assert torch.allclose(out, x)


def test_collect_epoch_stream_to_callback():
    N = 32
    x = torch.randn(N, 3)
    ds = TensorDataset(x)
    dl = DataLoader(ds, batch_size=8)

    acc = Accelerator()
    dl = acc.prepare(dl)

    chunks = []

    def writer(chunk, step_idx):
        # chunk is CPU tensor of shape [b_step, 3]
        assert isinstance(chunk, torch.Tensor)
        chunks.append((step_idx, chunk.clone()))

    summary = collect_epoch_for_metrics(acc, dl, lambda b: b[0], stream_to=writer)
    # 4 steps × 8 = 32 rows
    assert summary["steps"] == 4
    assert summary["count"] == 32
    assert len(chunks) == 4
    # preserve order
    assert chunks[0][0] == 0 and chunks[-1][0] == 3
    cat = torch.cat([c for _, c in chunks], dim=0)
    assert torch.allclose(cat, x)
