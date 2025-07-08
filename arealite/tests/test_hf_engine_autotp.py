# Copyright 2025 Ant Group Inc.
# Licensed under the Apache License, Version 2.0

"""Test script for HF Engine implementation."""

from typing import Dict
import os
import pytest
import torch
import deepspeed

from arealite.api.cli_args import (
    EngineBackendConfig,
    EngineConfig,
    MicroBatchSpec,
    ModelFamily,
    OptimizerConfig,
    TrainingArgs,
)
from arealite.api.engine_api import EngineFactory
from arealite.api.io_struct import FinetuneSpec
from arealite.utils import (
    compute_varlen_position_indices,
    split_dict_tensor_with_cu_seqlens,
)
from realhf.impl.model.utils.padding import unpad_input
import torch.distributed as dist
VOCAB_SIZE = 100


# @pytest.fixture(scope="module")
def mock_input(local_rank, bs: int = 3, min_seqlen: int = 3, max_seqlen: int = 12) -> Dict:
    """Create mock input data for testing."""
    seqlens = torch.randint(
        min_seqlen, max_seqlen, (bs,), dtype=torch.int, device=f"cuda:{local_rank}"
    )
    max_seqlen = int(max(seqlens))
    input_ids = torch.randint(
        0, VOCAB_SIZE, (bs, max_seqlen), dtype=torch.long, device=f"cuda:{local_rank}"
    )

    attn_mask = torch.zeros((bs, max_seqlen), dtype=torch.bool, device=f"cuda:{local_rank}")
    attn_mask[
        torch.arange(0, max_seqlen, device=f"cuda:{local_rank}").unsqueeze(0) < seqlens.unsqueeze(1)
    ] = 1

    packed_input_ids, indices, cu_seqlens, max_seqlen = unpad_input(
        input_ids, attn_mask
    )

    assert torch.allclose(
        cu_seqlens, torch.nn.functional.pad(seqlens.cumsum(0, dtype=torch.int), (1, 0))
    )
    position_ids = compute_varlen_position_indices(int(sum(seqlens)), cu_seqlens)

    return dict(
        input_ids=packed_input_ids.unsqueeze(0),
        attention_mask=None,
        position_ids=position_ids.unsqueeze(0),
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        use_cache=False,
    )


def mock_loss_fn(logits: torch.Tensor, input_data: Dict) -> torch.Tensor:
    """Mock loss function for testing."""
    return torch.mean(logits)


if __name__ == "__main__":
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if not torch.distributed.is_initialized():
        deepspeed.init_distributed(dist_backend="nccl")

    engine_config = EngineConfig(
        type=ModelFamily("qwen2", False),
        path="Qwen/Qwen2.5-0.5B",
        gradient_checkpointing=False,
        optimizer=OptimizerConfig(),
        backend=EngineBackendConfig(type="hf"),
    )

    mock_args = TrainingArgs(n_nodes=1, n_gpus_per_node=2)

    engine_factory = EngineFactory(mock_args)
    engine = engine_factory.make_engine(engine_config)
    ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=100, train_batch_size=2)
    engine.init_distributed(None, ft_spec)

    print("✓ Engine created successfully")

    input = mock_input(local_rank=local_rank)
    x2 = (
        engine.forward(
            input_=input,
            mb_spec=MicroBatchSpec(n_mbs=2),
            aggregate_fn=lambda x: torch.cat(x, dim=1),
        )
        .squeeze(0)
        .mean(-1)
    )
    x1 = (
        engine.forward(
            input_=input,
            mb_spec=MicroBatchSpec(n_mbs=1),
            aggregate_fn=lambda x: torch.cat(x, dim=1),
        )
        .squeeze(0)
        .mean(-1)
    )
    input_ids = input["input_ids"].squeeze(0)
    assert x1.shape[0] == input_ids.shape[0]
    assert x2.shape[0] == input_ids.shape[0]
    assert torch.allclose(x1, x2, atol=1e-2, rtol=1e-2), (x1 - x2).abs().max().item()

