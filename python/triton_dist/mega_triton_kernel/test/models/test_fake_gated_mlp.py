################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################
"""Hello-world style MegaKernel test with a small gated MLP model.

This script builds a synthetic model composed of five linear layers followed by a
SiLU activation-and-multiply stage.  It exercises the MegaKernel compiler with a
configuration that spans multiple execution tiles while staying well within the
memory budget of two 24GB GPUs.  With the default ``batch=512`` and
``hidden=1024`` dimensions, each linear launch covers thousands of tiles,
making the example fast to compile but still representative.  The goal is to
provide a concise example that can be profiled with both the torch profiler
(``--profile``) and the MegaKernel intra-kernel profiler
(``--intra-kernel-profile``).
"""

import argparse
import math
import os
from dataclasses import dataclass
from typing import List

import torch
import triton

from triton_dist.mega_triton_kernel import ModelBuilder
from triton_dist.mega_triton_kernel.test.torch_impl_utils import torch_gate_silu_mul_up
from triton_dist.utils import (
    finalize_distributed,
    get_torch_prof_ctx,
    initialize_distributed,
)


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


@dataclass
class FakeGatedMLPConfig:
    batch_size: int = 512
    hidden_size: int = 1024
    num_layers: int = 5
    dtype: torch.dtype = torch.bfloat16


class FakeGatedMLP:
    """Simple gated MLP composed of Linear + SiLU*Up blocks."""

    def __init__(
        self,
        builder: ModelBuilder,
        config: FakeGatedMLPConfig,
        seed: int,
    ) -> None:
        self._builder = builder
        self.config = config
        self._rng = torch.Generator(device="cuda")
        self._rng.manual_seed(seed)
        self.weights: List[torch.Tensor] = []
        self._init_parameters()
        self.input_buffer = torch.empty(
            (config.batch_size, config.hidden_size),
            dtype=config.dtype,
            device=torch.cuda.current_device(),
        )
        self.output_buffer = self._build_graph(self.input_buffer)
        self._builder.compile()
        torch.cuda.synchronize()

    def _init_parameters(self) -> None:
        for _ in range(self.config.num_layers):
            weight = torch.randn(
                (self.config.hidden_size * 2, self.config.hidden_size),
                dtype=self.config.dtype,
                device=torch.cuda.current_device(),
                generator=self._rng,
            )
            self.weights.append(weight)

    def _build_graph(self, hidden_states: torch.Tensor) -> torch.Tensor:
        current = hidden_states
        for layer_idx, weight in enumerate(self.weights):
            fc_out = torch.empty(
                (self.config.batch_size, self.config.hidden_size * 2),
                dtype=self.config.dtype,
                device=hidden_states.device,
            )
            self._builder.make_linear(current, weight, fc_out, layer_id=layer_idx)

            act_out = torch.empty(
                (self.config.batch_size, self.config.hidden_size),
                dtype=self.config.dtype,
                device=hidden_states.device,
            )
            self._builder.make_silu_mul_up(fc_out, act_out, layer_id=layer_idx)
            current = act_out
        return current

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.input_buffer.copy_(hidden_states)
        self._builder.run()
        return self.output_buffer

    def reference_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ref = hidden_states.to(torch.float32)
        for weight in self.weights:
            fc = torch.matmul(ref, weight.to(torch.float32).transpose(0, 1))
            ref = torch_gate_silu_mul_up(fc)
        return ref.to(hidden_states.dtype)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", default="bfloat16", choices=DTYPE_MAP.keys())
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile", action="store_true", help="Enable torch profiler")
    parser.add_argument(
        "--intra-kernel-profile",
        action="store_true",
        help="Enable MegaKernel intra-kernel profiling",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = DTYPE_MAP[args.dtype]

    tp_group = initialize_distributed(seed=args.seed)
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))

    config = FakeGatedMLPConfig(
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dtype=dtype,
    )

    if rank == 0:
        # The current linear/activation kernels use 16x16 tile shapes, so report how many
        # tiles each layer will cover to help spot configurations that may compile slowly.
        tiles_linear = math.ceil(config.batch_size / 16) * math.ceil((config.hidden_size * 2) / 16)
        tiles_silu = math.ceil(config.batch_size / 16) * math.ceil(config.hidden_size / 16)
        print(
            "Estimated tiles per layer -- linear: "
            f"{tiles_linear:,}, silu: {tiles_silu:,}"
        )

    def alloc_fn(size, alignment, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    builder = ModelBuilder(
        rank=rank,
        world_size=world_size,
        local_world_size=local_world_size,
        enable_profiling=args.intra_kernel_profile,
    )

    model = FakeGatedMLP(builder=builder, config=config, seed=args.seed + rank)

    inputs = torch.randn(
        (config.batch_size, config.hidden_size),
        dtype=dtype,
        device=torch.cuda.current_device(),
        generator=torch.Generator(device="cuda").manual_seed(args.seed + rank * 3),
    )

    prof_ctx = get_torch_prof_ctx(args.profile)
    with prof_ctx as prof:
        outputs = model.forward(inputs)
        torch.cuda.synchronize()
        if prof is not None:
            prof.step()
        if args.intra_kernel_profile:
            builder.dump_trace(trace_file_prefix=f"MEGA_KERNEL_TRACE_rank{rank}")

    if args.profile and prof is not None:
        prof_dir = "prof"
        os.makedirs(prof_dir, exist_ok=True)
        trace_path = os.path.join(prof_dir, f"fake_gated_mlp_rank{rank}.json.gz")
        prof.export_chrome_trace(trace_path)

    ref_outputs = model.reference_forward(inputs).to(torch.float32)
    diff = (outputs.to(torch.float32) - ref_outputs).abs()
    max_diff = diff.max()
    mean_diff = diff.mean()

    if rank == 0:
        print(
            f"Ran fake gated MLP with shape [batch={config.batch_size}, hidden={config.hidden_size}]"
        )
        print(f"Max error: {max_diff.item():.6f}, Mean error: {mean_diff.item():.6f}")

    torch.distributed.barrier(tp_group)
    builder.finalize()
    torch.distributed.barrier(tp_group)
    finalize_distributed()


if __name__ == "__main__":
    main()

