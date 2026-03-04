#!/usr/bin/env python3
"""Benchmark FlashInfer fused allreduce+rmsnorm against standard allreduce+rmsnorm.

This is the maintained SGLang benchmark for FlashInfer fused collective APIs.
It uses FlashInfer's unified workspace API:
  - create_allreduce_fusion_workspace
  - allreduce_fusion

Usage:
  torchrun --nproc_per_node=8 \
    benchmark/kernels/flashinfer_allreduce_fusion/benchmark_fused_collective.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import statistics
import sys
import types
from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

# Allow direct execution from repo root without installing the package.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_PYTHON_DIR = _REPO_ROOT / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))
if "sglang" not in sys.modules:
    sglang_pkg = types.ModuleType("sglang")
    sglang_pkg.__path__ = [str(_PYTHON_DIR / "sglang")]  # type: ignore[attr-defined]
    sys.modules["sglang"] = sglang_pkg

from sglang.srt.distributed.device_communicators.flashinfer_utils import (
    create_mnnvl_comm_backend,
)

logger = logging.getLogger(__name__)

DEFAULT_SEQUENCE_LENGTHS = [128, 512, 1024, 2048, 4096]


@dataclass
class BenchmarkEntry:
    seq_len: int
    hidden_dim: int
    dtype: str
    residual_mode: str
    backend: str
    mode: str
    latency_ms: float
    baseline_ms: float
    speedup: float
    correctness_ok: Optional[bool] = None
    max_abs_error: Optional[float] = None


def _dtype_from_arg(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name}")


def _rmsnorm_reference(
    x: torch.Tensor, gamma: torch.Tensor, eps: float
) -> torch.Tensor:
    x_fp32 = x.to(torch.float32)
    variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    out = x_fp32 * torch.rsqrt(variance + eps)
    out = out * gamma.to(torch.float32)
    return out.to(x.dtype)


def _allreduce_mean_cpu(value: float, group: ProcessGroup, world_size: int) -> float:
    buf = torch.tensor([value], dtype=torch.float64, device="cpu")
    dist.all_reduce(buf, op=dist.ReduceOp.SUM, group=group)
    return float(buf.item() / world_size)


def _benchmark_op(
    *,
    fn,
    warmup: int,
    trials: int,
    ops_per_trial: int,
    device: torch.device,
    cpu_group: ProcessGroup,
    world_size: int,
) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)

    latencies_ms: list[float] = []
    for _ in range(trials):
        dist.barrier(group=cpu_group)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(ops_per_trial):
            fn()
        end.record()

        torch.cuda.synchronize(device)
        latencies_ms.append(start.elapsed_time(end) / ops_per_trial)

    local_mean = statistics.mean(latencies_ms)
    return _allreduce_mean_cpu(local_mean, cpu_group, world_size)


def _standard_allreduce_rmsnorm(
    *,
    input_tensor: torch.Tensor,
    residual_tensor: torch.Tensor,
    rms_gamma: torch.Tensor,
    rms_eps: float,
    device_group: ProcessGroup,
) -> torch.Tensor:
    x = input_tensor.clone()
    dist.all_reduce(x, group=device_group)
    x = x + residual_tensor
    return _rmsnorm_reference(x, rms_gamma, rms_eps)


def _flashinfer_allreduce_rmsnorm(
    *,
    flashinfer_comm,
    workspace,
    input_tensor: torch.Tensor,
    residual_tensor: torch.Tensor,
    rms_gamma: torch.Tensor,
    rms_eps: float,
    use_oneshot: bool,
) -> torch.Tensor:
    x = input_tensor.clone()
    norm_out = torch.empty_like(x)
    residual_out = torch.empty_like(x)

    kwargs = dict(
        input=x,
        workspace=workspace,
        pattern=flashinfer_comm.AllReduceFusionPattern.kARResidualRMSNorm,
        residual_in=residual_tensor,
        residual_out=residual_out,
        norm_out=norm_out,
        rms_gamma=rms_gamma,
        rms_eps=rms_eps,
        use_oneshot=use_oneshot,
        launch_with_pdl=True,
        fp32_acc=True,
    )

    try:
        flashinfer_comm.allreduce_fusion(**kwargs)
    except TypeError:
        kwargs.pop("launch_with_pdl", None)
        kwargs.pop("fp32_acc", None)
        flashinfer_comm.allreduce_fusion(**kwargs)

    return norm_out


def _check_correctness(
    *,
    standard_out: torch.Tensor,
    fused_out: torch.Tensor,
    rtol: float,
    atol: float,
    cpu_group: ProcessGroup,
) -> tuple[bool, float]:
    max_abs_error_local = (standard_out - fused_out).abs().max()
    max_abs_error = max_abs_error_local.clone()
    dist.all_reduce(max_abs_error, op=dist.ReduceOp.MAX, group=cpu_group)

    passed_local = torch.tensor(
        [torch.allclose(standard_out, fused_out, rtol=rtol, atol=atol)],
        dtype=torch.int32,
        device="cpu",
    )
    passed_global = passed_local.clone()
    dist.all_reduce(passed_global, op=dist.ReduceOp.MIN, group=cpu_group)

    return bool(passed_global.item()), float(max_abs_error.item())


def _create_workspace(
    *,
    flashinfer_comm,
    backend: str,
    world_size: int,
    rank: int,
    max_token_num: int,
    hidden_dim: int,
    dtype: torch.dtype,
    cpu_group: ProcessGroup,
):
    kwargs = dict(
        backend=backend,
        world_size=world_size,
        rank=rank,
        max_token_num=max_token_num,
        hidden_dim=hidden_dim,
        dtype=dtype,
    )

    if backend in ("auto", "mnnvl"):
        comm_backend = create_mnnvl_comm_backend(cpu_group)
        if comm_backend is not None:
            kwargs["comm_backend"] = comm_backend

    return flashinfer_comm.create_allreduce_fusion_workspace(**kwargs)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark FlashInfer fused allreduce+rmsnorm vs standard path.",
    )
    parser.add_argument(
        "--sequence-lengths",
        type=int,
        nargs="+",
        default=DEFAULT_SEQUENCE_LENGTHS,
        help="Sequence lengths to benchmark (shape=[seq_len, hidden_dim]).",
    )
    parser.add_argument("--hidden-dim", type=int, default=8192)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["auto", "trtllm", "mnnvl"],
        default=["trtllm", "mnnvl"],
        help="FlashInfer backends to benchmark.",
    )
    parser.add_argument(
        "--residual-mode",
        choices=["with-residual", "zero-residual", "both"],
        default="both",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--ops-per-trial", type=int, default=16)
    parser.add_argument(
        "--disable-oneshot",
        action="store_true",
        help="Disable oneshot mode; benchmark twoshot only.",
    )
    parser.add_argument("--rms-eps", type=float, default=1e-6)
    parser.add_argument("--check-correctness", action="store_true")
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--atol", type=float, default=1.5e-1)
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument(
        "--cpu-backend",
        choices=["gloo"],
        default="gloo",
        help="Control-plane process group backend.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )

    args = parser.parse_args()
    args.sequence_lengths = sorted(set(args.sequence_lengths))
    return args


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    try:
        import flashinfer.comm as flashinfer_comm  # type: ignore
    except ImportError as exc:
        raise RuntimeError("flashinfer is required for this benchmark.") from exc

    if not (
        hasattr(flashinfer_comm, "allreduce_fusion")
        and hasattr(flashinfer_comm, "create_allreduce_fusion_workspace")
    ):
        raise RuntimeError("flashinfer.comm unified allreduce API is not available.")

    if not dist.is_initialized():
        dist.init_process_group(backend=args.cpu_backend, init_method="env://")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size <= 1:
        raise RuntimeError("Run with torchrun and world_size > 1.")

    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    cpu_group = dist.group.WORLD
    device_group: Optional[ProcessGroup] = None
    try:
        device_group = dist.new_group(backend="nccl")
    except Exception as exc:
        raise RuntimeError("Failed to initialize NCCL device group.") from exc

    dtype = _dtype_from_arg(args.dtype)
    max_seq_len = max(args.sequence_lengths)
    oneshot_modes = [False] if args.disable_oneshot else [True, False]
    residual_modes = (
        ["with-residual", "zero-residual"]
        if args.residual_mode == "both"
        else [args.residual_mode]
    )

    workspaces = {}
    results: list[BenchmarkEntry] = []

    try:
        for backend in args.backends:
            try:
                workspace = _create_workspace(
                    flashinfer_comm=flashinfer_comm,
                    backend=backend,
                    world_size=world_size,
                    rank=rank,
                    max_token_num=max_seq_len,
                    hidden_dim=args.hidden_dim,
                    dtype=dtype,
                    cpu_group=cpu_group,
                )
                workspaces[backend] = workspace
                if rank == 0:
                    logger.info(
                        "Created workspace backend=%s resolved_backend=%s",
                        backend,
                        getattr(workspace, "backend", backend),
                    )
            except Exception as exc:
                if rank == 0:
                    logger.warning(
                        "Skipping backend=%s, workspace init failed: %s", backend, exc
                    )

        if not workspaces:
            raise RuntimeError("No FlashInfer backend workspace could be initialized.")

        for seq_len in args.sequence_lengths:
            for residual_mode in residual_modes:
                input_tensor = torch.randn(
                    seq_len,
                    args.hidden_dim,
                    dtype=dtype,
                    device=device,
                )
                residual_tensor = (
                    torch.randn_like(input_tensor)
                    if residual_mode == "with-residual"
                    else torch.zeros_like(input_tensor)
                )
                rms_gamma = torch.ones(args.hidden_dim, dtype=dtype, device=device)

                baseline_fn = lambda: _standard_allreduce_rmsnorm(
                    input_tensor=input_tensor,
                    residual_tensor=residual_tensor,
                    rms_gamma=rms_gamma,
                    rms_eps=args.rms_eps,
                    device_group=device_group,
                )

                baseline_ms = _benchmark_op(
                    fn=baseline_fn,
                    warmup=args.warmup,
                    trials=args.trials,
                    ops_per_trial=args.ops_per_trial,
                    device=device,
                    cpu_group=cpu_group,
                    world_size=world_size,
                )

                if rank == 0:
                    logger.info(
                        "baseline seq_len=%d residual=%s: %.3f ms",
                        seq_len,
                        residual_mode,
                        baseline_ms,
                    )

                for backend, workspace in workspaces.items():
                    for use_oneshot in oneshot_modes:
                        mode_name = "oneshot" if use_oneshot else "twoshot"

                        fused_fn = lambda b=backend, ws=workspace, uo=use_oneshot: _flashinfer_allreduce_rmsnorm(
                            flashinfer_comm=flashinfer_comm,
                            workspace=ws,
                            input_tensor=input_tensor,
                            residual_tensor=residual_tensor,
                            rms_gamma=rms_gamma,
                            rms_eps=args.rms_eps,
                            use_oneshot=uo,
                        )

                        try:
                            fused_ms = _benchmark_op(
                                fn=fused_fn,
                                warmup=args.warmup,
                                trials=args.trials,
                                ops_per_trial=args.ops_per_trial,
                                device=device,
                                cpu_group=cpu_group,
                                world_size=world_size,
                            )
                        except Exception as exc:
                            if rank == 0:
                                logger.warning(
                                    "Failed backend=%s mode=%s seq_len=%d residual=%s: %s",
                                    backend,
                                    mode_name,
                                    seq_len,
                                    residual_mode,
                                    exc,
                                )
                            continue

                        correctness_ok: Optional[bool] = None
                        max_abs_error: Optional[float] = None
                        if args.check_correctness:
                            standard_out = baseline_fn()
                            fused_out = fused_fn()
                            correctness_ok, max_abs_error = _check_correctness(
                                standard_out=standard_out,
                                fused_out=fused_out,
                                rtol=args.rtol,
                                atol=args.atol,
                                cpu_group=cpu_group,
                            )

                        results.append(
                            BenchmarkEntry(
                                seq_len=seq_len,
                                hidden_dim=args.hidden_dim,
                                dtype=args.dtype,
                                residual_mode=residual_mode,
                                backend=backend,
                                mode=mode_name,
                                latency_ms=fused_ms,
                                baseline_ms=baseline_ms,
                                speedup=baseline_ms / fused_ms,
                                correctness_ok=correctness_ok,
                                max_abs_error=max_abs_error,
                            )
                        )

        if rank == 0:
            print("\n" + "=" * 140)
            print(
                "FlashInfer Fused Collective Benchmark "
                f"(world_size={world_size}, dtype={dtype}, hidden_dim={args.hidden_dim})"
            )
            print("=" * 140)
            print(
                "seq_len".ljust(10)
                + "residual".ljust(16)
                + "backend".ljust(10)
                + "mode".ljust(10)
                + "latency_ms".ljust(14)
                + "baseline_ms".ljust(14)
                + "speedup".ljust(10)
                + "correct"
            )
            print("-" * 140)
            for entry in results:
                correct_str = (
                    "N/A"
                    if entry.correctness_ok is None
                    else (
                        f"ok (max_err={entry.max_abs_error:.3e})"
                        if entry.correctness_ok
                        else f"FAIL (max_err={entry.max_abs_error:.3e})"
                    )
                )
                print(
                    f"{entry.seq_len:<10}"
                    f"{entry.residual_mode:<16}"
                    f"{entry.backend:<10}"
                    f"{entry.mode:<10}"
                    f"{entry.latency_ms:<14.3f}"
                    f"{entry.baseline_ms:<14.3f}"
                    f"{entry.speedup:<10.2f}"
                    f"{correct_str}"
                )

            if args.output_json:
                payload = {
                    "world_size": world_size,
                    "dtype": args.dtype,
                    "hidden_dim": args.hidden_dim,
                    "sequence_lengths": args.sequence_lengths,
                    "residual_mode": args.residual_mode,
                    "backends": args.backends,
                    "oneshot_modes": (
                        ["twoshot"] if args.disable_oneshot else ["oneshot", "twoshot"]
                    ),
                    "warmup": args.warmup,
                    "trials": args.trials,
                    "ops_per_trial": args.ops_per_trial,
                    "entries": [asdict(entry) for entry in results],
                }
                with open(args.output_json, "w", encoding="utf-8") as fout:
                    json.dump(payload, fout, indent=2)
                print(f"\nSaved JSON report to: {args.output_json}")

    finally:
        for workspace in workspaces.values():
            try:
                workspace.destroy()
            except Exception:
                pass

        if device_group is not None:
            try:
                dist.destroy_process_group(device_group)
            except Exception:
                pass

        try:
            dist.destroy_process_group()
        except Exception:
            pass


if __name__ == "__main__":
    main()
