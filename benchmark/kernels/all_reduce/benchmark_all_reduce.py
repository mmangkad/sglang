#!/usr/bin/env python3
"""Benchmark SGLang device communicators and recommend FlashInfer thresholds.

This script is the SGLang equivalent of vLLM's communicator benchmark. It
benchmarks available all-reduce backends on 2D tensors with shape
`[num_tokens, hidden_size]`, then optionally recommends a FlashInfer standalone
allreduce threshold for the current `(SM, world_size)` pair.

Typical usage:
  torchrun --nproc_per_node=8 \
    benchmark/kernels/all_reduce/benchmark_all_reduce.py

  torchrun --nproc_per_node=8 \
    benchmark/kernels/all_reduce/benchmark_all_reduce.py \
    --sequence-lengths 16 64 128 256 512 1024 2048 4096 \
    --threshold-candidates-mb 0.5 1 2 4 8 16 32 64
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
from dataclasses import dataclass
from typing import Callable, Optional

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

from sglang.srt.distributed.device_communicators.custom_all_reduce import (
    dispatch_custom_allreduce,
)
from sglang.srt.distributed.device_communicators.flashinfer_all_reduce import (
    _FI_ALLREDUCE_MAX_SIZE_MB,
    FlashInferAllReduce,
    MiB,
)
from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator
from sglang.srt.distributed.device_communicators.torch_symm_mem import (
    TorchSymmMemCommunicator,
)

logger = logging.getLogger(__name__)

DEFAULT_SEQUENCE_LENGTHS = [16, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
DEFAULT_THRESHOLD_CANDIDATES_MB = [0.5, 1, 2, 4, 8, 16, 32, 64, 96, 128]


@dataclass
class CommunicatorVariant:
    name: str
    should_use: Callable[[torch.Tensor], bool]
    run: Callable[[torch.Tensor], Optional[torch.Tensor]]
    cleanup: Callable[[], None]


@dataclass
class ThresholdRecommendation:
    backend: str
    candidate_mb: float
    mean_latency_ms: float
    baseline_mean_latency_ms: float
    overall_speedup: float
    worst_regression: float
    flashinfer_usage_ratio: float
    constrained: bool


class CommunicatorBenchmark:
    def __init__(
        self,
        *,
        args: argparse.Namespace,
        rank: int,
        world_size: int,
        device: torch.device,
        cpu_group: ProcessGroup,
        device_group: Optional[ProcessGroup],
    ):
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.cpu_group = cpu_group
        self.device_group = device_group
        self.hidden_size = args.hidden_size
        self.dtype = _dtype_from_arg(args.dtype)

        self._cleanup_fns: list[Callable[[], None]] = []
        self.variants = self._build_variants()

    def close(self) -> None:
        for cleanup_fn in reversed(self._cleanup_fns):
            try:
                cleanup_fn()
            except Exception as exc:
                logger.debug("Cleanup failed: %s", exc)
        self._cleanup_fns.clear()

    def _register_cleanup(self, fn: Callable[[], None]) -> None:
        self._cleanup_fns.append(fn)

    def _build_variants(self) -> dict[str, CommunicatorVariant]:
        variants: dict[str, CommunicatorVariant] = {}

        def register(variant: CommunicatorVariant) -> None:
            variants[variant.name] = variant

        if self.device_group is not None:
            register(
                CommunicatorVariant(
                    name="torch_nccl",
                    should_use=lambda _t: True,
                    run=lambda t: _run_torch_nccl(t, self.device_group),
                    cleanup=lambda: None,
                )
            )

        max_seq_len = max(self.args.sequence_lengths)
        max_size_override = (
            max_seq_len
            * self.hidden_size
            * torch.tensor([], dtype=self.dtype).element_size()
        ) + 1

        try:
            ca_cls = dispatch_custom_allreduce()
            ca_comm = ca_cls(
                group=self.cpu_group,
                device=self.device,
                max_size=max_size_override,
            )
            if not getattr(ca_comm, "disabled", True):
                register(
                    CommunicatorVariant(
                        name="custom_allreduce",
                        should_use=lambda t, c=ca_comm: _should_use_custom(c, t),
                        run=lambda t, c=ca_comm: c.custom_all_reduce(t),
                        cleanup=lambda c=ca_comm: _close_if_exists(c),
                    )
                )
                self._register_cleanup(lambda c=ca_comm: _close_if_exists(c))
            else:
                _close_if_exists(ca_comm)
        except Exception as exc:
            if self.rank == 0:
                logger.warning("CustomAllreduce unavailable: %s", exc)

        try:
            pynccl_comm = PyNcclCommunicator(
                group=self.cpu_group,
                device=self.device,
                use_current_stream=True,
            )
            if not getattr(pynccl_comm, "disabled", True):
                register(
                    CommunicatorVariant(
                        name="pynccl",
                        should_use=lambda _t: True,
                        run=lambda t, c=pynccl_comm: _run_pynccl(c, t),
                        cleanup=lambda: None,
                    )
                )
            else:
                _close_if_exists(pynccl_comm)
        except Exception as exc:
            if self.rank == 0:
                logger.warning("PyNcclCommunicator unavailable: %s", exc)

        try:
            symm_comm = TorchSymmMemCommunicator(
                group=self.cpu_group,
                device=self.device,
            )
            if not getattr(symm_comm, "disabled", True):
                register(
                    CommunicatorVariant(
                        name="torch_symm_mem",
                        should_use=lambda t, c=symm_comm: c.should_torch_symm_mem_allreduce(
                            t
                        ),
                        run=lambda t, c=symm_comm: c.all_reduce(t),
                        cleanup=lambda: None,
                    )
                )
            else:
                _close_if_exists(symm_comm)
        except Exception as exc:
            if self.rank == 0:
                logger.warning("TorchSymmMemCommunicator unavailable: %s", exc)

        for backend in self.args.flashinfer_backends:
            try:
                fi_comm = FlashInferAllReduce(
                    group=self.cpu_group,
                    device=self.device,
                    backend=backend,
                )
                if getattr(fi_comm, "disabled", True):
                    fi_comm.destroy()
                    continue

                # For tuning, benchmark with a large workspace ceiling so we can
                # evaluate multiple threshold candidates from one measurement pass.
                benchmark_workspace_mb = self.args.flashinfer_benchmark_max_workspace_mb
                if benchmark_workspace_mb is not None:
                    fi_comm.max_workspace_size = int(benchmark_workspace_mb * MiB)
                    fi_comm.max_num_tokens = 0
                    fi_comm.destroy()

                register(
                    CommunicatorVariant(
                        name=f"flashinfer_{backend}",
                        should_use=lambda t, c=fi_comm: c.should_use_fi_ar(t),
                        run=lambda t, c=fi_comm: c.all_reduce(t),
                        cleanup=lambda c=fi_comm: c.destroy(),
                    )
                )
                self._register_cleanup(lambda c=fi_comm: c.destroy())
            except Exception as exc:
                if self.rank == 0:
                    logger.warning("FlashInfer (%s) unavailable: %s", backend, exc)

        return variants

    def benchmark_all(self) -> dict[int, dict[str, float]]:
        results: dict[int, dict[str, float]] = {}

        for seq_len in self.args.sequence_lengths:
            if self.rank == 0:
                logger.info(
                    "Benchmarking seq_len=%d (shape=(%d, %d), dtype=%s)",
                    seq_len,
                    seq_len,
                    self.hidden_size,
                    self.dtype,
                )

            per_seq: dict[str, float] = {}
            for name, variant in self.variants.items():
                latency = self._benchmark_one_variant(seq_len=seq_len, variant=variant)
                if latency is not None:
                    per_seq[name] = latency
            results[seq_len] = per_seq

            dist.barrier(group=self.cpu_group)

        return results

    def _benchmark_one_variant(
        self,
        *,
        seq_len: int,
        variant: CommunicatorVariant,
    ) -> Optional[float]:
        tensor = torch.randn(
            seq_len,
            self.hidden_size,
            dtype=self.dtype,
            device=self.device,
        )

        try:
            if not variant.should_use(tensor):
                return None
        except Exception as exc:
            logger.debug("%s should_use failed: %s", variant.name, exc)
            return None

        # One-time init to avoid charging communicator setup to benchmark loops.
        try:
            out = variant.run(tensor)
        except Exception as exc:
            logger.debug("%s init run failed: %s", variant.name, exc)
            return None

        if out is None:
            return None
        tensor = out

        torch.cuda.synchronize(self.device)

        for _ in range(self.args.num_warmup):
            out = variant.run(tensor)
            if out is None:
                return None
            tensor = out

        torch.cuda.synchronize(self.device)

        latencies_ms: list[float] = []
        for _ in range(self.args.num_trials):
            dist.barrier(group=self.cpu_group)

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            current = tensor
            start.record()
            for _ in range(self.args.ops_per_trial):
                out = variant.run(current)
                if out is None:
                    return None
                current = out
            end.record()

            torch.cuda.synchronize(self.device)
            latencies_ms.append(start.elapsed_time(end) / self.args.ops_per_trial)
            tensor = current

        local_mean = statistics.mean(latencies_ms)
        global_mean = _allreduce_mean_cpu(local_mean, self.cpu_group, self.world_size)
        return global_mean


def _run_torch_nccl(tensor: torch.Tensor, group: ProcessGroup) -> torch.Tensor:
    dist.all_reduce(tensor, group=group)
    return tensor


def _run_pynccl(
    comm: PyNcclCommunicator, tensor: torch.Tensor
) -> Optional[torch.Tensor]:
    stream = torch.cuda.current_stream(device=tensor.device)
    with comm.change_state(enable=True, stream=stream):
        out = comm.outplace_all_reduce(tensor)
    return out


def _should_use_custom(comm, tensor: torch.Tensor) -> bool:
    fn = getattr(comm, "should_custom_ar", None)
    if callable(fn):
        return bool(fn(tensor))
    return not bool(getattr(comm, "disabled", True))


def _close_if_exists(obj: object) -> None:
    close_fn = getattr(obj, "close", None)
    if callable(close_fn):
        close_fn()


def _allreduce_mean_cpu(value: float, group: ProcessGroup, world_size: int) -> float:
    buf = torch.tensor([value], dtype=torch.float64, device="cpu")
    dist.all_reduce(buf, op=dist.ReduceOp.SUM, group=group)
    return float(buf.item() / world_size)


def _dtype_from_arg(dtype: str) -> torch.dtype:
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {dtype}")


def _tensor_size_mb(seq_len: int, hidden_size: int, dtype: torch.dtype) -> float:
    element_size = torch.tensor([], dtype=dtype).element_size()
    return (seq_len * hidden_size * element_size) / MiB


def _build_baseline_map(
    *,
    results: dict[int, dict[str, float]],
    preferred: str,
) -> dict[int, Optional[float]]:
    all_names = sorted({k for seq in results.values() for k in seq})
    non_flashinfer = [name for name in all_names if not name.startswith("flashinfer_")]

    baseline: dict[int, Optional[float]] = {}
    for seq_len, seq_result in results.items():
        if preferred == "best_non_flashinfer":
            candidates = [
                seq_result[name] for name in non_flashinfer if name in seq_result
            ]
            baseline[seq_len] = min(candidates) if candidates else None
            continue

        preferred_value = seq_result.get(preferred)
        if preferred_value is not None:
            baseline[seq_len] = preferred_value
            continue

        # Fallback if the preferred baseline is unavailable at this size.
        candidates = [seq_result[name] for name in non_flashinfer if name in seq_result]
        baseline[seq_len] = min(candidates) if candidates else None

    return baseline


def _recommend_thresholds(
    *,
    args: argparse.Namespace,
    results: dict[int, dict[str, float]],
    dtype: torch.dtype,
) -> list[ThresholdRecommendation]:
    baseline_map = _build_baseline_map(
        results=results, preferred=args.threshold_baseline
    )
    seq_lens = sorted(results)

    recommendations: list[ThresholdRecommendation] = []
    for backend in args.flashinfer_backends:
        fi_name = f"flashinfer_{backend}"
        if not any(fi_name in row for row in results.values()):
            continue

        candidate_stats: list[tuple[float, float, float, float, float]] = []
        # (candidate_mb, mean_latency_ms, baseline_mean_latency_ms,
        #  worst_regression, usage_ratio)

        for candidate_mb in sorted(args.threshold_candidates_mb):
            chosen: list[float] = []
            baselines: list[float] = []
            regressions: list[float] = []
            use_fi_count = 0
            valid = True

            for seq_len in seq_lens:
                base = baseline_map.get(seq_len)
                if base is None:
                    valid = False
                    break

                seq_result = results[seq_len]
                fi_latency = seq_result.get(fi_name)
                size_mb = _tensor_size_mb(seq_len, args.hidden_size, dtype)
                use_fi = fi_latency is not None and size_mb <= candidate_mb

                chosen_latency = fi_latency if use_fi else base
                if chosen_latency is None:
                    valid = False
                    break

                chosen.append(chosen_latency)
                baselines.append(base)
                regressions.append(chosen_latency / base)
                if use_fi:
                    use_fi_count += 1

            if not valid or not chosen:
                continue

            mean_latency = statistics.mean(chosen)
            baseline_mean = statistics.mean(baselines)
            worst_regression = max(regressions)
            usage_ratio = use_fi_count / len(seq_lens)
            candidate_stats.append(
                (
                    candidate_mb,
                    mean_latency,
                    baseline_mean,
                    worst_regression,
                    usage_ratio,
                )
            )

        if not candidate_stats:
            continue

        constrained = [
            item for item in candidate_stats if item[3] <= args.threshold_max_regression
        ]
        pool = constrained if constrained else candidate_stats

        best = min(pool, key=lambda x: x[1])
        candidate_mb, mean_latency, baseline_mean, worst_regression, usage_ratio = best
        recommendations.append(
            ThresholdRecommendation(
                backend=backend,
                candidate_mb=candidate_mb,
                mean_latency_ms=mean_latency,
                baseline_mean_latency_ms=baseline_mean,
                overall_speedup=baseline_mean / mean_latency,
                worst_regression=worst_regression,
                flashinfer_usage_ratio=usage_ratio,
                constrained=bool(constrained),
            )
        )

    return recommendations


def _print_results_table(
    *,
    results: dict[int, dict[str, float]],
    args: argparse.Namespace,
    dtype: torch.dtype,
) -> None:
    all_names = sorted({name for seq in results.values() for name in seq})

    print("\n" + "=" * 160)
    print(
        "Device Communicator Benchmark "
        f"(world_size={dist.get_world_size()}, dtype={dtype}, hidden_size={args.hidden_size})"
    )
    print("=" * 160)

    header = ["Tensor Shape", "Tensor Size"] + all_names + ["Best"]
    widths = [20, 14] + [18 for _ in all_names] + [24]

    def fmt_row(values: list[str]) -> str:
        return "".join(v.ljust(w) for v, w in zip(values, widths))

    print(fmt_row(header))
    print("-" * sum(widths))

    for seq_len in sorted(results):
        seq_result = results[seq_len]
        shape = f"({seq_len}, {args.hidden_size})"
        size_mb = _tensor_size_mb(seq_len, args.hidden_size, dtype)
        row = [shape, f"{size_mb:.2f} MB"]
        for name in all_names:
            value = seq_result.get(name)
            row.append("N/A" if value is None else f"{value:.3f} ms")

        if seq_result:
            best_name = min(seq_result, key=seq_result.get)
            row.append(f"{best_name} ({seq_result[best_name]:.3f} ms)")
        else:
            row.append("N/A")

        print(fmt_row(row))


def _print_recommendations(
    *,
    args: argparse.Namespace,
    recommendations: list[ThresholdRecommendation],
) -> None:
    if not recommendations:
        print("\nNo FlashInfer threshold recommendation could be generated.")
        return

    print("\n" + "=" * 120)
    print("Recommended FlashInfer Standalone AllReduce Thresholds")
    print("=" * 120)
    print(
        "backend".ljust(12)
        + "threshold_mb".ljust(14)
        + "mean_ms".ljust(12)
        + "baseline_ms".ljust(12)
        + "speedup".ljust(10)
        + "worst_reg".ljust(12)
        + "fi_usage".ljust(10)
        + "constraint"
    )
    print("-" * 120)

    for rec in recommendations:
        constraint_state = "applied" if rec.constrained else "relaxed"
        print(
            f"{rec.backend:<12}"
            f"{rec.candidate_mb:<14g}"
            f"{rec.mean_latency_ms:<12.3f}"
            f"{rec.baseline_mean_latency_ms:<12.3f}"
            f"{rec.overall_speedup:<10.2f}"
            f"{rec.worst_regression:<12.3f}"
            f"{rec.flashinfer_usage_ratio:<10.2f}"
            f"{constraint_state}"
        )

    try:
        major, minor = torch.cuda.get_device_capability()
        sm = major * 10 + minor
    except Exception:
        sm = None

    if sm is None:
        return

    selected_backend = args.threshold_backend_for_map
    selected = next((r for r in recommendations if r.backend == selected_backend), None)
    if selected is None:
        selected = recommendations[0]

    current_row = dict(_FI_ALLREDUCE_MAX_SIZE_MB.get(sm, {}))
    current_row[dist.get_world_size()] = selected.candidate_mb

    print("\nSuggested threshold update snippet:")
    print(
        f"# SM{sm}, world_size={dist.get_world_size()}, backend_hint={selected.backend}"
    )
    print(
        f"_FI_ALLREDUCE_MAX_SIZE_MB.setdefault({sm}, {{}})[{dist.get_world_size()}] = {selected.candidate_mb:g}"
    )
    print("# Suggested full row:")
    print(f"{sm}: {current_row},")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark SGLang device communicators and tune FlashInfer thresholds.",
    )

    parser.add_argument(
        "--sequence-lengths",
        type=int,
        nargs="+",
        default=DEFAULT_SEQUENCE_LENGTHS,
        help="Sequence lengths to benchmark (tensor shape: [seq_len, hidden_size]).",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=8192,
        help="Hidden size (second tensor dimension).",
    )
    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp16"],
        default="bf16",
        help="Tensor dtype for benchmark tensors.",
    )
    parser.add_argument("--num-warmup", type=int, default=5)
    parser.add_argument("--num-trials", type=int, default=50)
    parser.add_argument(
        "--ops-per-trial",
        type=int,
        default=16,
        help="Number of all-reduce ops executed inside each timing window.",
    )
    parser.add_argument(
        "--flashinfer-backends",
        nargs="+",
        choices=["auto", "trtllm", "mnnvl"],
        default=["auto", "trtllm", "mnnvl"],
        help="FlashInfer backends to benchmark.",
    )
    parser.add_argument(
        "--flashinfer-benchmark-max-workspace-mb",
        type=float,
        default=None,
        help=(
            "Override FlashInfer workspace ceiling during benchmarking. "
            "If unset, the runtime threshold map is used."
        ),
    )
    parser.add_argument(
        "--threshold-candidates-mb",
        type=float,
        nargs="+",
        default=DEFAULT_THRESHOLD_CANDIDATES_MB,
        help="Candidate FlashInfer threshold values (MB) to evaluate.",
    )
    parser.add_argument(
        "--threshold-baseline",
        type=str,
        default="best_non_flashinfer",
        help=(
            "Baseline communicator for threshold selection. "
            "Use 'best_non_flashinfer' or a communicator name from the result table."
        ),
    )
    parser.add_argument(
        "--threshold-max-regression",
        type=float,
        default=1.02,
        help=(
            "Hard regression cap for recommendations (chosen_latency / baseline_latency). "
            "If no candidate satisfies this, the best unconstrained candidate is used."
        ),
    )
    parser.add_argument(
        "--threshold-backend-for-map",
        choices=["auto", "trtllm", "mnnvl"],
        default="auto",
        help="Backend whose recommendation should be used for the printed map snippet.",
    )
    parser.add_argument(
        "--cpu-backend",
        choices=["gloo"],
        default="gloo",
        help="Torch distributed backend for the control process group.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional JSON file for benchmark timings and recommendations.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()
    args.sequence_lengths = sorted(set(args.sequence_lengths))
    args.threshold_candidates_mb = sorted(set(args.threshold_candidates_mb))
    return args


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    if not dist.is_initialized():
        dist.init_process_group(backend=args.cpu_backend, init_method="env://")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size <= 1:
        raise RuntimeError("Run this benchmark with torchrun and world_size > 1.")

    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    cpu_group = dist.group.WORLD
    device_group: Optional[ProcessGroup] = None
    try:
        device_group = dist.new_group(backend="nccl")
    except Exception as exc:
        if rank == 0:
            logger.warning(
                "Failed to create NCCL device group; torch_nccl baseline will be skipped: %s",
                exc,
            )

    benchmark = CommunicatorBenchmark(
        args=args,
        rank=rank,
        world_size=world_size,
        device=device,
        cpu_group=cpu_group,
        device_group=device_group,
    )

    results: dict[int, dict[str, float]] = {}
    recommendations: list[ThresholdRecommendation] = []
    dtype = _dtype_from_arg(args.dtype)

    try:
        results = benchmark.benchmark_all()
        recommendations = _recommend_thresholds(args=args, results=results, dtype=dtype)

        if rank == 0:
            _print_results_table(results=results, args=args, dtype=dtype)
            _print_recommendations(args=args, recommendations=recommendations)

            if args.output_json:
                output = {
                    "world_size": world_size,
                    "dtype": str(dtype),
                    "hidden_size": args.hidden_size,
                    "sequence_lengths": args.sequence_lengths,
                    "num_warmup": args.num_warmup,
                    "num_trials": args.num_trials,
                    "ops_per_trial": args.ops_per_trial,
                    "flashinfer_backends": args.flashinfer_backends,
                    "threshold_candidates_mb": args.threshold_candidates_mb,
                    "threshold_baseline": args.threshold_baseline,
                    "threshold_max_regression": args.threshold_max_regression,
                    "results": results,
                    "recommendations": [
                        {
                            "backend": r.backend,
                            "candidate_mb": r.candidate_mb,
                            "mean_latency_ms": r.mean_latency_ms,
                            "baseline_mean_latency_ms": r.baseline_mean_latency_ms,
                            "overall_speedup": r.overall_speedup,
                            "worst_regression": r.worst_regression,
                            "flashinfer_usage_ratio": r.flashinfer_usage_ratio,
                            "constrained": r.constrained,
                        }
                        for r in recommendations
                    ],
                }
                with open(args.output_json, "w", encoding="utf-8") as fout:
                    json.dump(output, fout, indent=2)
                print(f"\nSaved JSON report to: {args.output_json}")
    finally:
        benchmark.close()

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
