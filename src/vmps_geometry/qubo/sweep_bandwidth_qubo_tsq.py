from __future__ import annotations

import argparse
import ast
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Iterable, List, Optional

from .bandwidth_qubo_tsq import (
    CLUSTER_EDGES,
    compute_bandwidth,
    compute_envelope,
    normalize_edges,
    solve_bandwidth_qubo_tsq_fast,
)


@dataclass
class SweepRun:
    implementation: str
    cluster: str
    method: str
    backend: str
    random_starts: int
    tabu_tenure: int
    cutoff: int
    seed: int
    rcm_start: bool
    bandwidth: int
    envelope: float
    energy: float
    feasible_permutation: bool
    elapsed_s: float
    position: dict[int, int]
    ordering: list[int]


def parse_int_list(text: str) -> List[int]:
    values = []
    for part in text.split(","):
        item = part.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        raise ValueError("expected at least one integer")
    return values


def result_key(run: SweepRun) -> tuple[bool, int, float, float]:
    return (
        not run.feasible_permutation,
        run.bandwidth,
        run.envelope,
        run.energy,
    )


def progress_label(
    implementation: str,
    index: int,
    total: int,
    cluster: str,
    method: str,
    backend: str,
    random_starts: int,
    tabu_tenure: int,
    cutoff: int,
    seed: int,
    rcm_start: bool,
) -> str:
    return (
        f"[{implementation} {index}/{total}] "
        f"cluster={cluster} method={method} backend={backend} "
        f"random_starts={random_starts} tabu_tenure={tabu_tenure} "
        f"cutoff={cutoff} seed={seed} rcm_start={rcm_start}"
    )


def parse_bool(text: str) -> bool:
    if text == "True" or text == "true":
        return True
    if text == "False" or text == "false":
        return False
    raise ValueError(f"unexpected boolean value: {text!r}")


def parse_bandwidth_line(text: str) -> tuple[int, float]:
    prefix = "bandwidth: "
    if not text.startswith(prefix):
        raise ValueError(f"unexpected bandwidth line: {text!r}")
    body = text[len(prefix):]
    bandwidth_text, envelope_part = body.split(" (envelope: ", maxsplit=1)
    envelope_text = envelope_part[:-1]
    return int(bandwidth_text), float(envelope_text)


def run_python_sweep(
    cluster: str,
    method: str,
    backend: str,
    random_starts_values: Iterable[int],
    tabu_tenure_values: Iterable[int],
    cutoff_values: Iterable[int],
    seed_values: Iterable[int],
    rcm_start: bool,
) -> List[SweepRun]:
    edges = CLUSTER_EDGES[cluster]
    _, normalized_edges = normalize_edges(edges)
    runs: List[SweepRun] = []
    combinations = list(
        product(
            random_starts_values,
            tabu_tenure_values,
            cutoff_values,
            seed_values,
        )
    )
    total = len(combinations)

    for index, (random_starts, tabu_tenure, cutoff, seed) in enumerate(combinations, start=1):
        print(
            progress_label(
                implementation="python",
                index=index,
                total=total,
                cluster=cluster,
                method=method,
                backend=backend,
                random_starts=random_starts,
                tabu_tenure=tabu_tenure,
                cutoff=cutoff,
                seed=seed,
                rcm_start=rcm_start,
            )
        )
        t0 = time.perf_counter()
        result = solve_bandwidth_qubo_tsq_fast(
            normalized_edges,
            method=method,
            backend=backend,
            max_random_starts=random_starts,
            tabu_tenure=tabu_tenure,
            cutoff=cutoff,
            seed=seed,
            rcm_start=rcm_start,
        )
        elapsed_s = time.perf_counter() - t0
        bandwidth = compute_bandwidth(normalized_edges, result.position)
        envelope = compute_envelope(normalized_edges, result.position)
        print(
            f"  -> bandwidth={bandwidth}, envelope={envelope:.6g}, "
            f"energy={result.energy:.6g}, feasible={result.feasible_permutation}, "
            f"elapsed_s={elapsed_s:.3f}"
        )

        runs.append(
            SweepRun(
                implementation="python",
                cluster=cluster,
                method=method,
                backend=backend,
                random_starts=random_starts,
                tabu_tenure=tabu_tenure,
                cutoff=cutoff,
                seed=seed,
                rcm_start=rcm_start,
                bandwidth=bandwidth,
                envelope=envelope,
                energy=float(result.energy),
                feasible_permutation=result.feasible_permutation,
                elapsed_s=elapsed_s,
                position=dict(result.position),
                ordering=list(result.ordering),
            )
        )

    return runs


def run_cpp_single(
    cpp_binary: Path,
    cluster: str,
    method: str,
    random_starts: int,
    tabu_tenure: int,
    cutoff: int,
    seed: int,
    rcm_start: bool,
    exponential_base: Optional[float],
    exponential_max_distance: Optional[int],
) -> SweepRun:
    cpp_binary = cpp_binary.expanduser().resolve()
    if not cpp_binary.exists():
        raise FileNotFoundError(f"C++ solver binary not found: {cpp_binary}")

    cmd = [
        str(cpp_binary),
        "--cluster",
        cluster,
        "--method",
        method,
        "--random-starts",
        str(random_starts),
        "--tabu-tenure",
        str(tabu_tenure),
        "--cutoff",
        str(cutoff),
        "--seed",
        str(seed),
    ]
    if rcm_start:
        cmd.append("--rcm-start")
    if exponential_base is not None:
        cmd.extend(["--exponential-base", str(exponential_base)])
    if exponential_max_distance is not None:
        cmd.extend(["--exponential-max-distance", str(exponential_max_distance)])

    completed = subprocess.run(
        cmd,
        check=True,
        text=True,
        capture_output=True,
    )
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]

    parsed: dict[str, object] = {}
    for line in lines:
        if line.startswith("method: "):
            parsed["method"] = line.split(": ", maxsplit=1)[1]
        elif line.startswith("old vertex -> assigned position: "):
            parsed["position"] = ast.literal_eval(line.split(": ", maxsplit=1)[1])
        elif line.startswith("bandwidth: "):
            bandwidth, envelope = parse_bandwidth_line(line)
            parsed["bandwidth"] = bandwidth
            parsed["envelope"] = envelope
        elif line.startswith("energy: "):
            parsed["energy"] = float(line.split(": ", maxsplit=1)[1])
        elif line.startswith("valid permutation from solver: "):
            parsed["feasible_permutation"] = parse_bool(line.split(": ", maxsplit=1)[1])
        elif line.startswith("elapsed_s: "):
            parsed["elapsed_s"] = float(line.split(": ", maxsplit=1)[1])

    if "position" not in parsed or "bandwidth" not in parsed:
        raise RuntimeError(f"failed to parse C++ solver output:\n{completed.stdout}")

    position = {int(k): int(v) for k, v in dict(parsed["position"]).items()}
    ordering = [-1] * len(position)
    for vertex, pos in position.items():
        ordering[pos] = vertex

    return SweepRun(
        implementation="cpp",
        cluster=cluster,
        method=method,
        backend="tabu",
        random_starts=random_starts,
        tabu_tenure=tabu_tenure,
        cutoff=cutoff,
        seed=seed,
        rcm_start=rcm_start,
        bandwidth=int(parsed["bandwidth"]),
        envelope=float(parsed["envelope"]),
        energy=float(parsed["energy"]),
        feasible_permutation=bool(parsed["feasible_permutation"]),
        elapsed_s=float(parsed["elapsed_s"]),
        position=position,
        ordering=ordering,
    )


def run_cpp_sweep(
    cpp_binary: Path,
    cluster: str,
    method: str,
    random_starts_values: Iterable[int],
    tabu_tenure_values: Iterable[int],
    cutoff_values: Iterable[int],
    seed_values: Iterable[int],
    rcm_start: bool,
    exponential_base: Optional[float],
    exponential_max_distance: Optional[int],
) -> List[SweepRun]:
    if method == "optimization":
        raise ValueError("the C++ solver currently supports only decision and exponential")

    runs: List[SweepRun] = []
    combinations = list(
        product(
            random_starts_values,
            tabu_tenure_values,
            cutoff_values,
            seed_values,
        )
    )
    total = len(combinations)
    for index, (random_starts, tabu_tenure, cutoff, seed) in enumerate(combinations, start=1):
        print(
            progress_label(
                implementation="cpp",
                index=index,
                total=total,
                cluster=cluster,
                method=method,
                backend="tabu",
                random_starts=random_starts,
                tabu_tenure=tabu_tenure,
                cutoff=cutoff,
                seed=seed,
                rcm_start=rcm_start,
            )
        )
        run = run_cpp_single(
            cpp_binary=cpp_binary,
            cluster=cluster,
            method=method,
            random_starts=random_starts,
            tabu_tenure=tabu_tenure,
            cutoff=cutoff,
            seed=seed,
            rcm_start=rcm_start,
            exponential_base=exponential_base,
            exponential_max_distance=exponential_max_distance,
        )
        print(
            f"  -> bandwidth={run.bandwidth}, envelope={run.envelope:.6g}, "
            f"energy={run.energy:.6g}, feasible={run.feasible_permutation}, "
            f"elapsed_s={run.elapsed_s:.3f}"
        )
        runs.append(run)
    return runs


def save_results(
    output_path: Path,
    cluster: str,
    method: str,
    backend: str,
    implementation: str,
    runs: List[SweepRun],
) -> None:
    best = min(runs, key=result_key)
    payload = {
        "cluster": cluster,
        "method": method,
        "backend": backend,
        "implementation": implementation,
        "num_runs": len(runs),
        "best": asdict(best),
        "runs": [asdict(run) for run in sorted(runs, key=result_key)],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def default_output_path(cluster: str, method: str, backend: str, implementation: str) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return Path("results") / f"bandwidth_qubo_tsq_{cluster}_{method}_{backend}_{implementation}_{stamp}.json"


def format_position_map(position: dict[int, int]) -> str:
    items = sorted((int(vertex), int(pos)) for vertex, pos in position.items())
    return "{" + ", ".join(f"{vertex}: {pos}" for vertex, pos in items) + "}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster", type=str, default="C12")
    parser.add_argument(
        "--method",
        type=str,
        default="decision",
        choices=["decision", "exponential", "optimization"],
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="tabu",
        choices=["tabu", "gurobi"],
    )
    parser.add_argument(
        "--implementation",
        type=str,
        default="both",
        choices=["python", "cpp", "both"],
    )
    parser.add_argument("--random-starts", type=parse_int_list, default="2,4,8,12,24")
    parser.add_argument("--tabu-tenure", type=parse_int_list, default="8,12,16")
    parser.add_argument("--cutoff", type=parse_int_list, default="50,100,150")
    parser.add_argument("--seed", type=parse_int_list, default="123,456")
    parser.add_argument("--rcm-start", action="store_true")
    parser.add_argument("--exponential-base", type=float, default=None)
    parser.add_argument("--exponential-max-distance", type=int, default=None)
    parser.add_argument("--cpp-binary", type=Path, default=Path("bandwidth_qubo_tsq_cpp"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.cluster not in CLUSTER_EDGES:
        available = ", ".join(sorted(CLUSTER_EDGES))
        raise KeyError(f"Unknown cluster {args.cluster!r}. Available clusters: {available}")

    runs: List[SweepRun] = []
    if args.implementation in {"python", "both"}:
        runs.extend(
            run_python_sweep(
                cluster=args.cluster,
                method=args.method,
                backend=args.backend,
                random_starts_values=args.random_starts,
                tabu_tenure_values=args.tabu_tenure,
                cutoff_values=args.cutoff,
                seed_values=args.seed,
                rcm_start=args.rcm_start,
            )
        )

    if args.implementation in {"cpp", "both"}:
        if args.backend != "tabu":
            raise ValueError("the C++ sweep path currently supports only backend='tabu'")
        runs.extend(
            run_cpp_sweep(
                cpp_binary=args.cpp_binary,
                cluster=args.cluster,
                method=args.method,
                random_starts_values=args.random_starts,
                tabu_tenure_values=args.tabu_tenure,
                cutoff_values=args.cutoff,
                seed_values=args.seed,
                rcm_start=args.rcm_start,
                exponential_base=args.exponential_base,
                exponential_max_distance=args.exponential_max_distance,
            )
        )

    output_path = args.output or default_output_path(
        args.cluster,
        args.method,
        args.backend,
        args.implementation,
    )
    save_results(output_path, args.cluster, args.method, args.backend, args.implementation, runs)

    best = min(runs, key=result_key)
    print(f"saved {len(runs)} runs to {output_path}")
    print(
        "best: "
        f"implementation={best.implementation}, "
        f"bandwidth={best.bandwidth}, "
        f"envelope={best.envelope:.6g}, "
        f"energy={best.energy:.6g}, "
        f"feasible={best.feasible_permutation}, "
        f"random_starts={best.random_starts}, "
        f"tabu_tenure={best.tabu_tenure}, "
        f"cutoff={best.cutoff}, "
        f"seed={best.seed}"
    )
    print(f"old vertex -> assigned position: {format_position_map(best.position)}")


if __name__ == "__main__":
    main()
