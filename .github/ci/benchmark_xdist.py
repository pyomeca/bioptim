"""Measure serial versus two-worker pytest runs with identical tests and settings.

Requires pytest-xdist and psutil in the active development environment.
"""

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import psutil

ROOT = Path(__file__).resolve().parents[2]


def outcomes(xml):
    return Counter(
        (
            case.get("classname"),
            case.get("name"),
            (
                "failed"
                if case.find("failure") is not None
                else (
                    "error"
                    if case.find("error") is not None
                    else "skipped" if case.find("skipped") is not None else "passed"
                )
            ),
        )
        for case in ET.parse(xml).getroot().iter("testcase")
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", default="3,4,5,6")
    parser.add_argument("--order", default="serial,parallel")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coverage", action="store_true")
    args = parser.parse_args()
    order = args.order.split(",")
    if set(order) != {"serial", "parallel"}:
        parser.error("order must include serial and parallel")
    shards = [int(x) for x in args.shards.split(",")]
    if not set(shards) <= {3, 4, 5, 6}:
        parser.error("only independent shards 3 through 6 are supported")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    import biorbd_casadi
    import casadi
    import numpy

    report = {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "platform": platform.platform(),
        "python": sys.version,
        "cpu_count": os.cpu_count(),
        "production_tree_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD:bioptim"], cwd=ROOT, text=True
        ).strip(),
        "coverage": args.coverage,
        "order": order,
        "versions": {p: importlib.metadata.version(p) for p in ("pytest", "pytest-xdist", "execnet", "psutil")},
        "numerical_versions": {
            "biorbd": biorbd_casadi.__version__,
            "casadi": casadi.__version__,
            "numpy": numpy.__version__,
        },
        "thread_environment": {
            key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "runs": [],
        "comparisons": [],
    }
    valid = True
    for shard in shards:
        directory = ROOT / "tests" / f"shard{shard}"
        digest = hashlib.sha256()
        for path in sorted(directory.glob("*.py")):
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
        signatures = []
        coverage_sets = []
        for index, variant in enumerate(order, 1):
            label = f"shard{shard}-{index}-{variant}"
            xml = output / f"{label}.xml"
            command = [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "--color=no",
                f"tests/shard{shard}",
                "--durations=20",
                f"--junitxml={xml}",
                "-n",
                "2" if variant == "parallel" else "0",
                "--dist=worksteal",
                "--max-worker-restart=0",
            ]
            env = dict(os.environ)
            if args.coverage:
                env["COVERAGE_FILE"] = str(output / (label + ".coverage"))
                command += [
                    "--cov=bioptim",
                    "--cov-report=term",
                    f"--cov-report=xml:{output / (label + '-coverage.xml')}",
                ]
            print("Starting " + label, flush=True)
            peak_rss = 0
            start = time.perf_counter()
            with (output / f"{label}.log").open("w") as log:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                process = psutil.Process(child.pid)
                while child.poll() is None:
                    try:
                        processes = [process] + process.children(recursive=True)
                        rss = 0
                        for proc in processes:
                            try:
                                rss += proc.memory_info().rss
                            except psutil.NoSuchProcess:
                                pass
                        peak_rss = max(peak_rss, rss)
                    except psutil.NoSuchProcess:
                        pass
                    time.sleep(0.25)
            row = {
                "shard": shard,
                "variant": variant,
                "seconds": time.perf_counter() - start,
                "exit_code": child.returncode,
                "peak_process_tree_rss_bytes": peak_rss,
                "test_sources_sha256": digest.hexdigest(),
                "label": label,
            }
            if args.coverage:
                covered = {
                    (cls.get("filename"), int(line.get("number")))
                    for cls in ET.parse(output / (label + "-coverage.xml")).getroot().iter("class")
                    for line in cls.findall("./lines/line")
                    if int(line.get("hits")) > 0
                }
                row["covered_line_count"] = len(covered)
                row["combined_coverage_file_exists"] = (output / (label + ".coverage")).is_file()
                coverage_sets.append(covered)
            report["runs"].append(row)
            (output / "results.json").write_text(json.dumps(report, indent=2))
            print(json.dumps(row), flush=True)
            print("\n".join((output / f"{label}.log").read_text().splitlines()[-23:]), flush=True)
            signatures.append(outcomes(xml))
        same = all(sig == signatures[0] for sig in signatures)
        shard_runs = [r for r in report["runs"] if r["shard"] == shard]
        baseline = [r["seconds"] for r in shard_runs if r["variant"] == "serial"]
        parallel = [r["seconds"] for r in shard_runs if r["variant"] == "parallel"]
        lost_coverage = []
        if args.coverage:
            serial_covered = coverage_sets[order.index("serial")]
            parallel_covered = coverage_sets[order.index("parallel")]
            lost_coverage = sorted(serial_covered - parallel_covered)
        comparable = same and all(row["exit_code"] == 0 for row in shard_runs)
        valid &= comparable
        report["comparisons"].append(
            {
                "shard": shard,
                "identical_cases_and_outcomes": same,
                "cases": sum(signatures[0].values()),
                "outcomes": {
                    status: sum(n for (_, _, state), n in signatures[0].items() if state == status)
                    for status in ("passed", "skipped", "failed", "error")
                },
                "valid": comparable,
                "lost_covered_lines": [f"{file}:{line}" for file, line in lost_coverage],
                "mean_reduction_percent": 100 * (1 - (sum(parallel) / len(parallel)) / (sum(baseline) / len(baseline))),
            }
        )
        (output / "results.json").write_text(json.dumps(report, indent=2))
    report["valid"] = valid
    (output / "results.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["comparisons"]), flush=True)
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
