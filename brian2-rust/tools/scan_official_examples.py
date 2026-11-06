"""Layered, bounded compatibility scan for Brian's official Python examples.

The parent process performs syntax/dependency checks and starts every example in
an isolated child process.  The child forces ``rust_standalone`` and either:

* validates the first network with ``run(0*ms)``; or
* caps every run to a small duration for a bounded smoke test.

This is a compatibility scanner, not a scientific-result validator.  In
particular, shortened runs can make result-analysis code fail for reasons that
are unrelated to backend support; those failures are reported separately.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import json
import os
from pathlib import Path
import py_compile
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parent
EXAMPLES = REPOSITORY / "examples"
RUNNER = ROOT / "target/release/b2-runner"
RESULT_MARKER = "B2_EXAMPLE_SCAN_RESULT="


def discover_examples():
    return sorted(EXAMPLES.rglob("*.py"))


def relative(path):
    return str(path.relative_to(REPOSITORY))


def syntax_result(path):
    try:
        py_compile.compile(str(path), doraise=True)
    except py_compile.PyCompileError as error:
        return {"status": "failed", "error": str(error)}
    return {"status": "passed"}


def missing_imports(path):
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError):
        return []
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    local_names = {candidate.stem for candidate in path.parent.glob("*.py")}
    missing = []
    for name in sorted(names - local_names):
        if name in sys.stdlib_module_names:
            continue
        try:
            found = importlib.util.find_spec(name)
        except (ImportError, ModuleNotFoundError, ValueError):
            found = None
        if found is None:
            missing.append(name)
    return missing


def classify_failure(error_type, message):
    lowered = message.lower()
    if error_type in {"ModuleNotFoundError", "ImportError"}:
        return "dependency"
    if error_type in {"FileNotFoundError", "URLError", "HTTPError"}:
        return "external_resource"
    if any(token in lowered for token in (
        "no such file", "cannot open", "failed to open video", "download",
        "urlopen", "connection refused", "name or service not known",
    )):
        return "external_resource"
    if error_type in {"CapabilityError", "NotImplementedError"} or any(
        token in lowered for token in (
            "unsupported", "not supported", "capability", "fail closed",
            "cannot export", "rust backend",
        )
    ):
        return "backend_semantics"
    if error_type in {"MemoryError"}:
        return "resource_limit"
    return "script_or_short_run"


def parse_child_result(completed, elapsed):
    combined = completed.stdout + completed.stderr
    marker_lines = [
        line.split(RESULT_MARKER, 1)[1]
        for line in combined.splitlines()
        if RESULT_MARKER in line
    ]
    if marker_lines:
        result = json.loads(marker_lines[-1])
    else:
        result = {
            "status": "failed",
            "category": "probe_protocol",
            "error_type": "MissingResultMarker",
            "error": combined[-4000:],
        }
    result["seconds"] = round(elapsed, 3)
    result["returncode"] = completed.returncode
    return result


def run_child(
    path, mode, timeout, smoke_ms, max_runs, work_root, keep_artifacts=False
):
    directory = work_root / path.relative_to(EXAMPLES).with_suffix("") / mode
    directory.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        str(path),
        "--mode",
        mode,
        "--device-directory",
        str(directory / "device"),
        "--smoke-ms",
        str(smoke_ms),
        "--max-runs",
        str(max_runs),
    ]
    environment = os.environ.copy()
    python_path = [str(ROOT / "python"), str(REPOSITORY), str(path.parent)]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment.update({
        "PYTHONPATH": os.pathsep.join(python_path),
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": environment.get(
            "MPLCONFIGDIR",
            str(Path(tempfile.gettempdir()) / "b2-matplotlib"),
        ),
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    })
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=path.parent,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - started
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        output = (stdout or "") + (stderr or "")
        result = {
            "status": "timeout",
            "category": "timeout",
            "seconds": round(elapsed, 3),
            "error": output[-4000:],
        }
    else:
        completed = subprocess.CompletedProcess(
            command, process.returncode, stdout, stderr)
        result = parse_child_result(completed, time.monotonic() - started)
    if not keep_artifacts:
        shutil.rmtree(directory, ignore_errors=True)
    return result


def write_report(report, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)


def scan(args):
    if not RUNNER.is_file():
        raise SystemExit(f"release runner does not exist: {RUNNER}")
    examples = discover_examples()
    work_root = Path(tempfile.mkdtemp(prefix="b2-official-example-scan-"))
    if args.resume and args.output.is_file():
        report = json.loads(args.output.read_text())
        if report.get("schema") != "brian2-rust-official-example-scan-v1":
            raise SystemExit("cannot resume an incompatible scan report")
    else:
        report = {
            "schema": "brian2-rust-official-example-scan-v1",
            "repository": str(REPOSITORY),
            "runner": str(RUNNER),
            "example_count": len(examples),
            "examples": {},
        }
    report["settings"] = {
        "timeout_seconds": args.timeout,
        "smoke_ms": args.smoke_ms,
        "max_runs": args.max_runs,
        "workers": args.workers,
        "keep_artifacts": args.keep_artifacts,
    }
    report["work_directory"] = str(work_root)
    if args.retry_failures:
        for item in report["examples"].values():
            for layer in ("build", "smoke"):
                if item.get(layer, {}).get("status") in {"failed", "timeout"}:
                    item.pop(layer)

    print(f"syntax/dependency layer: {len(examples)} examples", flush=True)
    for path in examples:
        item = report["examples"].setdefault(relative(path), {})
        item["syntax"] = syntax_result(path)
        item["missing_imports"] = missing_imports(path)

    syntax_failures = sum(
        item["syntax"]["status"] != "passed"
        for item in report["examples"].values()
    )
    print(f"syntax complete: {syntax_failures} failures", flush=True)
    write_report(report, args.output)

    build_candidates = []
    for path in examples:
        item = report["examples"][relative(path)]
        if item["syntax"]["status"] != "passed":
            item["build"] = {"status": "not_run", "category": "syntax"}
        elif "build" not in item:
            build_candidates.append(path)
    print(
        f"build layer: {len(build_candidates)} pending, "
        f"{len(examples) - len(build_candidates)} reused/not runnable",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_child, path, "build", args.timeout, args.smoke_ms,
                args.max_runs, work_root, args.keep_artifacts): path
            for path in build_candidates
        }
        for index, future in enumerate(as_completed(futures), 1):
            path = futures[future]
            name = relative(path)
            result = future.result()
            report["examples"][name]["build"] = result
            write_report(report, args.output)
            print(
                f"build {index:03d}/{len(build_candidates):03d} "
                f"{result['status']:<10} {name}",
                flush=True,
            )

    smoke_candidates = [
        path for path in examples
        if (report["examples"][relative(path)]["build"]["status"] == "passed"
            and "smoke" not in report["examples"][relative(path)])
    ]
    print(f"smoke layer: {len(smoke_candidates)} build-pass examples", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_child, path, "smoke", args.timeout, args.smoke_ms,
                args.max_runs, work_root, args.keep_artifacts): path
            for path in smoke_candidates
        }
        for index, future in enumerate(as_completed(futures), 1):
            path = futures[future]
            name = relative(path)
            result = future.result()
            report["examples"][name]["smoke"] = result
            write_report(report, args.output)
            print(
                f"smoke {index:03d}/{len(smoke_candidates):03d} "
                f"{result['status']:<10} {name}",
                flush=True,
            )

    build_counts = Counter(
        item["build"]["status"] for item in report["examples"].values())
    smoke_counts = Counter(
        item.get("smoke", {}).get("status", "not_run")
        for item in report["examples"].values()
    )
    category_counts = Counter()
    for item in report["examples"].values():
        for layer in ("build", "smoke"):
            result = item.get(layer, {})
            if result.get("status") in {"failed", "timeout"}:
                category_counts[result.get("category", "unknown")] += 1
    report["summary"] = {
        "syntax_failures": syntax_failures,
        "missing_dependency_examples": sum(
            bool(item["missing_imports"])
            for item in report["examples"].values()
        ),
        "build": dict(sorted(build_counts.items())),
        "smoke": dict(sorted(smoke_counts.items())),
        "failure_categories": dict(sorted(category_counts.items())),
    }
    write_report(report, args.output)
    if not args.keep_artifacts:
        shutil.rmtree(work_root, ignore_errors=True)
    print(json.dumps(report["summary"], indent=2, sort_keys=True), flush=True)
    print(f"report: {args.output}", flush=True)


def child_probe(args):
    path = args.child.resolve()
    sys.path.insert(0, str(path.parent))
    sys.argv = [str(path)]

    class ProbeComplete(BaseException):
        pass

    import brian2 as brian
    import brian2_rust  # noqa: F401
    import importlib

    device_module = importlib.import_module("brian2.devices.device")
    from brian2.devices.device import all_devices

    rust_device = all_devices["rust_standalone"]
    rust_device.reinit()
    original_set_device = device_module.set_device

    def select_rust(*_positional, **keywords):
        rust_device.reinit()
        options = {
            "runner": RUNNER,
            "engine": "reference",
            "directory": args.device_directory,
        }
        if "build_on_run" in keywords:
            options["build_on_run"] = keywords["build_on_run"]
        return original_set_device("rust_standalone", **options)

    select_rust()
    brian.set_device = select_rust
    device_module.set_device = select_rust

    original_run = brian.Network.run
    run_count = 0

    def bounded_run(network, duration, *positional, **keywords):
        nonlocal run_count
        run_count += 1
        # The wrapper adds one Python frame between Brian's magic/explicit
        # Network.run caller and namespace resolution.
        keywords["level"] = keywords.get("level", 0) + 1
        if args.mode == "build":
            duration = 0 * brian.ms
        else:
            seconds = max(0.0, float(duration / brian.second))
            duration = min(seconds, args.smoke_ms / 1000.0) * brian.second
        result = original_run(network, duration, *positional, **keywords)
        if args.mode == "build" or run_count >= args.max_runs:
            raise ProbeComplete
        return result

    brian.Network.run = bounded_run
    try:
        import matplotlib.pyplot as pyplot

        pyplot.show = lambda *args, **kwargs: None
    except ImportError:
        pass

    started = time.monotonic()
    try:
        import runpy

        runpy.run_path(str(path), run_name="__main__")
        status = "passed" if run_count else "no_network"
        result = {"status": status, "run_count": run_count}
    except ProbeComplete:
        result = {"status": "passed", "run_count": run_count}
    except BaseException as error:
        message = str(error)
        result = {
            "status": "failed",
            "category": classify_failure(type(error).__name__, message),
            "error_type": type(error).__name__,
            "error": message,
            "traceback": traceback.format_exc()[-4000:],
            "run_count": run_count,
        }
    result["child_seconds"] = round(time.monotonic() - started, 3)
    print(RESULT_MARKER + json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["status"] in {"passed", "no_network"} else 1


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", type=Path,
                        default=Path(tempfile.gettempdir()) / "brian2-official-example-scan.json")
    result.add_argument("--timeout", type=float, default=20.0)
    result.add_argument("--smoke-ms", type=float, default=1.0)
    result.add_argument("--max-runs", type=int, default=20)
    result.add_argument("--workers", type=int, default=4)
    result.add_argument("--resume", action="store_true")
    result.add_argument("--retry-failures", action="store_true")
    result.add_argument("--keep-artifacts", action="store_true")
    result.add_argument("--child", type=Path, default=None,
                        help=argparse.SUPPRESS)
    result.add_argument("--mode", choices=("build", "smoke"), default="build",
                        help=argparse.SUPPRESS)
    result.add_argument("--device-directory", type=Path, default=None,
                        help=argparse.SUPPRESS)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.child is not None:
        if args.device_directory is None:
            raise SystemExit("--device-directory is required in child mode")
        return child_probe(args)
    scan(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
