"""Long-duration regressions for lightweight official Brian2 examples."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parent
SCANNER = ROOT / "tools/scan_official_examples.py"


def example_environment():
    environment = os.environ.copy()
    python_path = [str(ROOT / "python"), str(REPOSITORY)]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment.update({
        "PYTHONPATH": os.pathsep.join(python_path),
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(Path(tempfile.gettempdir()) / "b2-matplotlib"),
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    })
    return environment


def run_example(tmp_path, relative_path, *, smoke_ms=200_000, max_runs=20):
    example = REPOSITORY / relative_path
    completed = subprocess.run(
        [
            sys.executable,
            str(SCANNER),
            "--child", str(example),
            "--mode", "smoke",
            "--device-directory", str(tmp_path / "device"),
            "--smoke-ms", str(smoke_ms),
            "--max-runs", str(max_runs),
        ],
        cwd=example.parent,
        env=example_environment(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    assert '"status": "passed"' in output
    return output


def rust_aot_environment(tmp_path, directory_name):
    environment = example_environment()
    environment.update({
        "B2_RUNNER": str(ROOT / "target" / "release" / "b2-runner"),
        "BRIAN2_STANDALONE_DEVICE": "rust_standalone",
        "BRIAN2_STANDALONE_MODULE": "brian2_rust",
        "BRIAN2_STANDALONE_ENGINE": "aot",
        "BRIAN2_STANDALONE_DIRECTORY": str(tmp_path / directory_name),
    })
    return environment


def run_script(relative_path, environment, *, timeout=90):
    script = REPOSITORY / relative_path
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=script.parent,
        env=environment,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed


@pytest.mark.parametrize(
    ("relative_path", "run_count"),
    [
        ("examples/non_reliability.py", 1),
        ("examples/network_operation_stop.py", 1),
        ("examples/frompapers/Brette_Guigon_2003.py", 1),
        ("examples/advanced/Ornstein_Uhlenbeck.py", 1),
        ("examples/synapses/licklider.py", 1),
        ("examples/reliability.py", 1),
        ("examples/synapses/nonlinear.py", 1),
        ("examples/adaptive_threshold.py", 1),
        ("examples/standalone/simple_case_build.py", 2),
        ("examples/standalone/simple_case.py", 1),
        ("examples/synapses/synapses.py", 1),
        ("examples/frompapers/Brette_Gerstner_2005.py", 3),
        ("examples/compartmental/infinite_cable.py", 2),
        ("examples/synapses/gapjunctions.py", 1),
        ("examples/advanced/custom_events.py", 1),
        ("examples/compartmental/bipolar_with_inputs2.py", 1),
        ("examples/IF_curve_LIF.py", 1),
        ("examples/frompapers/Brette_2012/Fig4.py", 1),
        ("examples/frompapers/Morris_Lecar_1981.py", 1),
        ("examples/advanced/COBAHH_approximated.py", 1),
        ("examples/compartmental/cylinder.py", 1),
        ("examples/synapses/jeffress.py", 1),
        ("examples/frompapers/Brette_2012/Fig1.py", 1),
        ("examples/frompapers/Brette_2012/Fig3AB.py", 2),
        ("examples/frompapers/Brette_2004.py", 2),
        ("examples/advanced/compare_GSL_to_conventional.py", 20),
        ("examples/phase_locking.py", 1),
        ("examples/compartmental/bipolar_with_inputs.py", 1),
        ("examples/frompapers/Brette_2012/Fig3CF.py", 2),
        ("examples/standalone/standalone_multiplerun.py", 1),
        ("examples/frompapers/Hindmarsh_Rose_1984.py", 1),
        ("examples/frompapers/Jansen_Rit_1995_single_column.py", 1),
        ("examples/compartmental/bipolar_cell.py", 3),
        ("examples/frompapers/Diesmann_et_al_1999.py", 1),
        ("examples/compartmental/rall.py", 1),
        ("examples/synapses/continuous_interaction.py", 1),
        ("examples/IF_curve_Hodgkin_Huxley.py", 1),
        ("examples/frompapers/Izhikevich_2007.py", 2),
        ("examples/frompapers/Touboul_Brette_2008.py", 2),
        ("examples/compartmental/hodgkin_huxley_1952.py", 3),
        ("examples/synapses/homeostatic_stdp_at_inhibitory_synapes.py", 1),
        ("examples/synapses/spike_based_homeostasis.py", 1),
    ],
)
def test_lightweight_official_example_completes_full_duration(
    tmp_path, relative_path, run_count
):
    output = run_example(tmp_path, relative_path)
    assert f'"run_count": {run_count}' in output


def test_float_benchmark_profiling_works_with_numpy_2(tmp_path):
    # Two complete one-second runs are enough to execute the first profiling
    # reduction before the scanner intentionally stops the next iteration.
    output = run_example(
        tmp_path,
        "examples/advanced/float_32_64_benchmark.py",
        smoke_ms=1_000,
        max_runs=2,
    )
    assert '"run_count": 2' in output


def test_gaussian_connectivity_variants_construct_nonempty_topologies():
    script = """
from brian2 import prefs
prefs.codegen.target = 'numpy'
from examples.synapses.efficient_gaussian_connectivity import naive, limited, divided
counts = [naive(100), limited(100), divided(100)]
assert all(count > 0 for count in counts), counts
print(counts)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPOSITORY,
        env=example_environment(),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_graupner_pool_runs_full_duration_with_reduced_population(tmp_path):
    environment = rust_aot_environment(tmp_path, "graupner-device")
    environment.update({
        "BRIAN2_GRAUPNER_POINTS": "3",
        "BRIAN2_GRAUPNER_REPETITIONS": "4",
        "BRIAN2_GRAUPNER_PROCESSES": "2",
    })
    run_script("examples/frompapers/Graupner_Brunel_2012.py", environment)
    artifacts = list(tmp_path.glob("graupner-device-*/rust/summary.json"))
    artifacts += list(tmp_path.glob("graupner-device-*/run-*/rust/summary.json"))
    assert len(artifacts) == 3


def test_maass_pool_replays_network_with_bounded_artifacts(tmp_path):
    environment = rust_aot_environment(tmp_path, "maass-device")
    environment.update({
        "BRIAN2_MAASS_PAIRS": "1",
        "BRIAN2_MAASS_PROCESSES": "2",
    })
    run_script("examples/frompapers/Maass_Natschlaeger_Markram_2002.py", environment)
    roots = list(tmp_path.glob("maass-device-*"))
    retained = [run for root in roots for run in root.glob("run-*")]
    assert roots
    assert len(retained) == len(roots)
    assert sum(int(run.name.removeprefix("run-")) for run in retained) == 8
