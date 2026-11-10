"""Run selected frontend regressions against this checkout."""
import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="numpy", choices=("numpy", "cython"))
    args, pytest_args = parser.parse_known_args()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    import brian2
    import pytest
    from brian2.tests import PreferencePlugin

    if not Path(brian2.__file__).resolve().is_relative_to(root):
        raise RuntimeError("Tests must import Brian2 from this checkout")
    brian2.prefs.codegen.target = args.target
    if pytest_args and pytest_args[0] == "--":
        pytest_args = pytest_args[1:]
    if not pytest_args:
        pytest_args = [str(root / "brian2/tests" / name) for name in (
            "test_functions.py", "test_codegen.py", "test_statements.py",
            "test_codestrings.py", "test_units.py", "test_synapses.py", "test_monitor.py",
            "test_atlas_index_codegen.py",
        )]
    return pytest.main([
        "-q", "--timeout=300", "--maxfail=8",
        "-m", "not standalone_only and not cpp_standalone and not long and not gsl",
        *pytest_args,
    ], plugins=[PreferencePlugin(dict(brian2.prefs))])


if __name__ == "__main__":
    raise SystemExit(main())
