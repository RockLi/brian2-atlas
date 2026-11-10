"""Large metadata sums must compile without recursive Rust AST overflow."""
from pathlib import Path
import subprocess
import os
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from brian2_rust.codegen_sums import usize_sum


def test_twenty_thousand_counter_terms_compile_and_keep_the_total(tmp_path):
    expression=usize_sum(f'counts[{i}]' for i in range(20000))
    source=tmp_path/'sum.rs'
    source.write_text('fn main() { let counts: Vec<usize> = (0..20000).map(|i| i%7).collect(); let result = '+expression+'; assert_eq!(result, counts.iter().sum::<usize>()); }')
    binary=tmp_path/'sum'
    subprocess.run(['rustc','--edition=2021','-C','opt-level=0',str(source),'-o',str(binary)],check=True,capture_output=True,timeout=float(os.environ.get("B2_TEST_RUSTC_TIMEOUT", "60")))
    subprocess.run([str(binary)],check=True,capture_output=True,timeout=5)
    assert usize_sum([])=='0usize'
    assert usize_sum(['count'])=='count'
