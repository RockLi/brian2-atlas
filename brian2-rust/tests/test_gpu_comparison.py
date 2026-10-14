"""Comparison input contracts and unary promotion exposed by recurrent f32 ODEs."""
import sys
from pathlib import Path

import numpy as np
import pytest

from brian2_rust.spec import expression

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples'))
import gpu_recurrent


@pytest.mark.parametrize('neurons,degree',[(1,2),(64,1),(64,65),(65536,32)])
def test_recurrent_rejects_invalid_or_unbounded_topology(neurons,degree):
    with pytest.raises(ValueError):gpu_recurrent.validate_size(neurons,degree)


@pytest.mark.parametrize('dtype,conversion',[('f32','f32_to_f64'),('index','index_to_f64'),('tick','tick_to_f64')])
def test_unary_minus_promotes_semantic_float_operands(dtype,conversion):
    options=dict(symbol_dtypes={'x':dtype},logical_indices={'x'} if dtype=='index' else set(),
                 logical_ticks={'x'} if dtype=='tick' else set())
    assert expression('-x',**options)=={'op':'neg','arg':{'op':conversion,'arg':{'op':'load','name':'x'}}}


def test_unary_minus_preserves_signed_integer_domain():
    assert expression('-x',symbol_dtypes={'x':'i32'})=={'op':'neg','arg':{'op':'load','name':'x'}}
    with pytest.raises(NotImplementedError):expression('-x',symbol_dtypes={'x':'u32'})


def test_recurrent_fixture_has_fixed_ei_indegree_and_active_delayed_feedback():
    v,drive,projections=gpu_recurrent.arrays(64,8)
    assert all(a.dtype==np.float32 for a in (v,drive))
    for (sources,targets,weight),degree in zip(projections,(7,1)):
        np.testing.assert_array_equal(np.bincount(targets,minlength=64),degree)
        assert len(set(zip(sources.tolist(),targets.tolist())))==64*degree
        assert np.all(sources<int(.8*64)) if weight>0 else np.all(sources>=int(.8*64))
    short=gpu_recurrent.oracle(64,20,8)
    np.testing.assert_array_equal(short['synaptic_current'],0)
    result=gpu_recurrent.oracle(64,512,8)
    assert len(result['ticks'])>64 and np.any(result['synaptic_current']!=0)
    assert np.all(result['v']<=1)


def test_hh_oracle_singular_rates_recording_and_refractory():
    import gpu_hh as hh
    assert hh.rates(np.asarray([hh.VT+.013]))['m'][0][0]==1280
    assert hh.rates(np.asarray([hh.VT+.040]))['m'][1][0]==1400
    assert hh.rates(np.asarray([hh.VT+.015]))['n'][0][0]==160
    initial=hh.arrays(32);result=hh.oracle(32,512)
    for name in hh.FIELDS:
        assert np.isfinite(result[name]).all()
        np.testing.assert_array_equal(result['trace_'+name][0],initial[name][:16])
    for name in ('m','n','h'):
        assert np.all((result['trace_'+name]>=0)&(result['trace_'+name]<=1))
    for name,tau in [('ge',.005),('gi',.010)]:
        np.testing.assert_allclose(result[name],initial[name].astype(np.float64)*np.exp(-512*hh.DT/tau),rtol=1e-12,atol=0)
    assert len(result['ticks'])>32
    for neuron in range(32):
        assert np.all(np.diff(result['ticks'][result['indices']==neuron])>=hh.REF_TICKS)


def test_hh_acceptance_does_not_hide_bad_trace_or_spike_shift():
    import gpu_hh as hh
    expected=hh.oracle(4,64);actual={k:v.copy() for k,v in expected.items()}
    assert all(hh.checks(actual,expected,hh.configuration(4,64))[0].values())
    actual['trace_v'][2,0]+=.001
    assert not hh.checks(actual,expected,hh.configuration(4,64))[0]['trace_v']
    assert len(actual['ticks'])
    actual['ticks'][0]+=1
    assert not hh.checks(actual,expected,hh.configuration(4,64))[0]['ticks']


def test_comparison_f32_gate_preserves_spikes_and_excludes_phase_shifted_current():
    from modal_gpu_compare import paired_checks
    expected=dict(v=np.asarray([.5],np.float32),ticks=np.asarray([3],np.int64),
                  indices=np.asarray([0],np.int64),synaptic_current=np.asarray([.1],np.float32))
    actual={k:v.copy() for k,v in expected.items()};actual['synaptic_current']+=1
    result=paired_checks(actual,expected,'recurrent-cuba-v0',1,4)
    assert result['passed'] and not result['bitwise']['synaptic_current']
    actual['ticks'][0]+=1
    assert not paired_checks(actual,expected,'recurrent-cuba-v0',1,4)['passed']
    actual['ticks']=np.asarray([],np.int64)
    assert not paired_checks(actual,expected,'recurrent-cuba-v0',1,4)['passed']


def test_comparison_repeats_numerical_failure_but_never_accepts_crash_or_missing_output():
    from modal_gpu_compare import completed_trial,summarize_trials
    row=dict(backend='cuda',repeat=0,exit_code=1,result_npz=b'checked elsewhere',report=dict(
        status='correctness_failed',checks={'v':False},total_backend_wall_seconds=2.,result_arrays={'v':{'sha256':'a'}}))
    assert completed_trial(row)
    assert not completed_trial({**row,'exit_code':-9})
    assert not completed_trial({k:v for k,v in row.items() if k!='result_npz'})
    assert not completed_trial({**row,'report':{**row['report'],'status':'execution_error'}})
    report=dict(requested_repeats=2,trials=[row,{**row,'repeat':1}],
        pairwise_against_cpu_f32=[dict(backend='cuda',passed=True,repeat=i) for i in range(2)])
    summary=next(r for r in summarize_trials(report) if r['backend']=='cuda')
    assert summary['timing_eligible'] and not summary['unchanged_reference_gate_passed']
    assert summary['matched_cpu_f32_gate_passed'] and summary['median_seconds']==2
    report['trials'][1]={**row,'repeat':1,'report':{**row['report'],'result_arrays':{'v':{'sha256':'b'}}}}
    summary=next(r for r in summarize_trials(report) if r['backend']=='cuda')
    assert not summary['repeat_output_hashes_equal'] and summary['timing_eligible']
    report['pairwise_against_cpu_f32'][1]['passed']=False
    assert not next(r for r in summarize_trials(report) if r['backend']=='cuda')['timing_eligible']
    report['trials'].pop()
    assert not next(r for r in summarize_trials(report) if r['backend']=='cuda')['completed_all_trials']


@pytest.mark.parametrize('backend,case',[('cuda','recurrent-cuba-v0'),('genn','hh-ionic-v0'),('cpu-f32','independent-if-v0')])
def test_genn_arithmetic_policy_cannot_silently_apply_to_another_adapter(tmp_path,backend,case):
    import subprocess
    output=tmp_path/'must-not-exist'
    command=[sys.executable,str(Path(__file__).resolve().parents[1]/'examples/gpu_baseline.py'),
        '--backend',backend,'--case',case,'--genn-recurrent-policy','brian-euler','--output',str(output)]
    result=subprocess.run(command,text=True,capture_output=True)
    assert result.returncode==2
    assert 'GeNN recurrent policy requires' in result.stderr
    assert not output.exists()


def test_modal_genn_policy_rejects_nonrecurrent_case_before_cloud_submission(tmp_path):
    import subprocess
    output=tmp_path/'must-not-exist'
    result=subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'examples/modal_gpu_compare.py'),
        '--case','independent-if-v0','--genn-recurrent-policy','brian-euler','--output',str(output)],text=True,capture_output=True)
    assert result.returncode==2 and 'GeNN recurrent policy requires case=' in result.stderr
    assert not output.exists()
