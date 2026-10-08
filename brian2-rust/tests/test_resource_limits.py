"""Process budgets remain opt-in and cannot bypass independent validation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
import brian2 as b
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUNNER=Path(os.environ.get('B2_RUNNER',ROOT/'target/release/b2-runner'))
sys.path.insert(0,str(ROOT/'python'))
import brian2_rust
from brian2_rust.export import lower_network
from brian2_rust.resource_limits import (candidate_pair_budget,explicit_synapse_budget,
    initial_value_budget,ir_byte_budget,neuron_budget,population_step_budget,
    timed_array_value_budget)


@pytest.mark.parametrize('name,read,default,maximum',[
    ('B2_MAX_INITIAL_VALUES',initial_value_budget,100000000,8000000000),
    ('B2_MAX_EXPLICIT_SYNAPSES',explicit_synapse_budget,50000000,1000000000),
    ('B2_MAX_CANDIDATE_PAIRS',candidate_pair_budget,500000000,1000000000),
    ('B2_MAX_TIMED_ARRAY_VALUES',timed_array_value_budget,10000000,1000000000),
    ('B2_MAX_IR_BYTES',ir_byte_budget,2048*2**20,64*2**30),
    ('B2_MAX_NEURONS',neuron_budget,1000000,16000000),
    ('B2_MAX_POPULATION_STEPS',population_step_budget,10000000,10000000),
])
def test_explicit_finite_limits(monkeypatch,name,read,default,maximum):
    monkeypatch.delenv(name,raising=False);assert read()==default
    for text in ['','0','-1','+1',' 1','1.5','１',str(maximum+1)]:
        monkeypatch.setenv(name,text)
        with pytest.raises(ValueError):read()
    for value in [1,maximum]:
        monkeypatch.setenv(name,str(value));assert read()==value


def test_frontend_and_independent_validator_enforce_the_same_value_budget(tmp_path,monkeypatch):
    previous=b.get_device();b.set_device('rust_standalone',runner=RUNNER)
    try:
        # Every population fits below 6, but their combined total is 8.
        a=b.NeuronGroup(4,'v:1',name='limit_a');c=b.NeuronGroup(4,'v:1',name='limit_c')
        network=b.Network(a,c)
        monkeypatch.setenv('B2_MAX_NEURONS','7')
        with pytest.raises(NotImplementedError,match='neuron preparation budget'):
            lower_network(network,b.ms)
        monkeypatch.setenv('B2_MAX_NEURONS','8')
        monkeypatch.setenv('B2_MAX_INITIAL_VALUES','6')
        with pytest.raises(NotImplementedError,match='values including topology'):
            lower_network(network,b.ms)
        monkeypatch.setenv('B2_MAX_INITIAL_VALUES','8')
        model=lower_network(network,b.ms)
        path=tmp_path/'model.json';path.write_text(json.dumps(model))
        def check(values,byte_count):
            env=dict(os.environ,B2_MAX_INITIAL_VALUES=str(values),B2_MAX_IR_BYTES=str(byte_count))
            return subprocess.run([str(RUNNER),'--validate',str(path)],env=env,capture_output=True,text=True)
        size=path.stat().st_size
        assert check(8,size).returncode==0
        monkeypatch.setenv('B2_MAX_NEURONS','7')
        assert 'neuron count' in check(8,size).stderr
        monkeypatch.setenv('B2_MAX_NEURONS','8')
        assert 'probe array budget exceeded' in check(7,size).stderr
        assert 'configured byte budget' in check(8,size-1).stderr
        assert check(8,0).returncode!=0
        assert check('+8',size).returncode!=0
    finally:
        b.get_device().reinit();b.set_device(previous)


def test_synaptic_arrays_count_toward_the_shared_budget(tmp_path,monkeypatch):
    previous=b.get_device();b.set_device('rust_standalone',runner=RUNNER)
    try:
        a=b.NeuronGroup(2,'v:1',threshold='v>1',reset='v=0',name='budget_source')
        c=b.NeuronGroup(2,'v:1',name='budget_target')
        syn=b.Synapses(a,c,'w:1 (constant)',on_pre='v_post+=w',name='budget_syn')
        syn.connect(i=[0,1],j=[1,0]);syn.w=[0.25,0.5]
        network=b.Network(a,c,syn)
        monkeypatch.setenv('B2_MAX_INITIAL_VALUES','12')
        model=lower_network(network,b.ms)
        path=tmp_path/'synaptic.json';path.write_text(json.dumps(model))
        def validate(limit):
            return subprocess.run([str(RUNNER),'--validate',str(path)],
                env=dict(os.environ,B2_MAX_INITIAL_VALUES=str(limit)),capture_output=True,text=True)
        assert validate(12).returncode==0
        assert 'probe array budget exceeded' in validate(11).stderr
        monkeypatch.setenv('B2_MAX_INITIAL_VALUES','11')
        with pytest.raises(NotImplementedError,match='values including topology'):
            lower_network(network,b.ms)
    finally:
        b.get_device().reinit();b.set_device(previous)


def test_all_to_all_explicit_capacity_matches_independent_validator(tmp_path,monkeypatch):
    """Four-by-four reproduces the published 4096-by-4096 capacity failure."""
    previous=b.get_device();b.set_device('rust_standalone',runner=RUNNER)
    try:
        a=b.NeuronGroup(4,'v:1',threshold='v>1',reset='v=0',name='capacity_a')
        c=b.NeuronGroup(4,'v:1',name='capacity_c')
        syn=b.Synapses(a,c,'w:1 (constant)',on_pre='v_post+=w',name='capacity_syn')
        monkeypatch.setenv('B2_MAX_CANDIDATE_PAIRS','15')
        with pytest.raises(NotImplementedError,match='candidate connection pairs'):
            syn.connect()
        monkeypatch.setenv('B2_MAX_CANDIDATE_PAIRS','16')
        monkeypatch.setenv('B2_MAX_EXPLICIT_SYNAPSES','15')
        with pytest.raises(NotImplementedError,match='explicit synapses'):
            syn.connect()
        monkeypatch.setenv('B2_MAX_EXPLICIT_SYNAPSES','16')
        syn.connect()
        assert len(syn)==16
        model=lower_network(b.Network(a,c,syn),b.ms)
        assert model['instance']['synapses'][0]['source']==[i for i in range(4) for _ in range(4)]
        assert model['instance']['synapses'][0]['target']==list(range(4))*4
        path=tmp_path/'all_to_all.json';path.write_text(json.dumps(model))
        def validate(limit):
            env=dict(os.environ,B2_MAX_EXPLICIT_SYNAPSES=str(limit))
            return subprocess.run([str(RUNNER),'--validate',str(path)],
                                  env=env,capture_output=True,text=True)
        assert validate(16).returncode==0
        assert validate(15).returncode!=0
    finally:
        b.get_device().reinit();b.set_device(previous)


def test_timed_array_capacity_matches_independent_validator(tmp_path,monkeypatch):
    """A 4×4 table reproduces the external 10000×2048 input cap failure."""
    previous=b.get_device();b.set_device('rust_standalone',runner=RUNNER)
    try:
        group=b.NeuronGroup(4,'v:1',dt=.1*b.ms,name='table_target')
        field=b.TimedArray(np.arange(16,dtype=np.float64).reshape(4,4)/16,
                           dt=.1*b.ms)
        group.run_regularly('v += field(t,i)',when='groups',name='table_update')
        network=b.Network(group)
        monkeypatch.setenv('B2_MAX_TIMED_ARRAY_VALUES','15')
        with pytest.raises(NotImplementedError,match='TimedArray needs 1..15'):
            lower_network(network,.4*b.ms,namespace={'field':field})
        monkeypatch.setenv('B2_MAX_TIMED_ARRAY_VALUES','16')
        model=lower_network(network,.4*b.ms,namespace={'field':field})
        code=next(code for code in model['definition']['populations'][0]['code_objects']
                  if code['kind']=='run_regularly')
        assert '"op": "timed_array"' in json.dumps(code)
        path=tmp_path/'table.json';path.write_text(json.dumps(model))
        def validate(limit):
            return subprocess.run([str(RUNNER),'--validate',str(path)],
                                  env=dict(os.environ,B2_MAX_TIMED_ARRAY_VALUES=str(limit)),
                                  capture_output=True,text=True)
        assert validate(16).returncode==0
        assert 'invalid TimedArray layout' in validate(15).stderr
    finally:
        b.get_device().reinit();b.set_device(previous)


@pytest.mark.parametrize('kind', ['neuron', 'poisson', 'generator'])
def test_mainline_steps_and_explicit_lower_budget_in_both_validators(tmp_path, monkeypatch, kind):
    previous=b.get_device();b.set_device('rust_standalone',runner=RUNNER)
    try:
        clock=b.Clock(dt=.1*b.ms)
        if kind=='neuron':group=b.NeuronGroup(1,'v:1',clock=clock)
        elif kind=='poisson':group=b.PoissonGroup(1,rates=0*b.Hz,clock=clock)
        else:group=b.SpikeGeneratorGroup(1,indices=[0],times=[0]*b.ms,clock=clock)
        network=b.Network(group)
        monkeypatch.delenv('B2_MAX_POPULATION_STEPS',raising=False)
        default_model=lower_network(network,100.5*b.second)
        assert default_model['run']['clocks'][0]['steps']==1005000
        monkeypatch.setenv('B2_MAX_POPULATION_STEPS','1000000')
        with pytest.raises(NotImplementedError,match='population (?:duration )?budget'):
            lower_network(network,100.5*b.second)
        monkeypatch.setenv('B2_MAX_POPULATION_STEPS','1005000')
        model=lower_network(network,100.5*b.second)
        assert model['run']['clocks'][0]['steps']==1005000
        path=tmp_path/'primary.json';path.write_text(json.dumps(model))
        def validate(value):
            env=dict(os.environ)
            if value is None:env.pop('B2_MAX_POPULATION_STEPS',None)
            else:env['B2_MAX_POPULATION_STEPS']=value
            return subprocess.run([str(RUNNER),'--validate',str(path)],
                env=env,capture_output=True,text=True,timeout=30)
        assert validate(None).returncode==0
        for limit in ['1000000','1004999']:
            # Primary-clock validation checks the configured step ceiling first.
            result=validate(limit)
            assert result.returncode!=0
            assert 'positive population dt and valid Clock interval required' in result.stderr
        assert validate('1005000').returncode==0
        assert validate('10000000').returncode==0
        for limit in ['10000001','0','+1005000','１']:
            result=validate(limit)
            assert result.returncode!=0 and 'B2_MAX_POPULATION_STEPS' in result.stderr
    finally:
        b.get_device().reinit();b.set_device(previous)


def test_independent_monitor_clock_uses_its_sample_count_for_budget(tmp_path, monkeypatch):
    previous=b.get_device();b.set_device('rust_standalone',runner=RUNNER)
    try:
        group=b.NeuronGroup(100,'v : 1\ntau : 1',dt=1*b.us,name='fast_population')
        monitor=b.StateMonitor(group,['v','tau'],record=True,dt=1*b.ms,
                               name='slow_monitor')
        monkeypatch.setenv('B2_MAX_POPULATION_STEPS','1000000')
        model=lower_network(b.Network(group,monitor),1*b.second)
        assert model['definition']['populations'][0]['steps']==1000000
        monitor_clock=model['definition']['populations'][0]['state_monitors'][0]['clock']
        assert model['run']['clocks'][monitor_clock]['steps']==1000
        path=tmp_path/'slow-monitor.json';path.write_text(json.dumps(model))
        result=subprocess.run([str(RUNNER),'--validate',str(path)],
            env=dict(os.environ,B2_MAX_POPULATION_STEPS='1000000'),
            capture_output=True,text=True,timeout=30)
        assert result.returncode==0,result.stderr
    finally:
        b.get_device().reinit();b.set_device(previous)


def test_default_execution_retains_frozen_output_bytes(tmp_path, monkeypatch):
    monkeypatch.delenv('B2_MAX_POPULATION_STEPS',raising=False)
    frozen=ROOT/'tests/fixtures/reference-v1'
    subprocess.run([str(ROOT/'target/release/b2-runner'),str(frozen/'model.json'),str(tmp_path/'out')],
                   check=True,capture_output=True,timeout=30)
    for name in ['results.bin','events.bin']:
        assert (tmp_path/'out'/name).read_bytes()==(frozen/'reference'/name).read_bytes()
