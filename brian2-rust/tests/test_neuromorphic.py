"""Shared conversion contract, native validation, domain and semantic failures."""
import copy
import itertools
import json
import os
from pathlib import Path

import brian2 as b
import numpy as np
import pytest
from brian2_atlas.neuromorphic import convert_ir, export_ir, validate_inputs
from brian2_rust.akida_b2ir import _lower_validated
from brian2_rust.export import lower_network
from brian2_rust.capabilities import CapabilityError
from brian2_rust.protocol import canonical_bytes
from brian2_rust.specksim_dense_b2ir import graph_plan, B2IRSpecksimError


@pytest.fixture
def model():
    b.start_scope();b.set_device('runtime');b.prefs.codegen.target='numpy'
    b.prefs.codegen.loop_invariant_optimisations=False
    group=b.NeuronGroup(4,'a:1 (constant)\nbb:1 (constant)\nh0:1\nh1:1\ny:1',dt=b.ms)
    group.a=[0,0,1,1];group.bb=[0,1,0,1]
    group.run_regularly('h0=clip(floor((a+bb)/2),0,127)\nh1=clip(floor((2*a)/2),0,127)\ny=h0',when='start')
    return lower_network(b.Network(group),b.ms)


@pytest.fixture
def dense(model):
    plan=_lower_validated(model,profile='dense-int8-v1')
    plan.update(source_model_sha256='source',atlas_validator_sha256='runner')
    return plan


def test_count_proof_exhaustive_small_binary_domain(dense):
    plan=graph_plan(dense)
    x=np.array(list(itertools.product([0,1],repeat=2)))
    w=np.asarray(plan['weights'])
    voltage=np.zeros((len(x),len(w)),dtype=int)
    counts=np.zeros_like(voltage)
    for i in range(x.shape[1]):
        voltage+=x[:,i,None]*w[None,:,i]
        spikes=voltage>=plan['threshold']
        counts+=spikes;voltage-=spikes*plan['threshold']
    np.testing.assert_array_equal(counts,x@w.T//plan['threshold'])
    np.testing.assert_array_equal(voltage,x@w.T%plan['threshold'])
    assert plan['output_indices']==[0,1,0]
    assert not plan['physical_hardware']


@pytest.mark.parametrize('change,message',[
    ('signed','nonnegative'),('large','nonnegative'),('bias','offset'),
    ('shift','uniform'),('clip','clipping'),('readout','identity'),('outbias','output bias'),
])
def test_count_rejects_semantic_changes(dense,change,message):
    d=copy.deepcopy(dense)
    if change=='signed':d['weights1'][0][0]=-1
    elif change=='large':d['weights1'][0][0]=3
    elif change=='bias':d['bias1'][0]=0
    elif change=='shift':d['shift1'][0]=7
    elif change=='clip':
        d['inputs']=['x']*256;d['weights1']=[[2,2]]*256
    elif change=='readout':d['weights2'][1][-1]=1
    elif change=='outbias':d['bias2'][-1]=1
    with pytest.raises(B2IRSpecksimError,match=message):graph_plan(d)


@pytest.mark.parametrize('bad',[-1,2,.5,float('nan'),float('inf'),256])
def test_reject_inputs_before_sdk_or_cast(bad):
    bundle=dict(schema='atlas.neuromorphic.simulator-bundle.v1',profile='binary-count-v1',
                input_domain='binary',plan=dict(inputs=['x']))
    with pytest.raises(ValueError,match='domain'):validate_inputs(bundle,[[bad]])


def test_target_and_profile_errors_are_explicit():
    with pytest.raises(ValueError,match='unknown simulator'):convert_ir('missing',target='other',runner='missing')
    with pytest.raises(ValueError,match='biased/signed'):convert_ir('missing',target='specksim',runner='missing')


def test_all_targets_share_validated_ir_and_refuse_stale_protocol(model,tmp_path):
    runner=Path(os.environ.get('B2_RUNNER','missing'))
    if not runner.is_file():pytest.skip('set B2_RUNNER to trusted new-repository native runner')
    path=tmp_path/'source.json';path.write_bytes(canonical_bytes(model))
    results=[export_ir(path,tmp_path/t,target=t,runner=runner,profile='binary-count-v1') for t in ('akida','lava','specksim')]
    assert len({r['source_model_sha256'] for r in results})==1
    assert all(r['atlas_validator_sha256'] and r['input_domain']=='binary' for r in results)
    for target in ('akida','lava','specksim'):
        assert (tmp_path/target/'source.json').read_bytes()==path.read_bytes()
    model['instance']['rng_seed']+=1
    path.write_bytes(canonical_bytes(model))
    with pytest.raises(ValueError,match='hash mismatch'):
        convert_ir(path,target='specksim',runner=runner,profile='binary-count-v1')


def test_nonbinary_source_instance_rejected(model,tmp_path):
    from brian2_rust.protocol import attach_protocol
    from brian2_rust.spec import bits
    runner=Path(os.environ.get('B2_RUNNER','missing'))
    if not runner.is_file():pytest.skip('set B2_RUNNER')
    model['instance']['populations'][0]['parameters']['a'][0]=bits(2)
    attach_protocol(model)
    path=tmp_path/'source.json';path.write_bytes(canonical_bytes(model))
    with pytest.raises(ValueError,match='binary input domain'):
        convert_ir(path,target='akida',runner=runner,profile='binary-count-v1')


@pytest.mark.parametrize('kind,count,envname',[
    ('state',33,'B2_MAX_POPULATION_STATES'),
    ('parameter',129,'B2_MAX_POPULATION_PARAMETERS'),
    ('statement',129,'B2_MAX_STATEMENTS'),
])
def test_opt_in_budgets_preserve_defaults_in_python_and_rust(kind,count,envname,monkeypatch,tmp_path):
    import subprocess
    runner=Path(os.environ.get('B2_RUNNER','missing'))
    if not runner.is_file():pytest.skip('set B2_RUNNER')
    monkeypatch.delenv(envname,raising=False)
    b.start_scope();b.set_device('runtime');b.prefs.codegen.target='numpy'
    b.prefs.codegen.loop_invariant_optimisations=False
    if kind=='state':
        equations='\n'.join(f'v{i}:1' for i in range(count));code='v0=v0+1'
    elif kind=='parameter':
        equations='v:1\n'+'\n'.join(f'p{i}:1 (constant)' for i in range(count))
        def balanced(names):
            if len(names)==1:return names[0]
            mid=len(names)//2
            return '('+balanced(names[:mid])+'+'+balanced(names[mid:])+')'
        code='v='+balanced([f'p{i}' for i in range(count)])
    else:
        equations='v:1';code='\n'.join('v=v+1' for _ in range(count))
    group=b.NeuronGroup(1,equations,dt=b.ms)
    group.run_regularly(code,when='start')
    net=b.Network(group)
    with pytest.raises(CapabilityError,match='at most'):
        lower_network(net,b.ms)
    monkeypatch.setenv(envname,str(count))
    model=lower_network(net,b.ms)
    path=tmp_path/'source.json';path.write_bytes(canonical_bytes(model))
    cmd=[str(runner.resolve()),'--validate',str(path)]
    admitted=subprocess.run(cmd,capture_output=True,text=True)
    assert admitted.returncode==0,admitted.stderr
    env=dict(os.environ);env.pop(envname)
    rejected=subprocess.run(cmd,env=env,capture_output=True,text=True)
    assert rejected.returncode!=0
