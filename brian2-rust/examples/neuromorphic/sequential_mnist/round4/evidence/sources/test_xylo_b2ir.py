"""Reject source changes that would alter temporal Xylo arithmetic or timing."""
import copy
import importlib.util
import os
from pathlib import Path
import brian2 as b
import pytest
from brian2_atlas.neuromorphic import export_ir,convert_ir,run_simulator
from brian2_rust.export import lower_network
from brian2_rust.protocol import canonical_bytes
from brian2_rust.spec import bits
from brian2_rust.xylo_b2ir import _lower_validated

EXAMPLE=Path(__file__).resolve().parents[1]/'examples/neuromorphic/xylo_temporal/compare.py'
spec=importlib.util.spec_from_file_location('xylo_example',EXAMPLE)
example=importlib.util.module_from_spec(spec);spec.loader.exec_module(example)


@pytest.fixture
def model():
    device=b.get_device();target=b.prefs.codegen.target;optim=b.prefs.codegen.loop_invariant_optimisations
    try:
        c=example.cases()['recurrent']
        yield lower_network(example.build(c),len(c['raster'])*b.ms)
    finally:
        b.set_device(device);b.prefs.codegen.target=target;b.prefs.codegen.loop_invariant_optimisations=optim;b.start_scope()


def test_recovers_actual_graph_parameters_and_raster(model):
    p=_lower_validated(model);c=example.cases()['recurrent']
    assert p['sizes']==dict(input=2,hidden=3,output=2)
    assert p['input_raster']==c['raster']
    assert p['weights']['input']==c['input_weights']
    assert p['weights']['recurrent']==c['recurrent_weights']
    assert p['weights']['readout']==c['readout_weights']
    assert p['parameters']['hidden']['dash_syn']==c['hidden']['syn']
    assert p['hardware_dimension_limits_passed']
    assert p['hardware_config_validation_required']
    assert not p['physical_hardware']


@pytest.mark.parametrize('change,message',[
    ('adaptive','adaptive integrators'),('schedule','requires start'),('clock','1 ms'),('continuation','fresh clock'),
    ('initial','initial states'),('counts_overflow','input counts'),('counts_fractional','input counts'),
    ('counts_negative','input counts'),('raster_time','aligned'),('input_channels','1..16'),
    ('output_count','1..8'),('hidden_count','1..128'),('fractional_parameter','integer parameters'),
    ('mem_zero_shift','decay divisors'),('non_power_decay','decay divisors'),('threshold_zero','16 bits'),
    ('bias_overflow','16 bits'),('spike_cap','spike caps'),('threshold_ast','threshold AST'),
    ('update_ast','run_regularly AST'),('reset_ast','reset AST'),('delay','delays'),
    ('weight_overflow','8-bit'),('weight_fractional','8-bit'),('multapse','sorted unique'),
    ('connection_order','sorted unique'),('delivery_order','delivery'),('monitor','end-of-step'),
    ('synapse_ast','synapses AST'),('pending','pending'),
])
def test_semantic_admission_failures(model,change,message):
    m=copy.deepcopy(model);d=m['definition'];inst=m['instance'];roles=_lower_validated(model)['roles']
    h=d['populations'][roles['hidden']];inp=d['populations'][roles['input']];out=d['populations'][roles['output']]
    hp=inst['populations'][roles['hidden']]['parameters']
    hc={x['kind']:x for x in h['code_objects']};ic={x['kind']:x for x in inp['code_objects']}
    raster=ic['run_regularly']['vector'][0]['value'];syn=inst['synapses'][0]
    if change=='adaptive':hc['state_update']['adaptive']={'integrator':'rkf45'}
    elif change=='schedule':d['schedule']['base_slots']=['start','groups','thresholds','synapses','resets','end']
    elif change=='clock':d['clocks'][0]['dt']=bits(.002)
    elif change=='continuation':m['run']['start']=bits(.001)
    elif change=='initial':inst['populations'][roles['hidden']]['initial_state']['v'][0]=bits(1)
    elif change=='counts_overflow':raster['values'][0]=bits(16)
    elif change=='counts_fractional':raster['values'][0]=bits(.5)
    elif change=='counts_negative':raster['values'][0]=bits(-1)
    elif change=='raster_time':raster['upsampling']=4
    elif change=='input_channels':inp['count']=17
    elif change=='output_count':out['count']=9
    elif change=='hidden_count':h['count']=129
    elif change=='fractional_parameter':hp['bias'][0]=bits(.5)
    elif change=='mem_zero_shift':hp['mem_div'][0]=bits(1)
    elif change=='non_power_decay':hp['syn_div'][0]=bits(3)
    elif change=='threshold_zero':hp['threshold'][0]=bits(0)
    elif change=='bias_overflow':hp['bias'][0]=bits(32768)
    elif change=='spike_cap':hp['spike_cap']=[bits(2)]*h['count']
    elif change=='threshold_ast':hc['threshold']['vector'][0]['value']['op']='ge'
    elif change=='update_ast':hc['run_regularly']['vector'][0]['value']['op']='sub'
    elif change=='reset_ast':hc['reset']['vector'][0]['value']=dict(op='load',name='q')
    elif change=='delay':syn['pathways'][0]['delay_ticks'][0]=1
    elif change=='weight_overflow':syn['parameters']['w'][0]=bits(128)
    elif change=='weight_fractional':syn['parameters']['w'][0]=bits(.5)
    elif change=='multapse':syn['target'][1]=syn['target'][0]
    elif change=='connection_order':syn['target'][:2]=reversed(syn['target'][:2])
    elif change=='delivery_order':
        nodes=d['schedule']['nodes'];indices=[i for i,n in enumerate(nodes) if n['owner_kind']=='synapse']
        nodes[indices[0]],nodes[indices[1]]=nodes[indices[1]],nodes[indices[0]]
    elif change=='monitor':h['state_monitors'][0]['when']='start'
    elif change=='synapse_ast':d['synapses'][0]['code_objects'][0]['vector'][0]['value']=dict(op='load',name='I_post')
    elif change=='pending':syn['pathways'][0]['pending']=[[1,0]]
    with pytest.raises(ValueError,match=message):_lower_validated(m)


def test_native_validation_identity_and_no_overwrite(model,tmp_path):
    runner=Path(os.environ.get('B2_RUNNER','missing'))
    if not runner.is_file():pytest.skip('set B2_RUNNER')
    path=tmp_path/'source.json';path.write_bytes(canonical_bytes(model))
    bundle=export_ir(path,tmp_path/'target',target='xylosim',profile='xylo-audio2-temporal-v1',runner=runner)
    assert (tmp_path/'target/source.json').read_bytes()==path.read_bytes()
    assert bundle['input_domain']=='embedded integer spike-count raster'
    with pytest.raises(FileExistsError):
        export_ir(path,tmp_path/'target',target='xylosim',profile='xylo-audio2-temporal-v1',runner=runner)
    model['instance']['rng_seed']+=1;path.write_bytes(canonical_bytes(model))
    with pytest.raises(ValueError,match='hash mismatch'):
        convert_ir(path,target='xylosim',profile='xylo-audio2-temporal-v1',runner=runner)


def test_target_profile_and_input_separation():
    with pytest.raises(ValueError,match='only supported'):
        convert_ir('missing',target='lava',profile='xylo-audio2-temporal-v1',runner='missing')
    bundle=dict(schema='atlas.neuromorphic.simulator-bundle.v1',target='xylosim',profile='xylo-audio2-temporal-v1')
    with pytest.raises(ValueError,match='embedded spike-count'):
        run_simulator(bundle,[[1]])


@pytest.mark.parametrize('chunk_steps',[0,-1,1.5,True,'7'])
def test_invalid_chunk_size_rejected_before_sdk_import(chunk_steps):
    from brian2_rust.xylo_runtime import run
    with pytest.raises(ValueError,match='positive integer'):
        run({},chunk_steps=chunk_steps)


def test_coverage_suite_reproducibility_and_boundaries():
    spec=importlib.util.spec_from_file_location('xylo_coverage',EXAMPLE.with_name('coverage_cases.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    networks=module.cases()
    assert networks==module.cases()
    observed=module.parameter_coverage(networks)
    assert observed['synaptic_shifts']==list(range(16))
    assert observed['membrane_shifts']==list(range(1,16))
    assert observed['input_counts']==list(range(16))
    assert observed['weight_extrema']==[-128,127]
    maximum=networks['maximum_profile_dimensions']
    assert len(maximum['raster'][0])==16
    assert len(maximum['hidden']['threshold'])==64
    assert len(maximum['output']['threshold'])==8
    assert len(networks['maximum_duration']['raster'])==10000
    assert len(networks['one_tick']['raster'])==1
