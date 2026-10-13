"""Source contract and API separation for the narrow analog-model pilot."""
import copy,os
from pathlib import Path
import brian2 as b
import pytest
from brian2_atlas.neuromorphic import convert_ir,export_ir,run_simulator,TARGETS,PROFILES
from brian2_rust.dynapse_model import build,PARAMETERS
from brian2_rust.dynapse_b2ir import _lower_validated,tables
from brian2_rust.export import lower_network
from brian2_rust.protocol import canonical_bytes,attach_protocol
from brian2_rust.spec import bits

@pytest.fixture
def model():
    device=b.get_device();target=b.prefs.codegen.target;optim=b.prefs.codegen.loop_invariant_optimisations
    try:yield lower_network(build([[0],[1],[0],[0]]),4*b.ms)
    finally:
        b.set_device(device);b.prefs.codegen.target=target;b.prefs.codegen.loop_invariant_optimisations=optim;b.start_scope()


def test_recovers_raster_and_actual_parameters(model):
    model['instance']['populations'][0]['parameters']['weight']=[bits(2)]
    plan=_lower_validated(model)
    assert plan['parameters']['weight']==[2]
    assert plan['parameters']['Io']==[PARAMETERS['Io']]
    assert plan['input_raster']==[[0],[1],[0],[0]]
    assert plan['steps']==4 and not plan['mismatch_enabled']


@pytest.mark.parametrize('change',[
    'weight','dc','layout','initial','raster','clock','steps','continuation','monitor',
    'threshold','equation','schedule','recurrence','reset',
])
def test_rejects_semantic_changes(model,change):
    m=copy.deepcopy(model);p=m['definition']['populations'][0];s=m['instance']['populations'][0]
    if change=='weight':s['parameters']['weight']=[bits(-1)]
    elif change=='dc':s['parameters']['Idc']=[bits(1)]
    elif change=='layout':s['parameters']['Io']=[bits(2.**-39)]
    elif change=='initial':s['initial_state']['imem']=[bits(0)]
    elif change=='raster':next(tables(m['definition']))['values'][0]=bits(.5)
    elif change=='clock':m['definition']['clocks'][0]['dt']=bits(.002)
    elif change=='steps':p['steps']=513
    elif change=='continuation':m['run']['clocks'][0]['start_tick']=1
    elif change=='monitor':p['state_monitors']=[]
    elif change=='threshold':p['code_objects'][1]['vector'][0]['value']['op']='ge'
    elif change=='equation':p['code_objects'][2]['vector'][-1]['value']=dict(op='literal',bits=bits(0))
    elif change=='schedule':p['code_objects'][2]['order']=2
    elif change=='recurrence':m['definition']['synapses']=[{}]
    elif change=='reset':p['code_objects'].append(copy.deepcopy(p['code_objects'][0]))
    with pytest.raises(ValueError,match='DynapSim'):_lower_validated(m)


@pytest.mark.parametrize('profile',[p for p in PROFILES if p!='dynapse-dpi-v1'])
def test_other_profiles_rejected_before_reading_source(profile):
    with pytest.raises(ValueError,match='requires dynapse-dpi-v1'):
        convert_ir('missing',target='dynapse',profile=profile,runner='missing')


@pytest.mark.parametrize('target',[t for t in TARGETS if t!='dynapse'])
def test_profile_only_routes_to_dynapse(target):
    with pytest.raises(ValueError,match='only supported by dynapse'):
        convert_ir('missing',target=target,profile='dynapse-dpi-v1',runner='missing')


def test_export_validates_source_hashes_and_refuses_overwrite(model,tmp_path):
    runner=Path(os.environ.get('B2_RUNNER','missing'))
    if not runner.is_file():pytest.skip('set B2_RUNNER')
    source=tmp_path/'source.json';source.write_bytes(canonical_bytes(model))
    bundle=export_ir(source,tmp_path/'converted',target='dynapse',profile='dynapse-dpi-v1',runner=runner)
    assert bundle['target']=='dynapse' and bundle['atlas_validator_sha256']
    with pytest.raises(FileExistsError):
        export_ir(source,tmp_path/'converted',target='dynapse',profile='dynapse-dpi-v1',runner=runner)
    with pytest.raises(ValueError,match='embedded'):
        run_simulator(bundle,[[1]])
    model['instance']['rng_seed']+=1;source.write_bytes(canonical_bytes(model))
    with pytest.raises(ValueError,match='hash mismatch'):
        convert_ir(source,target='dynapse',profile='dynapse-dpi-v1',runner=runner)
