"""Early shared topology retains canonical inputs, identities and bounded storage."""
import json
import struct
import subprocess

import brian2 as b
import pytest

import brian2_rust
from brian2_rust.distributed import write_mpi_project, compile_mpi_project, run_mpi_project
from brian2_rust.export import lower_network
from brian2_rust.mpi_prebuild import prebuild_shared_source
from brian2_rust.distributed import build_distributed_plan
from test_mpi import device as device, real_mpi, RUNNER


def model_and_owners(*, hybrid=False, stochastic=False):
    clock = b.Clock(dt=b.ms)
    a = b.NeuronGroup(5, 'x:1', threshold='timestep(t,dt)%2==0', reset='x=0', clock=clock, name='prebuild_a')
    z = b.NeuronGroup(7, 'x:1', clock=clock, name='prebuild_z')
    tiny = b.NeuronGroup(1, 'x:1', clock=clock, name='prebuild_tiny')
    objects = [a,z,tiny]+[b.StateMonitor(g,'x',record=True) for g in [a,z,tiny]]
    # Two identical recipes at distinct canonical projection positions, with
    # whole-owner work between them. The tiny population has three empty ranks.
    for q,(target,edges,seed) in enumerate([(a,31,17),(z[2:6],53,19),(a,37,23),(z[2:6],53,19),(tiny,11,29)]):
        syn = b.Synapses(a[1:4],target,'w:1 (constant)',
            on_pre='x_post+=w*(0.5+rand())' if stochastic else 'x_post+=w',
            clock=clock,name=f'prebuild_s{q}')
        brian2_rust.connect_fixed_total(syn,edges,seed=seed,
            initializers={'w':brian2_rust.Uniform(-1,1)},
            delay_initializer=brian2_rust.Uniform(0*clock.dt,4*clock.dt))
        objects.append(syn)
    if hybrid:
        syn = b.Synapses(a,z,'w:1 (constant)',on_pre='x_post+=w',clock=clock,name='prebuild_s2a')
        syn.connect(i=[0,4],j=[1,5]);syn.w=[0.5,-0.25];syn.delay=clock.dt;objects.append(syn)
    model = lower_network(b.Network(*objects),9*clock.dt,rng_seed=71)
    owners = tuple(1 if p['name']=='prebuild_a' else None for p in model['definition']['populations'])
    return model,owners


def emit(model, owners, path, mode, *, enabled=True):
    write_mpi_project(model,path,ranks=4,population_owners=owners,
        compact_projections=mode=='projections',compact_populations=mode=='populations',
        prebuild_shared_topology=enabled)


def assert_bytes(a,b):
    for name in ['results.bin','events.bin']:
        assert (a/name).read_bytes()==(b/name).read_bytes()


@real_mpi
@pytest.mark.usefixtures('device')
@pytest.mark.parametrize('mode',['ordinary','projections','populations'])
@pytest.mark.parametrize('variant',['mixed','hybrid','stochastic'])
def test_prebuilt_topology_matches_reference_and_exact_limits(tmp_path,monkeypatch,mode,variant):
    model,owners=model_and_owners(hybrid=variant=='hybrid',stochastic=variant=='stochastic')
    emit(model,owners,tmp_path/'baseline',mode,enabled=False)
    emit(model,owners,tmp_path/'mpi',mode)
    old=json.loads((tmp_path/'baseline/manifest.json').read_text());new=json.loads((tmp_path/'mpi/manifest.json').read_text())
    assert old['plan_sha256']==new['plan_sha256']
    assert new['topology_prebuild']['shared_projections']==3
    for name,digest in old['files'].items():
        if name!='main.rs':assert new['files'][name]==digest
    compile_mpi_project(tmp_path/'mpi',opt_level=1,panic_strategy='abort')
    report=run_mpi_project(tmp_path/'mpi',tmp_path/'result',timeout=30)
    records=report['prebuilt_topology']['rank_records'];rows=[records[i:i+8] for i in range(0,len(records),8)]
    assert len(rows)==4
    for n,limit,metadata,peak,remaining,fixed,seconds,verify in rows:
        assert n==3 and fixed==5 and 0<metadata<=peak<=limit and remaining==0
        assert struct.unpack('d',struct.pack('Q',seconds))[0]>0
        assert struct.unpack('d',struct.pack('Q',verify))[0]>0
    assert any(0 in p['rank_stats'][::2] for p in report['procedural_topology'])
    (tmp_path/'model.json').write_text(json.dumps(model))
    subprocess.run([str(RUNNER),str(tmp_path/'model.json'),str(tmp_path/'reference')],check=True,capture_output=True,timeout=30)
    assert_bytes(tmp_path/'result',tmp_path/'reference')
    # The exact reported retained-byte limit succeeds; one byte less fails on
    # at least one rank. Exact edge limits also prove consumption is not counted
    # as a second construction.
    peak=max(row[3] for row in rows)
    monkeypatch.setenv('B2_MPI_MAX_PREBUILT_TOPOLOGY_BYTES',str(peak))
    edges=max(sum(p['rank_stats'][2*rank] for p in report['procedural_topology']) for rank in range(4))
    monkeypatch.setenv('B2_MPI_MAX_LOCAL_EDGES',str(edges))
    run_mpi_project(tmp_path/'mpi',tmp_path/'exact',timeout=30)
    assert_bytes(tmp_path/'result',tmp_path/'exact')
    monkeypatch.setenv('B2_MPI_MAX_PREBUILT_TOPOLOGY_BYTES',str(peak-1))
    with pytest.raises(RuntimeError,match='cache budget exceeded'):
        run_mpi_project(tmp_path/'mpi',tmp_path/'cache-limit',timeout=20)
    assert not (tmp_path/'cache-limit').exists()
    monkeypatch.setenv('B2_MPI_MAX_PREBUILT_TOPOLOGY_BYTES',str(peak))
    monkeypatch.setenv('B2_MPI_MAX_LOCAL_EDGES',str(edges-1))
    with pytest.raises(RuntimeError,match='edge budget exceeded'):
        run_mpi_project(tmp_path/'mpi',tmp_path/'edge-limit',timeout=20)


@real_mpi
@pytest.mark.usefixtures('device')
def test_prebuilt_corrupt_shard_is_rejected_before_cache_admission(tmp_path,monkeypatch):
    model,owners=model_and_owners();emit(model,owners,tmp_path/'mpi','populations')
    compile_mpi_project(tmp_path/'mpi',opt_level=1,panic_strategy='abort')
    shard=tmp_path/'mpi/instance.rank-2.bin';data=bytearray(shard.read_bytes());data[-1]^=1;shard.write_bytes(data)
    monkeypatch.setenv('B2_MPI_MAX_PREBUILT_TOPOLOGY_BYTES','0')
    command=['mpiexec','-n','4',str(tmp_path/'mpi/b2-mpi'),str(tmp_path/'mpi/instance.bin'),str(tmp_path/'result')]
    result=subprocess.run(command,capture_output=True,text=True,timeout=20)
    assert result.returncode!=0 and 'MPI instance differs from compiled plan' in result.stderr
    assert 'cache budget exceeded' not in result.stderr and not (tmp_path/'result').exists()


@pytest.mark.usefixtures('device')
def test_prebuilt_no_shared_is_source_noop_and_emitter_drift_fails(tmp_path):
    model,owners=model_and_owners();whole=tuple(1 for _ in owners)
    emit(model,whole,tmp_path/'old','populations',enabled=False);emit(model,whole,tmp_path/'new','populations')
    assert (tmp_path/'old/main.rs').read_bytes()==(tmp_path/'new/main.rs').read_bytes()
    emit(model,owners,tmp_path/'mixed','populations',enabled=False)
    source=(tmp_path/'mixed/main.rs').read_text();plan=build_distributed_plan(model,ranks=4,population_owners=owners)
    with pytest.raises(ValueError,match='generated source differs'):
        prebuild_shared_source(model,plan,source.replace('let local_count = usize::try_from(total[sources+mpi.rank])?;', 'let local_count = 0;'))
    with pytest.raises(TypeError,match='must be a boolean'):
        write_mpi_project(model,tmp_path/'bad',prebuild_shared_topology=1)
