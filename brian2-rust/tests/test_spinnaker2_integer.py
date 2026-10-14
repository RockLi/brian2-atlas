"""Independent C integer arithmetic, strict admission and SDK extension tests."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess

import brian2 as b
import numpy as np
import pytest
from brian2_atlas import NativeXyloTrainer, xylo_recurrent_training_plan
from brian2_atlas.neuromorphic import convert_ir, export_native_xylo, freeze_native_xylo
from brian2_rust.spinnaker2_integer_runtime import IntegerKernel, KERNEL_DIR
from brian2_rust.spinnaker2_integer_b2ir import graph_plan
from brian2_rust.xylo_b2ir import _lower_validated
from brian2_rust.xylo_network import build_xylo_network
from brian2_rust.export import lower_network

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT/'examples/neuromorphic/native_xylo_recurrent/fixtures/frozen.json'
TRAIN = os.environ.get('B2_TRAIN_RUNNER')
RUN = os.environ.get('B2_RUNNER')
SDK = os.environ.get('B2_SPINNAKER2_PYTHON')


@pytest.fixture
def kernel():
    if not shutil.which('cc'):
        pytest.skip('C compiler required')
    k = IntegerKernel()
    try:
        yield k
    finally:
        k.close()


@pytest.mark.parametrize('case', ['learned','signed_odd','saturation','random'])
def test_c_kernel_matches_independent_native_rust(kernel, case):
    if not TRAIN:
        pytest.skip('set B2_TRAIN_RUNNER')
    rng = np.random.default_rng(411)
    if case == 'learned':
        f = json.loads(FIXTURE.read_text()); p = f['plan']; ni,nh,no = p['sizes']
        weights = [np.array(w).reshape(shape) for w,shape in zip(f['target_weights'],((ni,nh),(nh,no),(nh,nh)))]
    else:
        ni,nh,no = (16,63,2) if case == 'saturation' else (3,7,2)
        p = xylo_recurrent_training_plan([ni,nh,no],readout_start=3,thresholds=(1,1),weight_scale=1.)
        weights = [rng.integers(-7,8,shape,dtype=np.int32) for shape in ((ni,nh),(nh,no),(nh,nh))]
        if case == 'saturation':
            weights = [np.full(shape,127,dtype=np.int32) for shape in ((ni,nh),(nh,no),(nh,nh))]
            weights[1][:,1] = -128; weights[2][:16,::2] = -128; weights[2][32:,1::2] = -128
        elif case == 'signed_odd':
            weights[0][:] = -3; weights[1][:] = 5; weights[2][:] = -1
    steps = 40 if case == 'learned' else 23
    x = rng.integers(0,2,(3,steps,ni),dtype=np.uint8); x[0] = 1; x[1] = 0
    trainer = NativeXyloTrainer(p,runner=TRAIN,weights=[w.ravel()/p['weight_scale'] for w in weights])
    native = trainer.evaluate(x,[0,1,0])
    thresholds = [p['threshold'][0]]*nh + [p['threshold'][1]]*no
    initial = np.zeros((3,3,nh+no),dtype=np.int32)
    whole, final = kernel.run(weights, thresholds, x, initial)
    assert not np.any(initial)
    for axis, key in enumerate(('currents','membranes','spikes')):
        np.testing.assert_array_equal(whole[:,:,axis], native[key])
    for width in (1,7,13):
        state = initial; parts = []
        for start in range(0,steps,width):
            part,state = kernel.run(weights,thresholds,x[:,start:start+width],state);parts.append(part)
        np.testing.assert_array_equal(np.concatenate(parts,axis=1),whole)
        np.testing.assert_array_equal(state,final)
    if case == 'saturation':
        assert np.max(whole[:,:,2,:nh]) == 31 and np.max(whole[:,:,2,nh:]) == 1


def test_threshold_minimum_decay_and_previous_tick_readout(kernel):
    weights = [np.array([[2]]),np.array([[4]]),np.array([[0]])]
    trace,_ = kernel.run(weights,[1,1],np.array([[[1],[0]]]),np.zeros((1,3,2)))
    # Equality emits; readout sees the previous count, not the new hidden count.
    np.testing.assert_array_equal(trace[0], [[[1,0],[0,0],[1,0]], [[0,2],[0,1],[0,1]]])
    weights[0][:] = -3
    trace,_ = kernel.run(weights,[1,1],np.array([[[1]]]),np.zeros((1,3,2)))
    assert trace[0,0,0,0] == -1 and trace[0,0,1,0] == -1


def test_c_abi_rejection_does_not_mutate_buffers(kernel):
    w = np.array([[2]],np.int32); zero = np.zeros((1,1),np.int32)
    threshold = np.array([1,1],np.int32); x = np.ones((1,1,1),np.uint8)
    state = np.zeros((1,3,2),np.int32); trace = np.full((1,1,3,2),123,np.int32)
    for invalid in ('dimension','weight','threshold','input','state'):
        wi=w.copy(); th=threshold.copy(); inp=x.copy(); st=state.copy()
        ni=17 if invalid=='dimension' else 1
        if invalid=='weight': wi[0,0]=128
        if invalid=='threshold': th[0]=0
        if invalid=='input': inp[0,0,0]=16
        if invalid=='state': st[0,2,1]=2
        before=st.copy()
        assert kernel.function(ni,1,1,1,1,wi,w,zero,th,inp,st,trace) != 0
        np.testing.assert_array_equal(st,before); np.testing.assert_array_equal(trace,123)


def test_c_undefined_behavior_and_bounds_under_sanitizers(tmp_path):
    cc=shutil.which('cc')
    if not cc: pytest.skip('C compiler required')
    harness=tmp_path/'probe.c'
    harness.write_text('''#include "kernel.h"
#include <assert.h>
#include <stdlib.h>
int main(void) {
 int32_t wi[16*64],wo[64*8],wr[64*64],th[72],state[2*3*72]={0};
 uint8_t input[2*128*16];
 int32_t *trace=calloc(2*128*3*72,sizeof(int32_t)); assert(trace);
 for(int i=0;i<16*64;i++)wi[i]=(i%3)?127:-128;
 for(int i=0;i<64*8;i++)wo[i]=(i%2)?127:-128;
 for(int i=0;i<64*64;i++)wr[i]=(i%5)?127:-128;
 for(int i=0;i<72;i++)th[i]=1;
 for(int i=0;i<2*128*16;i++)input[i]=15;
 state[0]=-32768;state[72]=32767;
 assert(atlas_s2_integer_run(16,64,8,2,128,wi,wo,wr,th,input,state,trace)==0);
 for(int t=0;t<2*128;t++)for(int j=0;j<72;j++) {
  assert(trace[t*3*72+j]>=-32768 && trace[t*3*72+j]<=32767);
  assert(trace[t*3*72+72+j]>=-32768 && trace[t*3*72+72+j]<=32767);
  assert(trace[t*3*72+144+j]>=0 && trace[t*3*72+144+j]<=(j<64?31:1));
 }
 free(trace);return 0;
}''')
    binary=tmp_path/'probe'
    subprocess.run([cc,'-std=c99','-O1','-Wall','-Wextra','-Werror','-fsanitize=address,undefined',
        '-fno-sanitize-recover=all','-I',str(KERNEL_DIR),str(harness),str(KERNEL_DIR/'kernel.c'),'-o',str(binary)],
        check=True,capture_output=True)
    subprocess.run([str(binary)],check=True,capture_output=True,timeout=30)


def test_strict_integer_admission_and_target_routing():
    previous=b.prefs.codegen.loop_invariant_optimisations
    try:
        b.prefs.codegen.loop_invariant_optimisations=False
        source=lower_network(build_xylo_network([1,1,1],[[2],[4],[0]],[1,1],np.ones((4,1))),4*b.ms)
        plan=_lower_validated(source)
        assert graph_plan(plan)['execution_kind']=='custom-c-host-kernel'
        for key in ('dash_syn','dash_mem','bias'):
            bad=copy.deepcopy(plan);bad['parameters']['hidden'][key][0]=2
            with pytest.raises(ValueError,match='shifts.*bias'):graph_plan(bad)
        bad=dict(plan,steps=129)
        with pytest.raises(ValueError,match='128'):graph_plan(bad)
    finally:
        b.prefs.codegen.loop_invariant_optimisations=previous;b.start_scope()
    for target in ('akida','lava','specksim','xylosim','dynapse'):
        with pytest.raises(ValueError,match='only supported'):
            convert_ir('missing',target=target,profile='spinnaker2-xylo-integer-v1',runner='missing')


@pytest.mark.parametrize('source_kind', ['recurrent', 'feedforward'])
def test_sdk_graph_execution_count_records_and_hardware_rejection(tmp_path, source_kind):
    if not RUN or not SDK: pytest.skip('set B2_RUNNER and B2_SPINNAKER2_PYTHON')
    f=json.loads(FIXTURE.read_text())
    if source_kind == 'feedforward':
        if not TRAIN: pytest.skip('set B2_TRAIN_RUNNER')
        from brian2_atlas import xylo_training_plan
        trainer=NativeXyloTrainer(xylo_training_plan([3,4,2],thresholds=(2,2),weight_scale=8.),runner=TRAIN,
            weights=[np.arange(12)/16,np.arange(8)/16])
        trainer.step(np.ones((1,4,3)),[0]);f=freeze_native_xylo(trainer)
    bundle=export_native_xylo(f,tmp_path/'export',steps=40,runner=RUN,target='spinnaker2')
    assert bundle['execution_kind']=='custom-c-host-kernel' and not bundle['hardware_mapping_tested']
    assert bundle['native_training_bridge']['frozen_sha256']==f['sha256']
    x=np.random.default_rng(5).integers(0,2,(3,40,3),dtype=np.uint8);x[1]=0;x[2]=x[0]
    np.save(tmp_path/'x.npy',x)
    script='''import importlib.util,json,numpy as np,sys
spec=importlib.util.spec_from_file_location('integer',sys.argv[1]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
b=json.load(open(sys.argv[2]));x=np.load(sys.argv[3]);r=m.run(b['plan'],x,bridge=b['native_training_bridge'],batch_size=2)
for width in (1,7,13):
 c=m.run(b['plan'],x,bridge=b['native_training_bridge'],batch_size=1,chunk_steps=width)
 for key in ('spikes','currents','membranes','counts','logits'):np.testing.assert_array_equal(r[key],c[key])
net=m.build_network(b['plan']);backend=m.IntegerHostBackend()
try:raw=backend.run(net,x[:1])
finally:backend.close()
pops={p.name:p for p in net.populations};h=pops['hidden'];out=pops['output']
nh=h.size;no=out.size
for pop,sl in ((h,slice(0,nh)),(out,slice(nh,nh+no))):
 for k,field in (('I','currents'),('v','membranes'),('q','spikes')):
  np.testing.assert_array_equal(np.stack(list(pop.get_variable(k).values()),axis=1),raw[field][0,:,sl])
 raster=np.zeros((40,pop.size),int)
 for cell,ticks in pop.get_spikes().items():np.add.at(raster,(ticks,cell),1)
 np.testing.assert_array_equal(raster,raw['spikes'][0,:,sl])
model=m.register_model()
for method in ('pe_config','get_compiler_flags','create_vertex','get_vertex_splitter'):
 try:getattr(model(),method)()
 except NotImplementedError:pass
 else:raise AssertionError('firmware hook admitted')
np.testing.assert_array_equal(r['spikes'][0],r['spikes'][2]);assert not np.any(r['spikes'][1])
np.savez(sys.argv[4],**{k:v for k,v in r.items() if isinstance(v,np.ndarray)})
'''
    result=subprocess.run([SDK,'-c',script,str(ROOT/'python/brian2_rust/spinnaker2_integer_runtime.py'),
        str(tmp_path/'export/converted/bundle.json'),str(tmp_path/'x.npy'),str(tmp_path/'sdk.npz')],
        cwd=tmp_path,env={**os.environ,'PYTHONPATH':'','PYTHONNOUSERSITE':'1'},capture_output=True,text=True,timeout=120)
    assert result.returncode==0,result.stderr
    if TRAIN:
        native=NativeXyloTrainer(f['plan'],runner=TRAIN,weights=[np.asarray(w)/f['plan']['weight_scale'] for w in f['target_weights']]).evaluate(x,[0,1,0])
        r=np.load(tmp_path/'sdk.npz')
        for key in ('spikes','currents','membranes','logits'):np.testing.assert_array_equal(r[key],native[key])


def test_counted_input_atlas_ir_and_custom_sdk(tmp_path):
    if not RUN or not SDK: pytest.skip('set B2_RUNNER and B2_SPINNAKER2_PYTHON')
    from brian2_atlas import export_network
    from brian2_atlas.neuromorphic import export_ir
    from brian2_rust.results import load_results
    x=np.array([[15,0],[0,15],[15,15],[1,1],[0,0],[0,0]],dtype=np.uint8)
    weights=[np.array([[127,-128,3],[-128,127,-3]]).ravel(),np.array([[127,-128],[-128,127],[3,5]]).ravel(),
             np.array([[127,-128,1],[-128,127,-1],[2,-3,4]]).ravel()]
    previous=b.prefs.codegen.loop_invariant_optimisations
    try:
        b.prefs.codegen.loop_invariant_optimisations=False
        model=export_network(build_xylo_network([2,3,2],weights,[1,2],x),len(x)*b.ms,tmp_path/'source.json')
    finally:
        b.prefs.codegen.loop_invariant_optimisations=previous;b.start_scope()
    bundle=export_ir(tmp_path/'source.json',tmp_path/'converted',target='spinnaker2',
        profile='spinnaker2-xylo-integer-v1',runner=RUN)
    subprocess.run([RUN,str(tmp_path/'source.json'),str(tmp_path/'atlas')],check=True,capture_output=True)
    script='''import importlib.util,json,numpy as np,sys
s=importlib.util.spec_from_file_location('runtime',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
b=json.load(open(sys.argv[2]));r=m.run_simulator(b,chunk_steps=2)
np.savez(sys.argv[3],**{k:v for k,v in r.items() if isinstance(v,np.ndarray)})
'''
    subprocess.run([SDK,'-c',script,str(ROOT/'python/brian2_rust/neuromorphic_runtime.py'),
        str(tmp_path/'converted/bundle.json'),str(tmp_path/'sdk.npz')],cwd=tmp_path,
        env={**os.environ,'PYTHONPATH':'','PYTHONNOUSERSITE':'1'},check=True,capture_output=True,timeout=120)
    sdk=np.load(tmp_path/'sdk.npz');atlas=load_results(model,tmp_path/'atlas')
    for role in ('hidden','output'):
        pop=atlas['populations'][bundle['plan']['roles'][role]]
        for key in ('I','v','q'):np.testing.assert_array_equal(pop['trace'][key],sdk[role+'_'+key][0])
