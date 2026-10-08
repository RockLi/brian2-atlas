"""Detached binary64 clocks on CPU/GPU, including int32 transport and MPI."""
import copy
import operator
import os
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer
from test_native_training import RUNNER
from test_training_integer_ir import engine, model as integer_model


def words(value):
    return list(struct.unpack('<ii', struct.pack('<d', value)))


def model(engine='cpu', ranks=None, time=.0014, last=.0008, kind='lt', rhs=.0006,
          constant=True, float32=False):
    p,w,x=integer_model(engine=engine,ranks=ranks)
    p['clock']['origin']=time
    d=p['dynamic'];d['initial'][4:8]=words(last)+[0.,0.]
    code=[dict(op='integer_state',index=0),dict(op='integer_state',index=1),
          dict(op='constant',value=rhs),dict(op='elapsed_compare',low=0,high=1,right=2,
          kind=kind,constant=rhs if constant else None)]
    d['program_sets']=[[copy.deepcopy(code)],
                       [[dict(op='time_word',word=k,float32=float32)] for k in (0,1)]]
    d['actions']=[dict(owner=0,reads=[4,5,8],writes=[8],program_set=0,threshold=None,trigger=None),
                  dict(owner=1,reads=[6,7],writes=[6,7],program_set=1,threshold=None,trigger=None)]
    d['actions'] += [dict(owner=j,reads=[j],writes=[],program_set=None,threshold=j,trigger=None) for j in range(4)]
    return p,w,x[:,:1]


@pytest.mark.parametrize('ranks',[None,2,8])
def test_precise_elapsed_boundaries_all_comparisons(engine,ranks):
    if ranks is not None and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    # Include cancellation, negative intervals, signed zero and subnormal lastspike.
    for time,last in [(.0014,.0008),(.0008,.0014),(0.,-0.),(1.,np.nextafter(1.,0.)),(0.,np.nextafter(0.,1.))]:
        delta=time-last
        p,w,x=model(engine,ranks,time,last)
        d=p['dynamic'];expected=[];slots=[]
        for rhs in [np.nextafter(delta,-np.inf),delta,np.nextafter(delta,np.inf)]:
            for kind in ('lt','le','eq','ne','ge','gt'):
                code=copy.deepcopy(d['program_sets'][0][0])
                code[2]['value']=float(rhs);code[3].update(kind=kind,constant=float(rhs))
                slot=len(d['initial']);slots.append(slot);d['initial'].append(0.)
                d['initial_parameters'].append(None);d['detached'].append(True);d['binary_states'].append(slot)
                d['actions'].insert(0,dict(owner=len(slots)%4,reads=[4,5,slot],writes=[slot],program_set=len(d['program_sets']),threshold=None,trigger=None))
                d['program_sets'].append([code]);expected.append(float(getattr(operator,kind)(delta,rhs)))
        out=NativeLIFTrainer(p,runner=RUNNER,weights=w).gradients(x,[0])
        np.testing.assert_array_equal(np.asarray(out['final_state'])[0,slots],expected)
        assert out['final_state'][0][6:8]==words(time)
        np.testing.assert_array_equal(out['initial_state_gradients'][0][4:],0)
        if engine!='cpu':assert out['gpu_dispatches']>0


@pytest.mark.parametrize('float32',[False,True])
def test_clock_words_checkpoint_and_dynamic_rhs(engine,float32,tmp_path):
    p,w,x=model(engine,time=.0014,last=.0008,rhs=float(np.float32(.0006)),constant=False,float32=float32)
    trainer=NativeLIFTrainer(p,runner=RUNNER,weights=w)
    out=trainer.step(x,[0]);value=float(np.float32(.0014)) if float32 else .0014
    assert out['final_state'][0][6:8]==words(value)
    assert out['final_state'][0][8]==float(.0014-.0008<float(np.float32(.0006)))
    path=tmp_path/'clock.json';trainer.store(path)
    restored=NativeLIFTrainer(p,runner=RUNNER,weights=w);restored.restore(path)
    np.testing.assert_array_equal(restored.neuron_state,trainer.neuron_state)
    tail=restored.gradients(x,[0],initial='carry')
    next_time=.0014+.0002;next_time=float(np.float32(next_time)) if float32 else next_time
    assert tail['final_state'][0][6:8]==words(next_time)


@pytest.mark.parametrize('selected',[False,True])
def test_invalid_timestamp_is_lazy_and_failure_atomic(engine,selected):
    p,w,x=model(engine);p['dynamic']['initial'][4:6]=words(float('inf'))
    code=p['dynamic']['program_sets'][0][0]
    code += [dict(op='constant',value=float(selected)),dict(op='constant',value=0.),
             dict(op='select',condition=4,yes=3,no=5)]
    trainer=NativeLIFTrainer(p,runner=RUNNER,weights=w);before=copy.deepcopy(trainer.state)
    if selected:
        with pytest.raises((ValueError,RuntimeError)):trainer.step(x,[0])
        assert trainer.state==before and trainer.neuron_state is None
    else:
        assert trainer.gradients(x,[0])['final_state'][0][8]==0.


@pytest.mark.parametrize('issue',['word','operand','future','constant','clock'])
def test_clock_ir_rejects_invalid_plan(issue):
    p,w,x=model();d=p['dynamic']
    if issue=='word':d['program_sets'][1][0][0]['word']=2
    elif issue=='operand':d['program_sets'][0][0][3]['low']=2
    elif issue=='future':d['program_sets'][0][0][3]['right']=3
    elif issue=='constant':d['program_sets'][0][0][3]['constant']=float('inf')
    elif issue=='clock':p.pop('clock')
    if issue=='constant':
        with pytest.raises(ValueError,match='JSON'):NativeLIFTrainer(p,runner=RUNNER,weights=w)
        return
    trainer=NativeLIFTrainer(p,runner=RUNNER,weights=w);before=copy.deepcopy(trainer.state)
    with pytest.raises(ValueError):trainer.step(x,[0])
    assert trainer.state==before and trainer.neuron_state is None


def test_cuda_clock_translation():
    from brian2_rust.training_cuda import kernel_source
    source=kernel_source()
    for name in ('shr_jam','sub','from_float'):
        assert '__device__ uint64_t atlas_clock_'+name+'(' in source
    assert '__device__ bool atlas_clock_compare(' in source
    assert '__float_as_uint(x)' in source and 'as_type<' not in source
    assert 'atlas_clock_sub(exact_time,last)' in source


def test_clock_rejects_previous_metal_abi(tmp_path,monkeypatch):
    if os.environ.get('B2_TEST_GPU')!='1':pytest.skip('actual Metal setup required')
    from brian2_rust import training_metal
    source=tmp_path/'old.c';source.write_text('int b2_train_metal_v5r6(void){return 0;}\n')
    library=tmp_path/'old.dylib'
    subprocess.run(['clang','-dynamiclib',str(source),'-o',str(library)],check=True,capture_output=True,text=True)
    monkeypatch.setattr(training_metal,'build',lambda directory:library)
    p,w,x=model('metal');trainer=NativeLIFTrainer(p,runner=RUNNER,weights=w)
    with pytest.raises(ValueError,match='ABI symbol missing'):trainer.gradients(x,[0])


def test_clock_arithmetic_against_hardware_binary64(tmp_path):
    compiler=shutil.which('clang++') or shutil.which('g++')
    if compiler is None:pytest.skip('C++ compiler required')
    source=Path(__file__).with_name('precise_clock_arithmetic.cpp')
    executable=tmp_path/'clock'
    subprocess.run([compiler,'-std=c++17','-O2',str(source),'-o',str(executable)],check=True,capture_output=True,text=True)
    result=subprocess.run([str(executable)],check=True,capture_output=True,text=True)
    assert '1000000 finite random' in result.stdout
