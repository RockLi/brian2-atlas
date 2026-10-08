"""CUDA policy, graph lifetime, buffer reset and exact direct/graph conformance."""
import copy

import brian2 as b
import numpy as np
import pytest

from brian2_rust.cuda import CudaExecutor
from brian2_rust.cuda_graphs import select_mode, validate_mode, ResidentDag, launch
from brian2_rust.export import lower_network
from test_cuda import real_cuda
from test_metal_delays import device
from test_gpu_monitors import setup, monitored_network, DT
from test_gpu_typed_storage import synaptic_network


def exact(actual,expected):
    if isinstance(expected,np.ndarray):
        assert actual.dtype==expected.dtype
        assert actual.shape==expected.shape
        assert actual.tobytes()==expected.tobytes()
    elif isinstance(expected,dict):
        assert actual.keys()==expected.keys()
        for key in expected:exact(actual[key],expected[key])
    elif isinstance(expected,(list,tuple)):
        assert len(actual)==len(expected)
        for a,e in zip(actual,expected,strict=True):exact(a,e)
    else:assert actual==expected


def result_exact(actual,expected):
    for key in ('populations','synapses'):exact(actual[key],expected[key])


def test_selection_is_bounded_and_explicit(monkeypatch):
    import brian2_rust.cuda_graphs as policy
    monkeypatch.setattr(policy,'MAX_GRAPH_LAUNCHES',8)
    assert select_mode('auto',8)==('resident','first-replay')
    assert select_mode('auto',8,replays=1)==('graph','bounded-replayed-dag')
    assert select_mode('auto',9)==('resident','capture-limit')
    assert select_mode('auto',1)[0]=='resident'
    assert select_mode('graph',0)==('resident','empty-dag')
    assert select_mode('direct',100)==('direct','explicit')
    with pytest.raises(ValueError,match='8 captured'):select_mode('graph',9)
    for invalid in (None,[],1,'typo'):
        with pytest.raises(ValueError):validate_mode(invalid)


def test_launch_preserves_order_ticks_and_empty_lanes():
    calls=[]
    def kernel(grid,block,args,stream):calls.append((grid,block,args,stream))
    tick=2**53+3
    launch([kernel,kernel],['a','b'],[(0,129,[1,0],True),(1,0,[0],True)],
           [(0,tick),(1,tick),(0,tick+1)],'stream')
    assert len(calls)==2
    assert calls[0][:2]==((2,),(128,))
    assert calls[0][2][:2]==('b','a')
    assert isinstance(calls[0][2][2],np.int64)
    assert calls[1][2][2]==tick+1


def test_capture_failure_ends_capture():
    events=[]
    class Stream:
        def begin_capture(self):events.append('begin')
        def end_capture(self):events.append('end');raise RuntimeError('invalid capture')
    cache=ResidentDag.__new__(ResidentDag);cache.stream=Stream();cache.gpu=[]
    def fail(*args,**kw):raise ValueError('original launch failure')
    with pytest.raises(ValueError,match='original launch'):
        cache.capture([fail],[(0,1,[],False)],[(0,0)])
    assert events==['begin','end']


@pytest.mark.parametrize('engine',['reference','metal'])
def test_device_rejects_cuda_option_for_other_engines(device,tmp_path,engine):
    opts=dict(numeric_mode='float32') if engine=='metal' else {}
    with pytest.raises(NotImplementedError,match="requires engine='cuda'"):
        b.set_device('rust_standalone',engine=engine,directory=tmp_path,cuda_dag_execution='graph',**opts)


@real_cuda
@pytest.mark.parametrize('kind',['monitors','typed-plasticity'])
@pytest.mark.parametrize('route',['scan','sparse'])
def test_graph_modes_reset_mutable_buffers_and_preserve_results(device,tmp_path,kind,route,monkeypatch):
    setup(tmp_path/'ref')
    net=monitored_network(True)[0] if kind=='monitors' else synaptic_network()[0]
    model=lower_network(net,8*DT)
    with CudaExecutor(model,tmp_path/'cuda',numeric_mode='float32',event_delivery=route) as ex:
        direct=ex.run(dag_execution='direct');saved=copy.deepcopy(direct)
        assert direct['cuda_runtime']['dag_execution']['selected']=='direct'
        resident=ex.run(dag_execution='resident');result_exact(resident,direct)
        cache=ex._resident_dag
        pointer_ids=[a.data.ptr for a in cache.gpu]
        graph=ex.run(dag_execution='graph');result_exact(graph,direct)
        assert graph['cuda_runtime']['dag_execution']['graph_build_seconds']>0
        assert graph['cuda_runtime']['dag_execution']['buffer_reused']
        # Poison every writable device allocation; replay must reset queues,
        # RNG/event cursors, typed storage, monitor buffers and fault bits.
        with ex.device,ex.stream:
            for i in cache.writable:cache.gpu[i].fill(99)
        replay=ex.run();result_exact(replay,direct)
        report=replay['cuda_runtime']['dag_execution']
        assert report['graph_reused'] and report['graph_build_seconds']==0
        assert [a.data.ptr for a in cache.gpu]==pointer_ids
        assert report['upload_bytes']<report['resident_buffer_bytes']
        result_exact(direct,saved)  # Prior host results remain independent.
        for pop in graph['populations']:
            for state in pop['states'].values():state.fill(123)
        result_exact(ex.run(dag_execution='resident'),direct)
        result_exact(ex.run(dag_execution='graph'),direct)
        # Evict a live graph before allocating a direct replay.
        result_exact(ex.run(dag_execution='direct'),direct)
        assert ex._resident_dag is None and cache.graph is None and cache.gpu==[]
        result_exact(ex.run(dag_execution='graph'),direct)
        size=ex._resident_dag.bytes
        with pytest.raises(MemoryError):ex.run(max_buffer_bytes=size-1)
        assert ex._resident_dag is None
        result_exact(ex.run(dag_execution='graph'),direct)
        import brian2_rust.cuda_graphs as policy
        monkeypatch.setattr(policy,'MAX_GRAPH_LAUNCHES',1)
        fallback=ex.run();result_exact(fallback,direct)
        assert fallback['cuda_runtime']['dag_execution']['reason']=='capture-limit'
        with pytest.raises(ValueError,match='captured kernel'):ex.run(dag_execution='graph')
        result_exact(ex.run(dag_execution='resident'),direct)
        cache=ex._resident_dag
    assert cache.graph is None and cache.gpu==[]
    with pytest.raises(RuntimeError,match='closed'):ex.run()


@real_cuda
def test_graph_fault_is_rejected_on_each_replay(device,tmp_path):
    setup(tmp_path/'ref');net,_,syn,_=synaptic_network();syn.k=0
    model=lower_network(net,4*DT)
    with CudaExecutor(model,tmp_path/'cuda',numeric_mode='float32',dag_execution='graph') as ex:
        for _ in range(2):
            with pytest.raises(FloatingPointError):ex.run()
        assert ex._resident_dag.replays==2


@real_cuda
@pytest.mark.parametrize('mode',['direct','resident','graph'])
def test_device_passes_execution_policy(device,tmp_path,mode):
    setup(tmp_path/'cuda','cuda',numeric_mode='float32',cuda_dag_execution=mode)
    net,_,_=monitored_network(True);net.run(4*DT)
    import json
    report=json.loads((device.last_run_directory/'rust/summary.json').read_text())
    assert report['cuda_runtime']['dag_execution']['selected']==mode


@real_cuda
def test_auto_defers_capture_until_executor_is_reused(device,tmp_path):
    setup(tmp_path/'ref');net=monitored_network(True)[0]
    model=lower_network(net,4*DT)
    with CudaExecutor(model,tmp_path/'cuda',numeric_mode='float32') as ex:
        first=ex.run();second=ex.run();third=ex.run()
        result_exact(second,first);result_exact(third,first)
        reports=[r['cuda_runtime']['dag_execution'] for r in (first,second,third)]
        assert [r['selected'] for r in reports]==['resident','graph','graph']
        assert [r['graph_reused'] for r in reports]==[False,False,True]
        assert reports[0]['graph_build_seconds']==0
        assert reports[1]['graph_build_seconds']>0
        assert reports[2]['graph_build_seconds']==0


@real_cuda
def test_failed_capture_releases_buffers_and_allows_retry(device,tmp_path):
    setup(tmp_path/'ref');net=monitored_network(True)[0]
    model=lower_network(net,4*DT)
    with CudaExecutor(model,tmp_path/'cuda',numeric_mode='float32',dag_execution='graph') as ex:
        saved=ex.kernels[0]
        def fail(*args,**kwargs):raise ValueError('injected capture failure')
        ex.kernels[0]=fail
        with pytest.raises(ValueError,match='injected capture'):ex.run()
        assert ex._resident_dag is None
        ex.kernels[0]=saved
        actual=ex.run()
        result_exact(actual,ex.run(dag_execution='direct'))
