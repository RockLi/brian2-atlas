"""Bounded native errors survive lost MPI stderr without weakening rollback."""
import copy,json,os,subprocess,sys
import errno,shutil,shlex
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lif_training_plan
from test_native_training import RUNNER
from test_training_poisson_zero_vjp import mpi
from test_training_static_timed_inputs import poisson_model
from test_training_poisson_static import fixture


@pytest.mark.parametrize('limit',[None,True,False,0,-1,float('inf'),float('nan'),'300'])
def test_invalid_request_timeout_fails_before_runner_lookup(limit,tmp_path):
    with pytest.raises(ValueError,match='request_timeout'):
        NativeLIFTrainer(lif_training_plan([1,1,2]),runner=tmp_path/'absent',request_timeout=limit)


@pytest.mark.parametrize('limit',[120,300.5])
def test_request_timeout_is_local_transport_setting(limit,tmp_path,monkeypatch):
    t,count=cleanup_runner(tmp_path,False)
    t=NativeLIFTrainer(t.plan,weights=t.state['weights'],runner=t.runner,request_timeout=limit)
    original=subprocess.Popen;limits=[]
    class Observed(original):
        def communicate(self,*args,**kwargs):
            limits.append(kwargs.get('timeout'));return super().communicate(*args,**kwargs)
    monkeypatch.setattr(subprocess,'Popen',Observed)
    assert t._run_request({})=={'schema':'b2-lif-training-result-v1'}
    assert limits==[limit] and count.read_text()=='1'
    path=tmp_path/'checkpoint.json';t.store(path)
    assert 'request_timeout' not in path.read_text()
    restored=NativeLIFTrainer(t.plan,weights=t.state['weights'],runner=t.runner)
    restored.restore(path);assert restored.request_timeout==120


def test_actual_timeout_kills_process_group_without_retry_or_state_commit(tmp_path,monkeypatch):
    import time
    count=tmp_path/'invocations';survivor=tmp_path/'surviving-peer';ready=tmp_path/'peer-ready'
    runner=tmp_path/'slow-runner'
    # Use the OS shell so the deadline measures the sleeping native transport,
    # rather than an unrelated Python interpreter cold-start under load.
    runner.write_text('#!/bin/sh\nprintf 1 >> '+shlex.quote(str(count))+'\n'
        '(printf ready > '+shlex.quote(str(ready))+'; sleep 1; printf alive > '
        +shlex.quote(str(survivor))+'; sleep 60) &\nwait\n')
    runner.chmod(0o755)
    # An external filesystem may take longer than .3s to start even /bin/sh.
    # Synchronize this fixture's actual peer startup before measuring the real
    # subprocess deadline; production transport still times out startup too.
    original=subprocess.Popen;intervals=[]
    class ReadyPeer(original):
        def communicate(self,*args,**kwargs):
            if not intervals:
                deadline=time.monotonic()+15
                while not ready.exists() and time.monotonic()<deadline:time.sleep(.01)
                assert ready.exists(), 'timeout fixture peer did not start'
            started=time.monotonic()
            try:return super().communicate(*args,**kwargs)
            finally:intervals.append(time.monotonic()-started)
    monkeypatch.setattr(subprocess,'Popen',ReadyPeer)
    t=NativeLIFTrainer(lif_training_plan([1,1,2]),weights=[[.3],[.1,.2]],runner=runner,request_timeout=.3)
    before=copy.deepcopy(t.__dict__)
    with pytest.raises(subprocess.TimeoutExpired):t.step(np.zeros((1,1,1)),[0])
    assert len(intervals)==2 and sum(intervals)<3 and t.__dict__==before
    time.sleep(1.1)
    assert count.read_text()=='1' and not survivor.exists()

@pytest.mark.parametrize('kind',['unicode','malformed','oversize','schema','type','empty','extra','list','success_schema','message_oversize','nested','empty_file'])
def test_bounded_error_envelope_and_legacy_fallback(kind,tmp_path):
    message='非有限方程：λ\nquoted "cause"';payload={'schema':'b2-native-training-error-v1','message':message}
    raw=json.dumps(payload)
    if kind=='malformed':raw='{broken'
    elif kind=='oversize':raw='x'*65537
    elif kind=='schema':payload['schema']='unknown';raw=json.dumps(payload)
    elif kind=='type':payload['message']=17;raw=json.dumps(payload)
    elif kind=='empty':payload['message']='';raw=json.dumps(payload)
    elif kind=='extra':payload['other']='ignored';raw=json.dumps(payload)
    elif kind=='list':raw=json.dumps([payload])
    elif kind=='success_schema':payload['schema']='b2-lif-training-result-v1';raw=json.dumps(payload)
    elif kind=='message_oversize':payload['message']='x'*4097;raw=json.dumps(payload)
    elif kind=='nested':payload['message']={'cause':message};raw=json.dumps(payload)
    elif kind=='empty_file':raw=''
    runner=tmp_path/'fake-runner';runner.write_text('#!'+sys.executable+'\nimport os,sys\nfrom pathlib import Path\nassert os.environ.get("B2_TRAIN_ERROR_RESULT")=="1"\nPath(sys.argv[2]).write_text('+repr(raw)+')\nsys.stderr.write("fallback stderr")\nsys.exit(9)\n');runner.chmod(0o755)
    p=lif_training_plan([1,1,2]);t=NativeLIFTrainer(p,weights=[[.3],[.1,.2]],runner=runner);before=copy.deepcopy(t.state)
    with pytest.raises(ValueError) as error:t.step(np.zeros((1,1,1)),[0])
    assert str(error.value)==(message if kind=='unicode' else 'fallback stderr')
    assert t.state==before and t.neuron_state is None and t.clock_tick==0 and t.next_noise_sequence==0

@pytest.mark.parametrize('ranks',[None,2,8])
@pytest.mark.parametrize('repetitions',[1,4])
def test_actual_owner_cause_survives_discarded_stderr_and_rolls_back(ranks,repetitions,monkeypatch):
    mpi(ranks);p,w=poisson_model(ranks,zero=True,invalid=True);x,y,initial=fixture(1)
    t=NativeLIFTrainer(p,weights=w,runner=RUNNER);before=copy.deepcopy(t.state);payloads=[]
    original=subprocess.Popen
    class LostStderr(original):
        def communicate(self,*args,**kwargs):
            out,err=super().communicate(*args,**kwargs)
            path=Path(self.args[-1]);payloads.append(json.loads(path.read_bytes()))
            return out,''
    monkeypatch.setattr(subprocess,'Popen',LostStderr)
    for _ in range(repetitions):
        with pytest.raises(ValueError,match='nonfinite equation'):t.step(x[:,:1],y,initial=initial)
        assert t.state==before and t.neuron_state is None and t.poisson_state is None and t.clock_tick==0 and t.next_noise_sequence==0
    assert len(payloads)==repetitions
    assert all(d['schema']=='b2-native-training-error-v1' and 'nonfinite equation' in d['message'] for d in payloads)


def test_native_cli_default_keeps_failure_output_absent(tmp_path):
    request=tmp_path/'request.json';output=tmp_path/'result.json';request.write_text('{}')
    env={k:v for k,v in os.environ.items() if k not in ('B2_TRAIN_ERROR_RESULT','B2_TRAIN_MPI_LIB')}
    result=subprocess.run([str(RUNNER),str(request),str(output)],capture_output=True,text=True,env=env,timeout=30)
    assert result.returncode!=0 and result.stderr and not output.exists()


def test_native_error_utf8_truncation_is_bounded_and_atomic(tmp_path):
    request=tmp_path/'request.json';output=tmp_path/'result.json';request.write_text(json.dumps({'神经'*4000:1}))
    env={k:v for k,v in os.environ.items() if k!='B2_TRAIN_MPI_LIB'};env['B2_TRAIN_ERROR_RESULT']='1'
    result=subprocess.run([str(RUNNER),str(request),str(output)],capture_output=True,text=True,env=env,timeout=30)
    payload=json.loads(output.read_bytes());assert result.returncode!=0 and payload['schema']=='b2-native-training-error-v1'
    assert payload['message'].endswith(' [truncated]') and len(payload['message'].encode('utf-8'))<=4096
    assert not list(tmp_path.glob('result.error-*.tmp'))


def cleanup_runner(tmp_path, failure):
    payload={'schema':'b2-native-training-error-v1','message':'primary native error'} if failure else {'schema':'b2-lif-training-result-v1'}
    count=tmp_path/'invocations';runner=tmp_path/'cleanup-runner'
    runner.write_text('#!'+sys.executable+'\nimport sys\nfrom pathlib import Path\n'
        'p=Path('+repr(str(count))+');p.write_text(str(int(p.read_text())+1) if p.exists() else "1")\n'
        'Path(sys.argv[2]).write_text('+repr(json.dumps(payload))+')\nsys.exit('+str(9 if failure else 0)+')\n')
    runner.chmod(0o755)
    t=NativeLIFTrainer(lif_training_plan([1,1,2]),weights=[[.3],[.1,.2]],runner=runner)
    return t,count


@pytest.mark.parametrize('failure',[False,True])
@pytest.mark.parametrize('late_writes',[1,3])
def test_late_peer_write_cleanup_preserves_result_without_native_retry(tmp_path,monkeypatch,failure,late_writes):
    t,count=cleanup_runner(tmp_path,failure);before=copy.deepcopy(t.state)
    original=os.rmdir;injected=[]
    def late_writer(path,*args,**kwargs):
        p=Path(path)
        if p.is_absolute() and p.name.startswith('b2-train-') and len(injected)<late_writes:
            (p/'result.error-7.tmp').write_text('late peer write');injected.append(p)
        return original(path,*args,**kwargs)
    monkeypatch.setattr(os,'rmdir',late_writer)
    if failure:
        with pytest.raises(ValueError,match='^primary native error$'):t._run_request({})
    else:assert t._run_request({})=={'schema':'b2-lif-training-result-v1'}
    assert len(injected)==late_writes and all(not p.exists() for p in injected)
    assert count.read_text()=='1' and t.state==before and t.clock_tick==0


@pytest.mark.parametrize('failure',[False,True])
@pytest.mark.parametrize('error_number',[errno.ENOTEMPTY,errno.EIO])
def test_persistent_cleanup_error_is_bounded_and_preserves_primary_cause(tmp_path,monkeypatch,failure,error_number):
    import brian2_rust.training as transport
    t,count=cleanup_runner(tmp_path,failure);before=copy.deepcopy(t.state);paths=[];pauses=[]
    original=os.rmdir
    def blocked_cleanup(path,*args,**kwargs):
        p=Path(path)
        if p.is_absolute() and p.name.startswith('b2-train-'):
            paths.append(p);raise OSError(error_number,'controlled cleanup failure',str(p))
        return original(path,*args,**kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(os,'rmdir',blocked_cleanup)
        patch.setattr(transport,'time',SimpleNamespace(sleep=pauses.append))
        if failure:
            with pytest.raises(ValueError) as error:t._run_request({})
            assert str(error.value)=='primary native error'
            if hasattr(error.value,'__notes__'):assert any('retained' in n for n in error.value.__notes__)
        else:
            with pytest.raises(OSError) as error:t._run_request({})
            assert error.value.errno==error_number
    assert 1<=len(paths)<=16 and sum(pauses)<=1 and count.read_text()=='1' and t.state==before
    if error_number==errno.EIO:assert not pauses
    for p in set(paths):shutil.rmtree(p)
