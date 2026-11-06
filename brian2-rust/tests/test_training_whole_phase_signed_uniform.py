"""Legacy numpy.random.uniform supports signed and zero interval widths."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer
from test_training_whole_neuron_phase_rng import model
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_training_uniform import uniform
from test_training_stochastic import normal
from test_native_training import RUNNER


@pytest.mark.parametrize('width',[-.13,0.,.13])
@pytest.mark.parametrize('mode',['none','fixed','duration','boolean'])
def test_signed_uniform_threshold_original_restore(engine,width,mode,monkeypatch,tmp_path):
    mpi(2);net,storage,g,dt,bundle=model(engine,'uniform','euler',mode,threshold_parameters=True)
    g.threshold_spread=width
    row=next(row for row in bundle.provenance['bindings'] if row['object']==g.name and 'threshold_spread' in row['variables'])
    bundle.weights[row['bank']][row['variables'].index('threshold_spread')]=width
    # Check the actual source API rather than imposing Generator.uniform's
    # separate contract on the legacy RandomState function used by callbacks.
    legacy=np.random.RandomState(7).uniform(.043,.043+width,size=4)
    assert legacy.shape==(4,) and np.all(legacy>=min(.043,.043+width)) and np.all(legacy<=max(.043,.043+width))
    monitor=b.SpikeMonitor(g);net.add(monitor)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    update=[row for row in bundle.provenance['whole_neuron_update_layout'][g.name]['draws'] if not row.get('named')]
    threshold=bundle.provenance['whole_neuron_threshold_layout'][g.name]['draws']
    assert update and len(threshold)==1 and all(row['stream']!=threshold[0]['stream'] for row in update)
    calls=[]
    for tick in range(4):
        streams=iter([*[row['stream'] for row in update],threshold[0]['stream']])
        def sample(*args,**kwargs):
            stream=next(streams);calls.append((tick,stream));assert kwargs['size']==4
            mean=float(g.threshold_mean[:] if stream==threshold[0]['stream'] else g.mean[:])
            spread=float(g.threshold_spread[:] if stream==threshold[0]['stream'] else g.spread[:])
            np.testing.assert_allclose(args,(mean,mean+spread))
            return mean+spread*np.array([uniform(p['seed'],9,0,1,j,tick,stream) for j in range(4)])
        def standard(n):
            assert n==2;return np.array([normal(p['seed'],9,0,1,j,tick,0) for j in range(2)])
        monkeypatch.setattr(np.random,'uniform',sample);monkeypatch.setattr(np.random,'randn',standard)
        out=trainer.step(np.zeros((1,1,1)),[0],**({'initial':'carry'} if tick else {'noise_sequence':9}))
        net.run(dt,namespace={});assert next(streams,None) is None
        for obj,names in [(storage,('v','a')),(g,('v','k','threshold_k'))]:
            for name in names:
                cells=bundle.provenance['neuron_state_layout'][obj.name][name]
                np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],obj.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        recorded=np.asarray(monitor.i)[np.isclose(np.asarray(monitor.t/b.second),tick*float(dt),rtol=0.,atol=1e-12)]
        np.testing.assert_array_equal(np.asarray(out['spikes'])[0,-1,4:],np.isin(np.arange(2),recorded).astype(float))
        path=tmp_path/str(tick);trainer.store(path);trainer=NativeLIFTrainer(p,runner=RUNNER);trainer.restore(path)
    assert len(calls)==4*(len(update)+1) and (out['gpu_dispatches']>0)==(engine!='cpu')
