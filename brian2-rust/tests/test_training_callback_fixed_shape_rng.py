"""Literal sampler sizes in authenticated callbacks retain full draw columns."""
import numpy as np
import pytest
import test_training_neuron_whole_reset_rng as reset
from test_training_integer_ir import engine


def normal_factory(array):
    def f(x,i,mean,spread):
        captured=array
        captured*=.8
        draw=np.random.normal(mean,spread,size=4)
        return x+.1*draw[i]
    return f


def uniform_factory(array):
    def f(x,i,mean,spread):
        captured=array
        captured*=.8
        draw=np.random.uniform(mean,mean+spread,size=4)
        return x+.1*draw[i]
    return f


def poisson_factory(array):
    def f(x,i,mean,spread):
        captured=array
        captured*=.8
        draw=np.random.poisson(mean,size=4)
        return x+.1*draw[i]
    return f


@pytest.mark.parametrize('kind',['normal','uniform','poisson'])
@pytest.mark.parametrize('selection',['empty','partial','full'])
def test_fixed_reset_original_restore(engine,kind,selection,monkeypatch,tmp_path):
    monkeypatch.setattr(reset.base,kind+'_factory',globals()[kind+'_factory'])
    reset.test_whole_reset_original_restore(engine,selection,kind,False,monkeypatch,tmp_path)


@pytest.mark.parametrize('kind',['normal','uniform','poisson'])
@pytest.mark.parametrize('window',[None,2])
def test_fixed_reset_all_vjps(engine,kind,window,monkeypatch):
    monkeypatch.setattr(reset.base,kind+'_factory',globals()[kind+'_factory'])
    reset.test_whole_reset_all_bank_initial_vjps(engine,kind,window,False)


def empty_normal_factory(array):
    def f(x,i,mean,spread):
        captured=array
        captured*=.8
        draw=np.random.normal(mean,spread,size=0)
        return x
    return f


def empty_uniform_factory(array):
    def f(x,i,mean,spread):
        captured=array
        captured*=.8
        draw=np.random.uniform(mean,mean+spread,size=0)
        return x
    return f


def empty_poisson_factory(array):
    def f(x,i,mean,spread):
        captured=array
        captured*=.8
        draw=np.random.poisson(mean,size=0)
        return x
    return f


@pytest.mark.parametrize('kind',['normal','uniform','poisson'])
def test_fixed_zero_actual_observations(engine,kind,monkeypatch,tmp_path):
    monkeypatch.setattr(reset.base,kind+'_factory',globals()['empty_'+kind+'_factory'])
    net,storage,g,dt,bundle=reset.model(engine,kind,'partial')
    bundle.plan['trainable']=[False]*len(bundle.weights)
    trainer=reset.NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=reset.RUNNER)
    calls=[]
    for tick in range(4):
        def sample(*args,**kwargs):
            assert kwargs['size']==0;calls.append(tick)
            return np.array([],dtype=np.int64 if kind=='poisson' else float)
        monkeypatch.setattr(np.random,kind,sample)
        out=trainer.step(np.zeros((1,1,1)),[0],**({'initial':'carry'} if tick else {}))
        net.run(dt,namespace={})
        for obj,names in [(storage,('a',)),(g,('v','k'))]:
            for name in names:
                cells=bundle.provenance['neuron_state_layout'][obj.name][name]
                np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],obj.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        row=next(row for row in bundle.provenance['bindings'] if row['object']==g.name and 'mean' in row['variables'])
        for name in ('mean','spread'):assert out['gradients'][row['bank']][row['variables'].index(name)]==0.
        if kind=='poisson':assert trainer.poisson_state is None or trainer.poisson_state['entries']==[]
        path=tmp_path/str(tick);trainer.store(path);trainer=reset.NativeLIFTrainer(bundle.plan,runner=reset.RUNNER);trainer.restore(path)
    assert calls==list(range(4))


@pytest.mark.parametrize('kind',['normal','poisson'])
def test_fixed_zero_scalar_domain_atomic(engine,kind,monkeypatch):
    import copy
    monkeypatch.setattr(reset.base,kind+'_factory',globals()['empty_'+kind+'_factory'])
    _,_,g,_,bundle=reset.model(engine,kind,'empty')
    name='spread' if kind=='normal' else 'mean'
    row=next(row for row in bundle.provenance['bindings'] if row['object']==g.name and name in row['variables'])
    bundle.weights[row['bank']][row['variables'].index(name)]=-1.
    if kind=='normal':
        with pytest.raises(ValueError):np.random.normal(.043,-1.,size=0)
    else:
        with pytest.raises(ValueError):np.random.poisson(-1.,size=0)
    trainer=reset.NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=reset.RUNNER);before=copy.deepcopy(trainer.state)
    with pytest.raises(ValueError):trainer.step(np.zeros((1,1,1)),[0])
    assert trainer.state==before and trainer.clock_tick==0 and trainer.next_noise_sequence==0


def singleton_normal_factory(array):
    def f(x,i,mean,spread):
        captured=array
        captured*=.8
        draw=np.random.normal(mean,spread,size=1)
        return x+.1*draw
    return f


def singleton_uniform_factory(array):
    def f(x,i,mean,spread):
        captured=array
        captured*=.8
        draw=np.random.uniform(mean,mean+spread,size=1)
        return x+.1*draw
    return f


def singleton_poisson_factory(array):
    def f(x,i,mean,spread):
        captured=array
        captured*=.8
        draw=np.random.poisson(mean,size=1)
        return x+.1*draw
    return f


@pytest.mark.parametrize('kind',['normal','uniform','poisson'])
@pytest.mark.parametrize('window',[None,2])
def test_fixed_singleton_all_vjps(engine,kind,window,monkeypatch):
    monkeypatch.setattr(reset.base,kind+'_factory',globals()['singleton_'+kind+'_factory'])
    original=reset.reference
    monkeypatch.setattr(reset,'reference',lambda *args,**kwargs:original(*args,scalar=True,**kwargs))
    reset.test_whole_reset_all_bank_initial_vjps(engine,kind,window,False)


def randn_factory(array):
    def f(x,i,mean,spread):
        captured=array
        captured*=.8
        draw=mean+spread*np.random.randn(4)
        return x+.1*draw[i]
    return f


def rand_factory(array):
    def f(x,i,mean,spread):
        captured=array
        captured*=.8
        draw=mean+spread*np.random.rand(4)
        return x+.1*draw[i]
    return f


@pytest.mark.parametrize('kind',['normal','uniform'])
@pytest.mark.parametrize('window',[None,2])
def test_fixed_raw_rand_all_vjps(engine,kind,window,monkeypatch):
    monkeypatch.setattr(reset.base,kind+'_factory',randn_factory if kind=='normal' else rand_factory)
    reset.test_whole_reset_all_bank_initial_vjps(engine,kind,window,False)


def test_fixed_zero_poisson_scalar_upper_domain_atomic(engine,monkeypatch):
    import copy
    monkeypatch.setattr(reset.base,'poisson_factory',empty_poisson_factory)
    _,_,g,_,bundle=reset.model(engine,'poisson','empty')
    row=next(row for row in bundle.provenance['bindings'] if row['object']==g.name and 'mean' in row['variables'])
    bundle.weights[row['bank']][row['variables'].index('mean')]=1.e19
    with pytest.raises(ValueError):np.random.poisson(1.e19,size=0)
    trainer=reset.NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=reset.RUNNER);before=copy.deepcopy(trainer.state)
    with pytest.raises(ValueError):trainer.step(np.zeros((1,1,1)),[0])
    assert trainer.state==before and trainer.clock_tick==0 and trainer.next_noise_sequence==0
