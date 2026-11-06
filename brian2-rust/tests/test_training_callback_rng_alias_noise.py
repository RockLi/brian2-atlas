"""Imported RNG callbacks: actual Brian/checkpoints and independent all-state VJP."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
import test_training_callback_rng_alias_functions as aliases
import test_training_callback_uncaptured_noise as uncaptured
from test_training_callback_scalar_noise import model as scalar_model
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_training_event_noise import draw
from test_native_training import RUNNER


def model(backend,ranks,mode,kind,spelling):
    net,g,syn,dt,_,x=scalar_model(backend,ranks,mode,'parameter_'+kind)
    syn.namespace['f']=b.Function(getattr(aliases,spelling+'_'+kind),arg_units=[1,1,1],return_unit=1,stateless=False)
    hidden=next(o for o in net.objects if isinstance(o,b.NeuronGroup) and o is not g)
    bundle=lower_brian_dynamic_training(net,input_group=syn.source,layers=[hidden,g],backend=backend,mpi_ranks=ranks,detach_reset=False,trainable_synapse_parameters={syn.name:['h','mean','scale','gain']})
    return net,g,syn,dt,bundle,x


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('kind',['normal','uniform'])
@pytest.mark.parametrize('spelling',['module','function'])
@pytest.mark.parametrize('ranks',[None,2])
def test_alias_original_restore(engine,mode,kind,spelling,ranks,tmp_path,monkeypatch):
    mpi(ranks);net,g,syn,dt,bundle,x=model(engine,ranks,mode,kind,spelling)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    domain=bundle.provenance['scheduled_noise_domains'][syn.pre.name];calls=[]
    for tick in range(5):
        def sample(*args,**kwargs):
            rows=[edge for edge in range(2) if x[0,tick-1,edge]>0] if tick else []
            assert rows;calls.append((tick,len(rows)))
            np.testing.assert_allclose(args,(.04,.13 if kind=='normal' else .17))
            assert kwargs==({} if kind=='normal' else {'size':None})
            return float(.04+.13*draw('randn' if kind=='normal' else 'rand',p['seed'],9,0,domain,rows[0],tick-1,None,0))
        monkeypatch.setattr(np.random,kind,sample);monkeypatch.setattr(aliases,'imported_'+kind,sample)
        out=t.step(x[:,tick:tick+1],[0],**({'initial':'carry'} if tick else {'noise_sequence':9}))
        net.run(dt,namespace={})
        for name in ('v','u'):
            cells=bundle.provenance['neuron_state_layout'][g.name][name]
            np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],g.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        path=tmp_path/str(tick);t.store(path);t=NativeLIFTrainer(p,runner=RUNNER);t.restore(path)
    assert calls==[(1,2),(3,1)]


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('kind',['normal','uniform'])
@pytest.mark.parametrize('spelling',['module','function'])
@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('ranks',[None,2])
def test_alias_all_vjps(engine,mode,kind,spelling,window,ranks,monkeypatch):
    monkeypatch.setattr(uncaptured,'model',lambda backend,ranks,mode,kind,variant:model(backend,ranks,mode,kind,spelling))
    uncaptured.test_uncaptured_all_vjps(engine,mode,kind,'scalar',window,ranks)


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('spelling',['keyword','imported'])
@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('zero',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
def test_alias_poisson_all_vjps(engine,mode,spelling,window,zero,ranks,monkeypatch):
    import test_training_callback_body_poisson as poisson
    monkeypatch.setattr(poisson,'sample_selected',getattr(aliases,spelling+'_captured_poisson'))
    poisson.test_callback_body_poisson_all_vjps(engine,mode,zero,window,ranks)


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('ranks',[None,2])
def test_alias_poisson_original_restore(engine,mode,ranks,tmp_path,monkeypatch):
    import test_training_callback_body_poisson as poisson
    monkeypatch.setattr(poisson,'sample_selected',aliases.keyword_captured_poisson)
    # Brian's original invocation retains its keywords; adapt only the test
    # observer so it checks the rate without changing callback source.
    setter=monkeypatch.setattr
    def set_sampler(owner,name,value,*args,**kwargs):
        if owner is np.random and name=='poisson':
            original=value
            def value(*args,**kwargs):
                assert not args and set(kwargs)=={'lam','size'} and kwargs['size'] is None
                return original(kwargs['lam'])
        return setter(owner,name,value,*args,**kwargs)
    monkeypatch.setattr(monkeypatch,'setattr',set_sampler)
    poisson.test_callback_body_poisson_original_restore(engine,mode,ranks,tmp_path,monkeypatch)
