"""Independent event-bin simulation and frozen-routing finite differences."""
import copy
import numpy as np
import pytest
import brian2 as b
from brian2_rust import NativeLIFTrainer, lower_brian_dynamic_training
from test_native_training import RUNNER
from test_training_integer_ir import engine
from test_training_delays import cython_cache


def model(**options):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target = 'cython'
    dt = .2*b.ms
    x = np.array([[1,0],[0,1],[1,1],[0,0],[1,0],[1,1],[0,1],[1,0],[0,0],[1,1],[1,0],[1,1]], float)
    times, ids = np.nonzero(x)
    inp = b.SpikeGeneratorGroup(2, ids, times*dt, dt=dt, name='smooth_input')
    layers = [b.NeuronGroup(2, 'dv/dt=-v/ms:1', threshold='v>1', reset='v-=1', dt=dt,
                           method='euler', name=f'smooth_layer_{i}') for i in range(2)]
    layers[0].v = [.2,.4];layers[1].v = [.3,.6]
    synapses = []
    for i, (source, target) in enumerate([(inp,layers[0]), (layers[0],layers[1])]):
        syn = b.Synapses(source, target, 'w:1', dt=dt, name=f'smooth_syn_{i}',
                         on_pre='v_post += w+.08*delay/ms\ndelay=.6*delay+.12*ms*w')
        syn.connect();syn.w = [[.85,.95,1.05,.75],[.7,.8,1.,.9]][i]
        syn.pre.delay = np.array([[.24,.44,.04,.64],[.04,.24,.64,.44]][i])*b.ms
        synapses.append(syn)
    net = b.Network(inp,*layers,*synapses)
    bundle = lower_brian_dynamic_training(net, input_group=inp, layers=layers,
        detach_reset=False, surrogate_slope=3., surrogate_scale=.7, **options)
    bundle.plan['trainable'] = [False]*len(bundle.weights)
    banks = [next(e['bank'] for e in bundle.provenance['bindings'] if e['object']==s.name and e['variables']==['w']) for s in synapses]
    return net, layers, synapses, x, bundle, banks


def routing(plan, initial):
    """Interpret queue metadata only; never read native actions or SSA programs."""
    paths = plan['dynamic']['delay_layout']['paths'];dt = plan['clock']['dt']
    delays = [[int(np.floor(initial[e['delay_state']]/dt+.5)) for e in p['edges']] for p in paths]
    changed = []
    for p, values in zip(paths, delays):
        expected = {(edge,d) for edge,e in enumerate(p['edges']) for d in
                    [values[edge], int(np.floor(plan['dynamic']['initial'][e['delay_state']]/dt+.5))]}
        routes = p.get('routes')
        changed.append(routes is None or {(r['edge'],len(r['states'])) for r in routes} != expected
                       or any(initial[r['selection']] != float(values[r['edge']]==len(r['states'])) for r in routes))
    rebuild = any(changed)
    histories = []
    for path, p in enumerate(paths):
        pending = [(e['edge'], e['states'], True) for e in p['pending']]
        emitted = p.get('routes', [dict(e, edge=i) for i,e in enumerate(p['edges'])])
        emitted = sorted(emitted, key=lambda e:e['event'])
        for edge, indices, old_generation in pending+[(e['edge'], e['states'], changed[path]) for e in emitted]:
            if rebuild and old_generation:
                # Queue presence/compaction is discrete. Freeze this support
                # for finite differences, just as arrival tick choices are frozen.
                nonzero = [i for i,k in enumerate(indices) if initial[k] or plan['dynamic']['initial'][k]]
                indices = indices[:nonzero[-1]+1] if nonzero else []
            histories.append((path, edge, indices))
    return delays, histories


def oracle(plan, weights, banks, x, initial, anchors=None):
    """Pending event bins and analytic model equations, independent of native IR."""
    paths = plan['dynamic']['delay_layout']['paths']
    v = np.array(initial)[plan['dynamic']['voltage']].copy()
    delay_cells = [[e['delay_state'] for e in p['edges']] for p in paths]
    delay = np.array(initial)[delay_cells].copy()
    latched, histories = routing(plan, initial) if anchors is None else (anchors['latched'], anchors['histories'])
    bins = [{},{}]
    for path, edge, indices in histories:
        for offset,k in enumerate(indices):
            bins[path].setdefault(offset,[]).append((edge, initial[k]))
    margins = [];spikes = [];boundaries = []
    window = plan.get('tbptt_window')
    for tick, external in enumerate(x):
        if anchors is not None and window and tick and tick%window == 0:
            v, delay, bins = copy.deepcopy(anchors['boundaries'][tick])
        boundaries.append(copy.deepcopy((v,delay,bins)))
        v *= .8
        margin = v-1.;s = (margin>0).astype(float)
        if anchors is not None:
            m = anchors['margins'][tick]
            s = (m>0).astype(float)+plan['surrogate']['scale']/(1+plan['surrogate']['slope']*abs(m))**2*(margin-m)
        margins.append(margin.copy());spikes.append(s.copy())
        for path in range(2):
            source = external if path==0 else s[:2]
            # Append emissions after already queued generations, in source and
            # original edge order. Each bin is an event list, not a state FIFO.
            for edge in range(4):
                bins[path].setdefault(tick+latched[path][edge],[]).append((edge,source[edge//2]))
            for edge, amplitude in bins[path].pop(tick,[]):
                target = path*2+edge%2;w = weights[banks[path]][edge]
                v[target] += amplitude*(w+.08*delay[path,edge]/.001)
                delay[path,edge] += amplitude*(-.4*delay[path,edge]+.00012*w)
        v -= s
    logits = np.array(spikes)[:,2:].mean(0)*plan['logit_scale']
    loss = np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return dict(loss=loss, v=v, delay=delay, spikes=np.array(spikes), margins=np.array(margins),
                boundaries=boundaries, latched=latched, histories=histories)


@pytest.mark.parametrize('window', [None, 2])
@pytest.mark.parametrize('carry', [False, True])
def test_smooth_delay_state_and_weight_vjps_match_independent_finite_differences(engine, window, carry):
    _, _, _, x, bundle, banks = model(tbptt_window=window)
    bundle.plan['backend'] = engine
    trainer = NativeLIFTrainer(bundle.plan, runner=RUNNER, weights=bundle.weights)
    if carry:
        trainer.step(x[None,:3], [0]);x = x[3:]
    initial = copy.deepcopy(trainer.neuron_state[0] if carry else bundle.initial_state)
    plan = trainer.plan;weights = trainer.state['weights']
    expected = oracle(plan, weights, banks, x, initial)
    actual = trainer.gradients(x[None], [0], initial='carry' if carry else None)
    np.testing.assert_array_equal(actual['spikes'][0], expected['spikes'])
    np.testing.assert_allclose(actual['final_membrane'][0], expected['v'], rtol=4e-5, atol=5e-6)
    for path,p in enumerate(plan['dynamic']['delay_layout']['paths']):
        ids = [e['delay_state'] for e in p['edges']]
        np.testing.assert_allclose(np.asarray(actual['final_state'])[0,ids], expected['delay'][path], rtol=5e-5, atol=2e-9)
    assert actual['loss'] == pytest.approx(expected['loss'], abs=5e-6)
    assert actual['gradient_scope'].endswith('delay-routing')
    for bank, row in enumerate(weights):
        for i,value in enumerate(row):
            a=copy.deepcopy(weights);c=copy.deepcopy(weights);eps=1e-6
            a[bank][i]+=eps;c[bank][i]-=eps
            fd=(oracle(plan,a,banks,x,initial,expected)['loss']-oracle(plan,c,banks,x,initial,expected)['loss'])/(2*eps)
            assert actual['gradients'][bank][i] == pytest.approx(fd, rel=1e-3, abs=1e-5)
    delay_ids = {e['delay_state'] for p in plan['dynamic']['delay_layout']['paths'] for e in p['edges']}
    assert len(actual['initial_state_gradients'][0]) == len(initial)
    for i,value in enumerate(initial):
        eps=1e-9 if i in delay_ids else 1e-6
        a=initial.copy();c=initial.copy();a[i]+=eps;c[i]-=eps
        fd=(oracle(plan,weights,banks,x,a,expected)['loss']-oracle(plan,weights,banks,x,c,expected)['loss'])/(2*eps)
        assert actual['initial_state_gradients'][0][i] == pytest.approx(fd, rel=1e-3, abs=6e-5)


def test_smooth_delay_equations_match_actual_brian():
    net,layers,synapses,x,bundle,banks = model()
    trainer = NativeLIFTrainer(bundle.plan, runner=RUNNER, weights=bundle.weights)
    cursor = 0
    for length in [3,2,3,4]:
        initial = trainer.neuron_state[0] if cursor else bundle.initial_state
        expected = oracle(trainer.plan,trainer.state['weights'],banks,x[cursor:cursor+length],initial)
        result = trainer.step(x[None,cursor:cursor+length],[0],initial='carry' if cursor else None)
        net.run(length*.2*b.ms,namespace={})
        values=np.r_[layers[0].v[:],layers[1].v[:]]
        np.testing.assert_allclose(result['final_membrane'][0],values,rtol=3e-13,atol=1e-13)
        np.testing.assert_allclose(expected['v'],values,rtol=3e-13,atol=1e-13)
        for i,syn in enumerate(synapses):
            np.testing.assert_allclose(expected['delay'][i],np.asarray(syn.pre.delay[:]),rtol=3e-13,atol=1e-15)
            assert syn.pre.codeobj.compiled_code['run'] is not None
        cursor+=length
