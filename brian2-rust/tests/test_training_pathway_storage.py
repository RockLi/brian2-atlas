"""Pathway-owned storage must not silently become an event-code temporary."""
import numpy as np
import pytest
import brian2 as b

from brian2_rust import NativeLIFTrainer, TrainingConversionError, lower_brian_dynamic_training
from test_native_training import RUNNER
from test_training_brian_dynamic import network


@pytest.mark.parametrize('which', ['static_pre', 'stdp_pre', 'stdp_post'])
@pytest.mark.parametrize('statement', ['_source_dt = 0*ms', '_source_dt += .2*ms', '_source_dt *= 2'])
def test_pathway_clock_writes_are_rejected_without_mutation(which, statement):
    net, inp, layers, static, syn, *_ = network()
    path = {'static_pre': static.pre, 'stdp_pre': syn.pre, 'stdp_post': syn.post}[which]
    path.delay = np.arange(len(syn))*.2*b.ms
    path.code += '\n' + statement
    before = [np.asarray(obj.variables[name].get_value()).copy()
              for obj, name in [(path, 'delay'), (static, 'w'), (syn, 'w'),
                                (syn, 'apre'), (syn, 'apost'), (layers[0], 'v'), (layers[1], 'v')]]
    with pytest.raises(TrainingConversionError, match='pathway-owned storage.*_source_dt') as error:
        lower_brian_dynamic_training(net, input_group=inp, layers=layers)
    assert error.value.code == 'pathway'
    assert error.value.owner == path.name
    after = [np.asarray(obj.variables[name].get_value()).copy()
             for obj, name in [(path, 'delay'), (static, 'w'), (syn, 'w'),
                               (syn, 'apre'), (syn, 'apost'), (layers[0], 'v'), (layers[1], 'v')]]
    for a, c in zip(before, after):
        np.testing.assert_array_equal(a, c)
    assert float(net.t/b.second) == 0.


@pytest.mark.parametrize('which', ['static_pre', 'stdp_pre', 'stdp_post'])
def test_event_local_temporaries_keep_their_original_semantics(which):
    net, inp, layers, static, syn, _, _, x = network()
    options = dict(input_group=inp, layers=layers)
    baseline = lower_brian_dynamic_training(net, **options)
    expected = NativeLIFTrainer(baseline.plan, runner=RUNNER, weights=baseline.weights).gradients(x[None], [0])
    path = {'static_pre': static.pre, 'stdp_pre': syn.pre, 'stdp_post': syn.post}[which]
    # This is a true local, not a physical variable in either variable owner.
    assert 'local_increment' not in path.variables
    if which == 'stdp_post':
        path.code = path.code.replace('apost+=Am', 'local_increment=Am\napost+=local_increment')
    else:
        path.code = path.code.replace('v_post+=w', 'local_increment=w\nv_post+=local_increment')
    bundle = lower_brian_dynamic_training(net, **options)
    actual = NativeLIFTrainer(bundle.plan, runner=RUNNER, weights=bundle.weights).gradients(x[None], [0])
    for key in ('final_state', 'spikes', 'initial_state_gradients', 'logits'):
        np.testing.assert_array_equal(actual[key], expected[key])
    for a, c in zip(actual['gradients'], expected['gradients']):
        np.testing.assert_array_equal(a, c)
