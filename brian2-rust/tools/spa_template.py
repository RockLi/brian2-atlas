"""Export classic single-cell model templates without running a simulation.

Izhikevich (2003): https://izhikevich.org/publications/spikes.htm
HH: Brian2's hodgkin_huxley_1952 example, with voltage shifted by -65 mV,
reduced to independent isopotential cells (no spatial cable).
Numeric v is in mV for Izhikevich/HH; time is expressed in Brian units.
"""
import brian2 as b
from brian2_rust.export import lower_network

MODEL_IDS = ('adaptive_lif', 'izhikevich', 'hodgkin_huxley')


def create_group(model_id, count=8, dt_ms=None):
    """Shared equation definitions; also usable with Brian's NumPy runtime."""
    if model_id == 'adaptive_lif':
        group = b.NeuronGroup(count, '''dv/dt=(drive-v-w)/tau:1 (unless refractory)
            dw/dt=-w/tau_adapt:1
            drive:1 (constant)
            tau:second (constant, shared)
            tau_adapt:second (constant, shared)
            adaptation:1 (constant, shared)''',
            threshold='v>1', reset='v=0; w+=adaptation', refractory=2*b.ms,
            method='euler', dt=(dt_ms or .1)*b.ms, name='cortex')
        group.v=0; group.w=0; group.drive=1.5; group.tau=20*b.ms
        group.tau_adapt=120*b.ms; group.adaptation=.08
        return group, ['v', 'w']
    if model_id == 'izhikevich':
        # Factor the quadratic to avoid libm pow(x, 2) differences amplifying
        # across threshold/reset events in long fast-spiking runs.
        group = b.NeuronGroup(count, '''dv/dt=(0.04*v*(v+125)+140-u+drive)/ms:1
            du/dt=a*(b*v-u)/ms:1
            drive:1 (constant)
            a:1 (constant, shared)
            b:1 (constant, shared)
            c:1 (constant, shared)
            d:1 (constant, shared)''',
            threshold='v>=30', reset='v=c; u+=d', method='euler',
            dt=(dt_ms or .1)*b.ms, name='cortex')
        group.v=-65; group.u=-13; group.drive=10
        group.a=.02; group.b=.2; group.c=-65; group.d=8
        return group, ['v', 'u']
    if model_id == 'hodgkin_huxley':
        group = b.NeuronGroup(count, '''
            dv/dt=(drive-g_na*m**3*h*(v-50)-g_k*n**4*(v+77)-g_l*(v+54.387))/ms:1
            dm/dt=(alpha_m*(1-m)-beta_m*m)/ms:1
            dh/dt=(alpha_h*(1-h)-beta_h*h)/ms:1
            dn/dt=(alpha_n*(1-n)-beta_n*n)/ms:1
            alpha_m=1/exprel(-(v+40)/10):1
            beta_m=4*exp(-(v+65)/18):1
            alpha_h=.07*exp(-(v+65)/20):1
            beta_h=1/(1+exp(-(v+35)/10)):1
            alpha_n=.1/exprel(-(v+55)/10):1
            beta_n=.125*exp(-(v+65)/80):1
            drive:1 (constant)
            g_na:1 (constant, shared)
            g_k:1 (constant, shared)
            g_l:1 (constant, shared)''',
            threshold='v>0', refractory='v>-40',
            method='exponential_euler', dt=(dt_ms or .025)*b.ms, name='cortex')
        group.v=-65; group.drive=10; group.g_na=120; group.g_k=36; group.g_l=.3
        group.m='alpha_m/(alpha_m+beta_m)'
        group.h='alpha_h/(alpha_h+beta_h)'
        group.n='alpha_n/(alpha_n+beta_n)'
        return group, ['v', 'm', 'h', 'n']
    raise ValueError(f'Unknown model: {model_id}')


def make_template(model_id='adaptive_lif'):
    device=b.get_device()
    b.set_device('rust_standalone', build_on_run=False)
    try:
        group, variables=create_group(model_id)
        states=b.StateMonitor(group, variables, record=list(range(8)), name='state_trace')
        spikes=b.SpikeMonitor(group, name='spike_trace')
        return lower_network(b.Network(group, states, spikes), 600*b.ms, rng_seed=42)
    finally:
        b.get_device().reinit()
        b.set_device(device)


def write_templates(output):
    import json
    for model_id in MODEL_IDS:
        filename='editor-template.json' if model_id=='adaptive_lif' else f'{model_id}-template.json'
        (output/filename).write_text(json.dumps(make_template(model_id)))
    for count in (240,1024,4096):
        suffix='' if count==240 else f'-{count}'
        (output/f'flywire{suffix}-template.json').write_text(json.dumps(make_flywire_template(count)))


def create_flywire_network(circuit, dt_ms=.1):
    """Reduced version of examples/flywire_device.py, using the induced graph.

    v is numerically in mV. Background channels are independent per cell here,
    unlike the shared 512-channel full-brain benchmark. Both random draws are
    made each tick in every condition to permit matched stimulus/cut controls.
    """
    import numpy as np
    count=len(circuit['nodes'])
    group=b.NeuronGroup(count, '''
        dv/dt=(-(v+52)-ge*v-gi*(v+70))/(20*ms):1 (unless refractory)
        dge/dt=-ge/(5*ms):1
        dgi/dt=-gi/(5*ms):1
        is_sensory:1 (constant)
        transmission:1
        background_rate:Hz (constant, shared)
        background_gain:1 (constant, shared)
        sensory_rate:Hz (constant, shared)
        sensory_gain:1 (constant, shared)
        stim_start:second (constant, shared)
        stim_end:second (constant, shared)''',
        threshold='v>-45', reset='v=-52', refractory=2.2*b.ms,
        method='euler',dt=dt_ms*b.ms,name='cortex')
    group.v=-52;group.ge=300*.005*3.5/52;group.gi=0
    group.is_sensory=[int(n['group']=='sensory') for n in circuit['nodes']]
    group.transmission=1;group.background_rate=300*b.Hz;group.background_gain=3.5/52
    group.sensory_rate=80*b.Hz;group.sensory_gain=40/52
    group.stim_start=80*b.ms;group.stim_end=220*b.ms
    group.run_regularly('''ge += background_gain*int(rand()<background_rate*dt)
        ge += sensory_gain*is_sensory*int(t>=stim_start)*int(t<stim_end)*int(rand()<sensory_rate*dt)''',when='start',order=-1,name='external_input')
    syn=b.Synapses(group,group,'signed_contacts:1 (constant)',on_pre='''
        ge_post += recurrent_weight*signed_contacts*int(signed_contacts>0)*transmission_pre
        gi_post -= recurrent_weight*inhibitory_gain*signed_contacts*int(signed_contacts<0)*transmission_pre''',
        namespace={'recurrent_weight':.275/52,'inhibitory_gain':4.},delay=1.8*b.ms,
        clock=group.clock,name='flywire_recurrent')
    syn.connect(i=np.array([e['source'] for e in circuit['edges']]),j=np.array([e['target'] for e in circuit['edges']]))
    syn.signed_contacts=[e['signed_contacts'] for e in circuit['edges']]
    records=[]
    for category in circuit['groups']:
        indices=[i for i,node in enumerate(circuit['nodes']) if node['group']==category['id']]
        records.extend([indices[0],indices[-1]])
    states=b.StateMonitor(group,['v','ge','gi'],record=sorted(set(records)),name='state_trace')
    spikes=b.SpikeMonitor(group,name='spike_trace')
    return b.Network(group,syn,states,spikes),group,syn,states,spikes


def make_flywire_template(count=240):
    import json
    from pathlib import Path
    suffix='' if count==240 else f'-{count}'
    circuit=json.loads((Path(__file__).resolve().parents[1]/f'wasm/flywire-circuit{suffix}.json').read_text())
    device=b.get_device();b.set_device('rust_standalone',build_on_run=False)
    try:
        net,*_=create_flywire_network(circuit)
        return lower_network(net,300*b.ms,rng_seed=783)
    finally:
        b.get_device().reinit();b.set_device(device)
