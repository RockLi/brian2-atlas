"""Transmitter-informed conductance LIF with frozen Poisson background and DM1 input.

This reduced model has explicit engineering calibration targets, not a claim of
experimentally established whole-brain physiology or an actual odor identity.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from brian2_rust.binary_topology import inspect_csr,csr_arrays,file_hash
from flywire_benchmark import rss
from performance_suite import source_provenance

DEFAULTS={'seed':783,'duration_ms':1000.,'dt_ms':.1,'weight_mv':.275,
          'inhibitory_gain':4.,'background_rate_hz':300.,'background_weight_mv':3.5,
          'background_channels':512,'sensory_rate_hz':80.,'sensory_weight_mv':40.,
          'stimulus_start_ms':300.,'stimulus_end_ms':700.}


def poisson_schedule(n, rate, dt, duration, seed, start=0., end=None):
    """Frozen Brian-style per-bin Bernoulli approximation to Poisson spikes."""
    end=duration if end is None else min(end,duration)
    begin=int(round(start/dt)); stop=int(round(end/dt))
    if n < 1 or not 0 < rate*dt/1000 < 1 or begin < 0:
        raise ValueError('invalid frozen Poisson configuration')
    if stop <= begin:return np.empty(0,dtype=np.int32),np.empty(0,dtype=np.int32)
    rng=np.random.default_rng(seed); ids=[]; ticks=[]
    for i in range(n):
        # Independent geometric waiting times equal Bernoulli events per bin.
        t=begin-1
        while True:
            t+=int(rng.geometric(rate*dt/1000))
            if t>=stop:break
            ids.append(i);ticks.append(t)
    ids=np.asarray(ids,dtype=np.int32);ticks=np.asarray(ticks,dtype=np.int32)
    order=np.lexsort((ids,ticks))
    return ids[order],ticks[order]


def read_snapshot(backend,folder,result):
    if backend=='cpp':
        layout=json.loads((folder/'array_layout.json').read_text())
        return {key:np.fromfile(result/item['file'],dtype=item['dtype']).reshape(item['shape'])
                for key,item in layout.items()}
    from brian2_rust.results import load_results
    model=json.loads((folder/'project/model.json').read_text())
    populations=load_results(model,result)['populations']
    index=next(i for i,p in enumerate(model['definition']['populations']) if p['name']=='flywire_neurons')
    p=populations[index]
    return {**{k:p['states'][k] for k in ('v','ge','gi','transmission')},
            **{'trace_'+k:p['trace'][k] for k in ('v','ge','gi')},
            'trace_t':p['times'],'lastspike':p['refractory']['lastspike'],
            'not_refractory':p['refractory']['not_refractory'],
            'spike_i':p['indices'],'spike_t':p['spike_times'],'spike_count':p['counts']}


def build(graph, output, backend='aot', threads=1, condition='odor', config=None,
          *, ranks=2, export_only=False):
    import brian2 as b
    import brian2_rust as rust
    cfg={**DEFAULTS,**(config or {})}
    if backend not in ('aot', 'cpp', 'mpi') or (export_only and backend == 'cpp'):
        raise ValueError('export requires a Rust backend')
    if condition not in ('rest','odor','cut','cut_rest'):raise ValueError('unknown condition')
    if any(not np.isfinite(cfg[k]) or cfg[k]<0 for k in cfg):raise ValueError('invalid model parameter')
    started=time.perf_counter();graph=Path(graph).resolve();output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    info=inspect_csr(graph/'connectome.b2csr');manifest=json.loads((graph/'manifest.json').read_text())
    if file_hash(info['path'])!=manifest['csr_sha256']:raise ValueError('EI graph checksum mismatch')
    data=np.load(graph/'annotations.npz'); n=info['source_count']; sensory=data['sensory']
    channels=min(n,int(cfg['background_channels']));dt=cfg['dt_ms']*b.ms
    bg_i,bg_ticks=poisson_schedule(channels,cfg['background_rate_hz'],cfg['dt_ms'],cfg['duration_ms'],cfg['seed']+1)
    stim_i,stim_ticks=poisson_schedule(len(sensory),cfg['sensory_rate_hz'],cfg['dt_ms'],cfg['duration_ms'],cfg['seed']+2,
                                      cfg['stimulus_start_ms'],cfg['stimulus_end_ms'])
    project=output/'project'
    if backend in ('aot', 'mpi'):
        b.set_device('atlas',engine=backend,threads=threads,directory=project,
                     runner=ROOT/'target/release/b2-runner',
                     **({'ranks':ranks} if backend=='mpi' else {}))
    else:
        b.prefs.codegen.cpp.extra_compile_args=['-O3','-std=c++17','-fno-fast-math','-ffp-contract=off']
        b.prefs.devices.cpp_standalone.extra_make_args_unix=['-j2']
        b.prefs.devices.cpp_standalone.openmp_threads=0 if threads==1 else threads
        b.set_device('cpp_standalone',build_on_run=False)
    # Inputs below already use cfg['seed']; also pin the (currently unused)
    # backend RNG field so deferred exports are identical across activations.
    b.seed(int(cfg['seed']))
    g=b.NeuronGroup(n,'''dv/dt=(-(v+52*mV)-ge*v-gi*(v+70*mV))/(20*ms) : volt (unless refractory)
                         dge/dt=-ge/(5*ms) : 1
                         dgi/dt=-gi/(5*ms) : 1
                         transmission : 1''',
                     threshold='v>-45*mV',reset='v=-52*mV',refractory=2.2*b.ms,
                     method='euler',dt=dt,name='flywire_neurons')
    rng=np.random.default_rng(int(cfg['seed']))
    g.v=(-52+rng.uniform(-.8,.8,n))*b.mV
    g.ge=cfg['background_rate_hz']*.005*cfg['background_weight_mv']/52
    g.gi=0
    transmission=np.ones(n)
    if condition in ('cut','cut_rest'):transmission[sensory]=0
    g.transmission=transmission
    # Convert reference current amplitude into conductance relative to leak;
    # E_exc=0 mV and E_inh=-70 mV make current sign voltage-dependent.
    w=cfg['weight_mv']/52; inhibition=cfg['inhibitory_gain']
    syn=b.Synapses(g,g,'signed_contacts : 1 (constant)',
                   on_pre='''ge_post += w * signed_contacts * int(signed_contacts>0) * transmission_pre
                             gi_post -= w * inhibition * signed_contacts * int(signed_contacts<0) * transmission_pre''',
                   delay=1.8*b.ms,clock=g.clock,name='flywire_recurrent',
                   namespace={'w':w,'inhibition':inhibition})
    if backend in ('aot', 'mpi'):rust.connect_binary_csr(syn,info['path'],parameters={'signed_contacts':0})
    else:
        offsets,targets,values=csr_arrays(info)
        syn.connect(i=np.repeat(np.arange(n,dtype=np.int32),np.diff(offsets).astype(np.int64)),
                    j=targets.astype(np.int32),namespace={})
        syn.signed_contacts=values[0]
        del offsets,targets,values
    background=b.SpikeGeneratorGroup(channels,bg_i,bg_ticks*dt,clock=g.clock,sorted=True,name='flywire_background')
    bg=b.Synapses(background,g,'amplitude : 1 (constant)',on_pre='ge_post += amplitude',
                  delay=0*b.ms,clock=g.clock,name='flywire_background_connections')
    mapping=rng.permutation(n)%channels
    bg.connect(i=mapping,j=np.arange(n),namespace={})
    bg.amplitude=cfg['background_weight_mv']/52
    stimulus=b.SpikeGeneratorGroup(len(sensory),stim_i,stim_ticks*dt,clock=g.clock,sorted=True,name='flywire_stimulus')
    drive=b.Synapses(stimulus,g,'amplitude : 1 (constant)',on_pre='ge_post += amplitude',
                     delay=0*b.ms,clock=g.clock,name='flywire_stimulus_connections')
    drive.connect(i=np.arange(len(sensory)),j=sensory,namespace={})
    drive.amplitude=0 if condition in ('rest','cut_rest') else cfg['sensory_weight_mv']/52
    records=np.unique(np.r_[np.linspace(0,n-1,min(12,n),dtype=int),
                             *[data[k][:2] for k in ('sensory','pn','kc','mbon')]])
    trace=b.StateMonitor(g,['v','ge','gi'],record=records,name='flywire_trace')
    spikes=b.SpikeMonitor(g,name='flywire_spikes')
    net=b.Network(g,syn,background,bg,stimulus,drive,trace,spikes)
    if export_only:
        # Use the identical model and frozen inputs for a cluster build; only
        # execution is deferred. No alternate equations or reduced topology.
        from brian2_rust.export import lower_network
        model=lower_network(net,cfg['duration_ms']*b.ms,namespace={},
                            rng_seed=b.get_device()._rng_seed)
        (output/'model.json').write_text(json.dumps(model,sort_keys=True)+'\n')
        exported={'condition':condition,'config':cfg,'neurons':n,
                  'weighted_edges':info['edge_count'],'graph_sha256':manifest['csr_sha256'],
                  'recorded_neuron_indices':records.tolist(),
                  'background_spikes':len(bg_i),'sensory_spikes':len(stim_i),
                  'input_sha256':hashlib.sha256(bg_i.tobytes()+bg_ticks.tobytes()+stim_i.tobytes()+stim_ticks.tobytes()+mapping.tobytes()).hexdigest()}
        (output/'export.json').write_text(json.dumps(exported,indent=2)+'\n')
        return model
    net.run(cfg['duration_ms']*b.ms,namespace={})
    if backend=='cpp':b.device.build(directory=str(project),compile=True,run=True,with_output=False)
    arrays={**{k:g.variables[k] for k in ('v','ge','gi','transmission','lastspike','not_refractory')},
            **{'trace_'+k:trace.variables[k] for k in ('v','ge','gi')},'trace_t':trace.variables['t'],
            'spike_i':spikes.variables['i'],'spike_t':spikes.variables['t'],'spike_count':spikes.variables['count']}
    snapshot={k:np.asarray(v.get_value()).copy() for k,v in arrays.items()}
    np.savez(output/'snapshot.npz',**snapshot)
    if backend=='cpp':
        layout={k:{'file':b.get_device().get_array_filename(v),'dtype':np.dtype(v.dtype).str,
                   'shape':list(snapshot[k].shape)} for k,v in arrays.items()}
        (output/'array_layout.json').write_text(json.dumps(layout,indent=2)+'\n')
        loop=float((project/'results/last_run_info.txt').read_text().split()[0])
    else:loop=json.loads((project/'rust/summary.json').read_text())['timings']['simulation_and_recording_seconds']
    report={'backend':backend,'threads':threads,'condition':condition,'config':cfg,
            'model':'conductance LIF; E_exc=0mV, E_inh=-70mV, E_leak=-52mV; g relative to leak; 5ms conductance decay',
            'neurons':n,'weighted_edges':info['edge_count'],'spikes':len(snapshot['spike_i']),
            'first_loop_seconds':loop,'end_to_end_seconds':time.perf_counter()-started,
            'frontend_peak_rss_bytes':rss(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            'graph_sha256':manifest['csr_sha256'],'recorded_neuron_indices':records.tolist(),
            'background_spikes':len(bg_i),'sensory_spikes':len(stim_i),
            'input_sha256':hashlib.sha256(bg_i.tobytes()+bg_ticks.tobytes()+stim_i.tobytes()+stim_ticks.tobytes()+mapping.tobytes()).hexdigest(),
            'brian2':b.__version__,'python':platform.python_version(),
            **source_provenance(ROOT)}
    (output/'build.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    return snapshot


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--backend',choices=['aot','cpp','mpi'],default='aot');p.add_argument('--threads',type=int,default=1)
    p.add_argument('--ranks',type=int,default=2);p.add_argument('--export-only',action='store_true')
    p.add_argument('--condition',choices=['rest','odor','cut','cut_rest'],default='odor');p.add_argument('--config',type=Path)
    a=p.parse_args();build(a.graph,a.output,a.backend,a.threads,a.condition,
                         None if a.config is None else json.loads(a.config.read_text()),
                         ranks=a.ranks,export_only=a.export_only)
