"""Opt-in, workload-specific Brian2GeNN 1.7.0 delayed-STDP schedule adapter.

This is a corrected generated-source experiment, never the stock backend. It
retains Brian2GeNN's storage, connectivity, event code and compilation. Exact
preconditions fail closed if the pinned generated form changes. Original and
adapted source plus unified diffs are saved before compiling.
"""
import argparse
import difflib
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import time
import traceback
import numpy as np
from gpu_stdp_compare import DT, FIELDS, checks, configuration, oracle, topology, workload_options

POLICY='brian2genn-1.7.0-stdp-schedule-v1'
FACTOR_POLICY='brian2genn-1.7.0-stdp-schedule-f32-factor-v1'


def replace_once(text, old, new):
    if text.count(old)!=1:raise ValueError(f'Expected one generated fragment: {old!r}')
    return text.replace(old,new)


def transform_model(source,steps,delays,*,drive=.125,post_delay=2,decay_mode="generated"):
    """Narrow transformation of this benchmark's generated GeNN model."""
    if decay_mode not in {"generated","f32-factor"}:raise ValueError("Unknown decay mode")
    def transform_class(text,name,fn):
        # Weight update classes have a blank line before IMPLEMENT_MODEL.
        pattern=rf'(class {name} : .*?\n\}};\s*IMPLEMENT_MODEL\({name}\);)'
        matches=list(re.finditer(pattern,text,re.S))
        if len(matches)!=1:raise ValueError(f'Expected one class {name}')
        match=matches[0]
        return text[:match.start()]+fn(match.group())+text[match.end():]
    def macro(text,name,fn):
        pattern=rf'{name}\("((?:\\.|[^"\\])*)"\);'
        matches=list(re.finditer(pattern,text,re.S))
        if len(matches)!=1:raise ValueError(f'Expected one {name}')
        match=matches[0]
        code=json.loads('"'+match.group(1).replace('\\\n','')+'"')
        return text[:match.start()]+name+'('+json.dumps(fn(code))+');'+text[match.end():]
    def neuron(text):
        for var in ('{"previous_spike", "int32_t"}','{"v_start", "float"}'):
            if var not in text:raise ValueError(f'Missing adapter storage {var}')
        def sim(code):
            code=replace_once(code,'char _cond = $(v) > 1;','_cond = $(v) > 1;')
            return ('char _cond = false; $(previous_spike)=0; '
                    f'if(t < {steps*DT!r}) {{ $(v)+={float(drive)!r}f; $(v_start)=$(v); '+code+' }')
        text=macro(text,'SET_SIM_CODE',sim)
        return macro(text,'SET_RESET_CODE',lambda code:replace_once(code,'$(v) = 0;','$(v) = 0; $(previous_spike)=1;'))
    source=transform_class(source,'populationNEURON',neuron)
    for name,delay in delays.items():
        def weight(text):
            def shift(code):
                # Both event-driven traces and lastupdate share the shifted time.
                if len(re.findall(r'\bt\b',code))!=3:raise ValueError('Unexpected event time expressions')
                return re.sub(r'\bt\b','(t - DT)',code)
            return macro(macro(text,'SET_SIM_CODE',shift),'SET_LEARN_POST_CODE',shift)
        source=transform_class(source,name+'WEIGHTUPDATE',weight)
        def postsyn(text):
            return macro(text,'SET_APPLY_INPUT_CODE',lambda code:replace_once(code,
                '$(v) += $(inSyn);','if(!$(previous_spike)) $(v) += $(inSyn);'))
        source=transform_class(source,name+'POSTSYN',postsyn)
        pattern=rf'(const unsigned int delaySteps = )\d+(;\s+auto \*syn = model.addSynapsePopulation<{name}WEIGHTUPDATE, {name}POSTSYN>)'
        source,count=re.subn(pattern,lambda m:m[1]+str(delay)+m[2],source)
        if count!=1:raise ValueError(f'Missing delay for {name}')
        needle=f'syn->setMaxSourceConnections(maxCol{name});'
        source=replace_once(source,needle,needle+f'\n    syn->setBackPropDelaySteps({post_delay});')
    return round_decay_factor(source,len(delays)) if decay_mode=="f32-factor" else source


def round_decay_factor(source, groups):
    """Change only the four generated decay expressions per delay group."""
    if type(groups) is not int or not 1 <= groups <= 16:
        raise ValueError('Expected 1..16 delay groups')
    if source.count('{"lastupdate", "double"}') != groups:
        raise ValueError('Expected double event timestamps in every group')
    if source.count('model.setTimePrecision(TimePrecision::DOUBLE);') != 1:
        raise ValueError('Expected double model time')
    for variable, tau in [('Apre', 'taupre'), ('Apost', 'taupost')]:
        original = f'float _{variable} = $({variable}) * exp(1.0f*(- ((t - DT) - $(lastupdate)))/$({tau}));'
        if source.count(original) != 2 * groups:
            raise ValueError('Unexpected generated decay expression: ' + variable)
        source = source.replace(original, original.replace(' * exp(', ' * (float)exp('))
    return source


def transform_engine(source):
    # One unobserved final step delivers the last Brian tick's events. The neuron
    # guard prevents a further integration/spike/drive. Report logical duration.
    if source.count('stepTime();')!=1:raise ValueError('Unexpected engine stepping')
    if '_run_population_run_regularly_' in source:raise ValueError('Drive must be in GPU neuron code')
    return replace_once(source,'  current= std::clock();',
        '  stepTime(); // adapter: final synapse flush, no neuron advance\n'
        '  iT--; t = iT*DT; // report the requested logical duration\n'
        '  current= std::clock();')


def build_network(neurons,degree,*,delay_span=4,post_delay=2,topology_kind="ring",topology_seed=0):
    import brian2 as b
    dt=DT*b.second;b.defaultclock.dt=dt
    pop=b.NeuronGroup(neurons,'dv/dt=-v/tau:1\nprevious_spike:integer\nv_start:1',
        threshold='v>1',reset='v=0',method='euler',namespace={'tau':32*dt},clock=b.defaultclock,name='population')
    pop.v=np.arange(neurons)%16/16
    source,target,delays=topology(neurons,degree,delay_span=delay_span,topology_kind=topology_kind,topology_seed=topology_seed);groups=[];mapping={}
    for q,d in enumerate(reversed(range(delay_span))):
        edges=np.flatnonzero(delays==d)
        if not len(edges):continue
        name='plastic'+str(q);mapping[name]=d
        syn=b.Synapses(pop,pop,'dApre/dt=-Apre/taupre:1 (event-driven)\ndApost/dt=-Apost/taupost:1 (event-driven)\nw:1',
            on_pre='v_post+=w/16; Apre+=0.0078125; w=clip(w+Apost,0,0.5)',
            on_post='Apost-=0.00390625; w=clip(w+Apre,0,0.5)',
            namespace={'taupre':16*dt,'taupost':32*dt},clock=pop.clock,name=name,
            delay={'pre':d*dt,'post':post_delay*dt})
        syn.connect(i=source[edges],j=target[edges]);syn.w=.25;groups.append((syn,edges))
    spikes=b.SpikeMonitor(pop,name='spikes')
    # v_start is captured on GPU after arrivals and drive, before integration;
    # its end-slot readback records the original model's start-slot observation.
    trace=b.StateMonitor(pop,'v_start',record=[0,neurons-1],when='end',name='voltage')
    return b.Network(pop,*(s for s,e in groups),spikes,trace),pop,groups,spikes,trace,mapping


def run(neurons,degree,steps,output,*,prepared=None,drive=.125,delay_span=4,post_delay=2,topology_kind="ring",topology_seed=0,decay_mode="generated"):
    if decay_mode not in {'generated','f32-factor'}:raise ValueError('Unknown decay mode')
    policy=FACTOR_POLICY if decay_mode=='f32-factor' else POLICY
    opts=workload_options(drive,delay_span,post_delay,topology_kind,topology_seed)
    import brian2 as b
    import brian2genn
    if importlib.metadata.version('brian2genn')!='1.7.0':raise ValueError('Adapter requires Brian2GeNN 1.7.0')
    b.get_device().reinit();b.prefs.core.default_float_dtype=np.float32
    project=output/'project'
    b.set_device('genn',directory=str(project),use_GPU=True,compile=False,run=False)
    b.prefs.devices.cpp_standalone.extra_make_args_unix=['-j2']
    net,pop,groups,spikes,trace,mapping=build_network(neurons,degree,delay_span=delay_span,post_delay=post_delay,topology_kind=topology_kind,topology_seed=topology_seed)
    start=time.perf_counter();net.run(steps*DT*b.second)
    # Generate inside network_run: Brian2GeNN marks run_statement_used on exit,
    # so its delayed manual build path incorrectly rejects queued initialization.
    dev=b.get_device()
    transformations={}
    for name,transform in [('magicnetwork_model.cpp',lambda s:transform_model(s,steps,mapping,drive=drive,post_delay=post_delay,decay_mode=decay_mode)),('engine.cpp',transform_engine)]:
        path=project/name;before=path.read_text();after=transform(before)
        (output/(name+'.original')).write_text(before)
        (output/(name+'.patch')).write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=name+'.original',tofile=name)))
        path.write_text(after)
        transformations[name]=dict(before_sha256=hashlib.sha256(before.encode()).hexdigest(),after_sha256=hashlib.sha256(after.encode()).hexdigest())
    (output/'transformations.json').write_text(json.dumps(dict(policy=policy,pre_delay_steps=mapping,post_delay_steps=post_delay,workload=opts,files=transformations),indent=2)+'\n')
    dev.compile_source(debug=False,directory=str(project),use_GPU=True)
    compiled=time.perf_counter();dev.run(str(project),True,True)
    context=dict(device=dev,network=net,population=pop,groups=groups,monitor=spikes,trace=trace,
        neurons=neurons,degree=degree,project=project)
    if prepared is not None:prepared.update(context)
    result=read_result(context)
    return result,dict(build_seconds=compiled-start,run_readback_seconds=time.perf_counter()-compiled,policy=policy)


def invalidate_results(context):
    """Invalidate every recorded/learned output before a fresh native process."""
    pop=context['population'];monitor=context['monitor'];trace=context['trace']
    variables=[pop.variables['v'],monitor.variables['t'],monitor.variables['i'],trace.variables['v_start']]
    for syn,edges in context['groups']:variables.extend(syn.variables[k] for k in FIELDS)
    for var in variables:context['device'].array_cache[var]=None


def read_result(context):
    import brian2 as b
    pop=context['population'];spikes=context['monitor'];trace=context['trace']
    groups=context['groups'];neurons=context['neurons'];degree=context['degree']
    ticks=np.rint(np.asarray(spikes.t[:]/b.second)/DT).astype(np.int64);indices=np.asarray(spikes.i[:],np.int64)
    order=np.lexsort((indices,ticks))
    result=dict(v=np.asarray(pop.v[:]).copy(),ticks=ticks[order],indices=indices[order],trace=np.asarray(trace.v_start).T.copy())
    for key in FIELDS:
        result[key]=np.empty(neurons*degree,np.float64 if key=='lastupdate' else np.float32)
        for syn,edges in groups:
            values=getattr(syn,key)[:];result[key][edges]=np.asarray(values/b.second if key=='lastupdate' else values)
    return result


def replay(context,count):
    """Reset through a fresh process; keep every previous result off the read path."""
    project=context['project'];archive=project.parent/f'previous-results-{count}'
    started=time.perf_counter();archive.mkdir(exist_ok=False)
    for name in ('results','test_output'):
        path=project/name
        if path.exists():
            if path.is_symlink() or not path.is_dir():raise RuntimeError('Unexpected output directory: '+str(path))
            path.rename(archive/name)
    (project/'results').mkdir()
    invalidate_results(context);prepared=time.perf_counter()
    context['device'].run(str(project),True,False);executed=time.perf_counter()
    result=read_result(context);decoded=time.perf_counter()
    return result,dict(output_rotation_and_cache_reset_seconds=prepared-started,
        process_initialize_run_and_write_seconds=executed-prepared,
        array_read_and_decode_seconds=decoded-executed)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name,default in [('neurons',256),('degree',128),('steps',128)]:p.add_argument('--'+name,type=int,default=default)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();topology(a.neurons,a.degree)
    if not 1<=a.steps<=512:p.error('steps must be 1..512')
    a.output.mkdir(parents=True,exist_ok=False)
    config=configuration(a.neurons,a.degree,a.steps)
    report=dict(schema='b2-genn-stdp-adapter-v1',backend='brian2genn-corrected',policy=POLICY,configuration=config,
        model_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),
        timing_eligible=False,scope='workload-specific corrected generated source; not stock Brian2GeNN performance',
        versions={k:importlib.metadata.version(k) for k in ('brian2genn','brian2','numpy')})
    try:
        result,timing=run(a.neurons,a.degree,a.steps,a.output)
        np.savez_compressed(a.output/'result.npz',**result)
        report.update(timing=timing,reference_f64=checks(result,oracle(a.neurons,a.degree,a.steps)),
            reference_f32=checks(result,oracle(a.neurons,a.degree,a.steps,np.float32)))
        report['status']='passed' if report['reference_f64']['passed'] and report['reference_f32']['passed'] else 'correctness_failed'
    except Exception as error:report.update(status='execution_error',error=str(error),traceback=traceback.format_exc())
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    if report['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
