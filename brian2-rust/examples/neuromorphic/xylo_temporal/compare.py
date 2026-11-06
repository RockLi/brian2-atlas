"""Compare native Atlas temporal dynamics with validated Xylo Audio 2 configs."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import brian2 as b
from brian2_atlas import export_network
from brian2_atlas.neuromorphic import export_ir
from brian2_rust.results import load_results

UPDATE = """I = I - (floor(I/syn_div) + int(I>0)*int(I<syn_div))
delta = clip(-(floor(v/mem_div) + int(v>0)*int(v<mem_div)) + I, -32768, 32767)
delta = clip(delta+bias, -32768, 32767)
v = clip(v+delta, -32768, 32767)
q = clip(floor(v/threshold), 0, spike_cap)"""


def cases():
    raster=np.zeros((64,2),int)
    raster[[0,4,8,20,31],0]=[1,3,2,1,4]
    raster[[1,8,12,21,40],1]=[1,2,1,3,2]
    base=dict(raster=raster.tolist(),input_weights=[[8,-4,2],[0,6,8]],
              recurrent_weights=[[2,0,-1],[1,0,0],[0,-2,0]],readout_weights=[[4,0],[-3,2],[2,4]],
              hidden=dict(syn=[1,2,3],mem=[2,3,4],threshold=[5,8,7],bias=[0,0,0]),
              output=dict(syn=[2,3],mem=[2,2],threshold=[3,4],bias=[0,0]))
    result={'recurrent':copy.deepcopy(base)}
    def add(name,**changes):
        c=copy.deepcopy(base);c.update(changes);result[name]=c;return c
    add('silent',raster=np.zeros_like(raster).tolist())
    for name,tick in [('impulse_at_zero',0),('event_at_end',63)]:
        r=np.zeros_like(raster);r[tick,0]=15 if name=='event_at_end' else 1;add(name,raster=r.tolist())
    c=add('threshold_equality');c['raster']=np.zeros_like(raster).tolist();c['raster'][0][0]=1
    c['hidden']['threshold']=[4,8,1];c['hidden']['syn']=[1,1,1];c['recurrent_weights']=np.zeros((3,3),int).tolist()
    c=add('minimum_decay');c['raster']=np.zeros_like(raster).tolist();c['raster'][0][0]=1
    c['input_weights']=[[2,-3,-1],[0,0,0]];c['hidden']['syn']=[15,1,15]
    c['hidden']['threshold']=[32767]*3;c['recurrent_weights']=np.zeros((3,3),int).tolist()
    c=add('multi_spike_caps',raster=np.full((64,2),15).tolist())
    c['input_weights']=[[127,127,127],[127,127,127]]
    c['hidden']['threshold']=[1]*3;c['output']['threshold']=[1]*2
    c['readout_weights']=[[127,127]]*3
    for name,weight in [('positive_saturation',127),('negative_saturation',-128)]:
        c=add(name,raster=np.full((128,16),15).tolist())
        c['input_weights']=[[weight]*3 for _ in range(16)]
        c['hidden']['syn']=[15]*3;c['hidden']['mem']=[15]*3;c['hidden']['threshold']=[32767]*3
        c['hidden']['bias']=[100,-100,0];c['recurrent_weights']=np.zeros((3,3),int).tolist()
    c=add('mixed_saturation_order',raster=np.full((128,16),15).tolist())
    c['input_weights']=[[0,127,0] for _ in range(16)]
    c['recurrent_weights']=[[0,-128,0],[0,0,0],[0,127,0]]
    c['hidden']['bias']=[32767,0,32767];c['hidden']['threshold']=[1,32767,1]
    c['hidden']['syn']=[15]*3
    c=add('bias_only',raster=np.zeros_like(raster).tolist())
    c['hidden']['bias']=[7,-7,3];c['output']['bias']=[2,-2]
    rng=np.random.default_rng(20261013)
    c=add('seeded_1024',raster=rng.integers(0,4,(1024,2)).tolist())
    c['hidden']['syn']=[0,1,15];c['hidden']['mem']=[1,15,2]
    c=add('max_input_output_dimensions',raster=rng.integers(0,3,(128,16)).tolist())
    c['input_weights']=rng.integers(-8,9,(16,16)).tolist()
    c['recurrent_weights']=rng.integers(-2,3,(16,16)).tolist()
    c['readout_weights']=rng.integers(-8,9,(16,8)).tolist()
    c['hidden']=dict(syn=[2]*16,mem=[3]*16,threshold=[20]*16,bias=[0]*16)
    c['output']=dict(syn=[2]*8,mem=[2]*8,threshold=[20]*8,bias=[0]*8)
    return result


def build(c):
    b.start_scope();b.set_device('runtime');b.prefs.codegen.target='numpy'
    b.prefs.codegen.loop_invariant_optimisations=False
    clock=b.Clock(dt=b.ms,name='clock');raster=np.array(c['raster'],dtype=float)
    table=b.TimedArray(raster,dt=b.ms)
    source=b.NeuronGroup(raster.shape[1],'q:1',threshold='q>0',clock=clock,name='input_spikes',
                         namespace={'stimulus':table})
    source.run_regularly('q=stimulus(t,i)',when='start',order=0,name='input_update')
    source.thresholder['spike'].when='start';source.thresholder['spike'].order=1
    objects=[source];groups=[]
    for role,cap in [('hidden',31),('output',1)]:
        params=c[role];n=len(params['threshold'])
        g=b.NeuronGroup(n,'I:1\nv:1\nq:1\nsyn_div:1 (constant)\nmem_div:1 (constant)\nbias:1 (constant)\nthreshold:1 (constant)\nspike_cap:1 (constant)',
            threshold='q>0',reset='v=v-q*threshold',clock=clock,name=role)
        g.syn_div=2**np.array(params['syn']);g.mem_div=2**np.array(params['mem'])
        g.bias=params['bias'];g.threshold=params['threshold'];g.spike_cap=cap
        g.run_regularly(UPDATE,when='groups',name=role+'_update')
        objects += [g,b.SpikeMonitor(g,name=role+'_spikes'),
                    b.StateMonitor(g,['I','v','q'],record=True,when='end',name=role+'_state')]
        groups.append(g)
    for name,pre,post,key in [('a_input',source,groups[0],'input_weights'),
                              ('b_recurrent',groups[0],groups[0],'recurrent_weights'),
                              ('c_readout',groups[0],groups[1],'readout_weights')]:
        weights=np.array(c[key]);i,j=np.nonzero(weights)
        if not len(i):continue
        s=b.Synapses(pre,post,'w:1 (constant)',on_pre='I_post=clip(I_post+w*q_pre,-32768,32767)',
                     clock=clock,name=name)
        s.connect(i=i,j=j);s.w=weights[i,j];objects.append(s)
    net=b.Network(*objects);net.schedule=['start','synapses','groups','thresholds','resets','end']
    return net


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('output','runner','sdk-python'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--suite',choices=('baseline','coverage'),default='baseline')
    p.add_argument('--chunk-steps',type=int)
    a=p.parse_args()
    if a.chunk_steps is not None and a.chunk_steps<1:p.error('--chunk-steps must be positive')
    a.output=a.output.absolute();a.output.mkdir(parents=True,exist_ok=False)
    bundles=[]
    networks=cases();coverage=None
    if a.suite=='coverage':
        from coverage_cases import cases as coverage_cases,parameter_coverage
        extra=coverage_cases();networks.update(extra);coverage=parameter_coverage(extra)
    for name,c in networks.items():
        root=a.output/name;root.mkdir();source=root/'source.json'
        model=export_network(build(c),len(c['raster'])*b.ms,source)
        bundle=export_ir(source,root/'converted',runner=a.runner,target='xylosim',profile='xylo-audio2-temporal-v1')
        subprocess.run([str(a.runner.resolve()),str(source),str(root/'atlas')],check=True,capture_output=True,text=True)
        results=load_results(model,root/'atlas');values={}
        for role in ('hidden','output'):
            pop=results['populations'][bundle['plan']['roles'][role]]
            for key in ('I','v','q'):values[role+'_'+key]=pop['trace'][key]
            values[role+'_events']=np.column_stack((pop['spike_ticks'],pop['indices']))
        np.savez_compressed(root/'atlas.npz',**values)
        bundles.append(root/'converted/bundle.json');print(name,'Atlas complete',flush=True)
    env={**os.environ,'PYTHONPATH':'','PYTHONNOUSERSITE':'1'}
    runtime=Path(__file__).resolve().parents[3]/'python/brian2_rust/neuromorphic_runtime.py'
    run=subprocess.run([str(a.sdk_python.absolute()),str(Path(__file__).with_name('sdk_worker.py')),
                        '--runtime',str(runtime),
                        *(['--chunk-steps',str(a.chunk_steps)] if a.chunk_steps is not None else []),
                        '--bundles',*map(str,bundles)],env=env,cwd=a.output,
                        capture_output=True,text=True,timeout=300)
    (a.output/'sdk.log').write_text(run.stdout+run.stderr)
    if run.returncode:raise RuntimeError(run.stderr[-5000:])
    reports=[]
    for path in bundles:
        root=path.parent.parent;plan=json.loads(path.read_text())['plan']
        atlas=np.load(root/'atlas.npz');sdk=np.load(path.parent/'sdk.npz')
        comparison={}
        for role in ('hidden','output'):
            for key in ('I','v','q'):
                name=role+'_'+key;np.testing.assert_array_equal(atlas[name],sdk[name],err_msg=str(root))
                comparison[name]=int(atlas[name].size)
            events=np.column_stack(np.nonzero(sdk[role+'_q']))
            np.testing.assert_array_equal(atlas[role+'_events'],events)
        execution=json.loads((path.parent/'execution.json').read_text())
        assert execution['official_configuration_valid'] and execution['bundle_sha256']==sha(path)
        assert execution['result_sha256']==sha(path.parent/'sdk.npz')
        if a.chunk_steps is not None:
            assert execution['chunked']['result_sha256']==sha(path.parent/'sdk-chunked.npz')
            assert execution['chunked']['differences']==0
        reports.append(dict(case=root.name,steps=plan['steps'],sizes=plan['sizes'],compared_values=comparison,
            hidden_spikes=int(sdk['hidden_q'].sum()),output_spikes=int(sdk['output_q'].sum()),
            max_hidden_spikes_per_tick=int(sdk['hidden_q'].max()),differences=0,
            source_model_sha256=plan['source_model_sha256'],atlas_results_sha256=sha(root/'atlas.npz'),sdk=execution))
    def observed(name):return np.load(a.output/name/'converted/sdk.npz')
    assert observed('silent')['hidden_q'].sum()==observed('silent')['output_q'].sum()==0
    assert observed('event_at_end')['hidden_q'][-1].sum()>0 and observed('event_at_end')['output_q'].sum()==0
    assert observed('threshold_equality')['hidden_q'][0,0]==1
    assert observed('multi_spike_caps')['hidden_q'].max()==31 and observed('multi_spike_caps')['output_q'].max()==1
    assert observed('positive_saturation')['hidden_I'].max()==32766
    assert observed('negative_saturation')['hidden_v'].min()==-32768
    assert observed('minimum_decay')['hidden_I'].min()==-1 and observed('minimum_decay')['hidden_I'].max()==1
    if a.suite=='coverage':
        boundary=observed('chunk_boundary_impulse')
        assert boundary['hidden_q'][6,0]>0 and boundary['output_q'][7,0]>0
        assert boundary['hidden_q'][14:,0].sum()>0
    report=dict(schema='atlas.xylo.temporal-audit.v1',passed=True,cases=reports,
                native_atlas_runner_sha256=sha(a.runner),same_tick_comparison=True,
                official_configuration_validation=True,nonvacuous_boundary_checks=True,physical_hardware=False,hardware_mapping_tested=False,
                classification_accuracy_tested=False,suite=a.suite,parameter_coverage=coverage)
    (a.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(cases=len(reports),state_values=sum(sum(r['compared_values'].values()) for r in reports),
                         hidden_spikes=sum(r['hidden_spikes'] for r in reports),output_spikes=sum(r['output_spikes'] for r in reports),differences=0)))


if __name__=='__main__':main()
