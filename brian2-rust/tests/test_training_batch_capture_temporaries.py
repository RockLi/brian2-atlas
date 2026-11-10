"""Materialised caller temporaries keep borrowed-array and scalar lifetimes."""
import copy
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_native_training import RUNNER
from test_training_capture_callbacks import add_constant_selected
import test_training_masked_multistatement_captures as base

CODES={
 'value':'tmp=f(h);v_post+=gain*tmp',
 'borrow':'tmp=f(h);borrow=tmp;tmp*=.8;v_post+=gain*borrow;h=borrow+.1*h',
 'rebind':'tmp=f(h);borrow=tmp;tmp=tmp+.01;v_post+=gain*borrow;h=tmp+.1*h',
 'scalar':'tmp=f(.03);borrow=tmp;tmp*=.8;u_post+=gain*borrow;h=tmp+.1*h',
 'time':'tmp=f(h+.01*t/dt);v_post+=gain*tmp;h=tmp+.1*h',
 'random_borrow':'tmp=f(h+.003*randn()+.004*rand());borrow=tmp;tmp*=.8;v_post+=gain*borrow;h=borrow+.1*h',
 'random_rebind':'tmp=f(h+.003*randn()+.004*rand());borrow=tmp;tmp=tmp+.01;v_post+=gain*borrow;h=tmp+.1*h',
}


def model(backend,ranks,mode,kind,alias):
    old=base.add_selected;base.add_selected=add_constant_selected;base.CODES['caller_tmp']=CODES[kind]
    try:return base.model(backend,ranks,mode,'caller_tmp',alias)
    finally:base.add_selected=old;del base.CODES['caller_tmp']


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('kind',list(CODES))
@pytest.mark.parametrize('alias',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
def test_batch_caller_temporary_original_restore(engine,mode,kind,alias,ranks,tmp_path,monkeypatch):
    mpi(ranks);net,g,syn,dt,bundle,x=model(engine,ranks,mode,kind,alias)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights);t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    random=kind.startswith('random_');calls=[]
    for tick in range(5):
        if random:
            from test_training_event_noise import draw
            domain=bundle.provenance['scheduled_noise_domains'][syn.pre.name]
            def sampler(draw_kind,stream):
                def sample(*shape):
                    raw=np.flatnonzero(x[0,tick-1]) if tick else np.array([],int)
                    assert shape==(len(raw),),(tick,draw_kind,shape,raw)
                    calls.append((tick,draw_kind,len(raw)))
                    return np.array([draw(draw_kind,p['seed'],9,0,domain,int(edge),tick-1,None,stream) for edge in raw])
                return sample
            monkeypatch.setattr(np.random,'randn',sampler('randn',0));monkeypatch.setattr(np.random,'rand',sampler('rand',1))
        options={'initial':'carry'} if tick else {'noise_sequence':9} if random else {}
        out=t.step(x[:,tick:tick+1],[0],**options);net.run(dt,namespace={})
        for obj,key,names in [(g,'neuron_state_layout',['v','u']),(syn,'dynamic_state_layout',['h'])]:
            for name in names:
                cells=bundle.provenance[key][obj.name][name];np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],obj.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        path=tmp_path/str(tick);t.store(path);t=NativeLIFTrainer(t.plan,runner=RUNNER);t.restore(path)
    if random:
        assert len(calls)==6
        assert t.next_noise_sequence==10


def reference(data,mode,kind,alias,weights,window,initial=None,anchors=None):
    import ast
    _,g,syn,_,bundle,_=data;p=bundle.plan
    z=np.array(bundle.initial_state if initial is None else initial,float)
    if initial is None:
        for cell,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[cell]=weights[ref[0]][ref[1]]
    layout=bundle.provenance['neuron_state_layout'][g.name];v=layout['v'];u=layout['u'];counter=layout['__refractory_ticks'];h=bundle.provenance['dynamic_state_layout'][syn.name]['h']
    activity=bundle.provenance['refractory_activity_layout'][g.name]
    gain=next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==syn.name and row['variables']==['gain'])
    spec=bundle.provenance['event_callback_caller_locals'][syn.pre.name];trace=spec['trace'];fields={row['key'][1]:row['fields'] for row in spec['rows'] if row['key'][0]=='new'}
    paths=bundle.provenance['event_callback_stage_groups'][syn.pre.name][1:];posts=np.asarray(syn.j[:],int);capture=v if alias else u
    before=[];margins=[];hard_rows=[];free_rows=[];raw_rows=[];spikes=[]
    for tick in range(5):
        if anchors is not None and window and tick and tick%window==0:z=anchors['before'][tick].copy()
        before.append(z.copy());free=z[counter]<.5;z[counter]=np.maximum(z[counter]-1,0.)
        if anchors is not None:free=anchors['free'][tick]
        margin=z[v]-.5;hard=(margin>0)&free;event=hard.astype(float)
        if anchors is not None:
            hard=anchors['hard'][tick];old=anchors['margins'][tick]
            event=hard.astype(float)+free*p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
        margins.append(margin.copy());hard_rows.append(hard.copy());free_rows.append(free.copy());spikes.append(event.copy());flag=free&~hard;z[activity]=flag
        noise=bundle.provenance.get('event_callback_random_fields',{}).get(syn.pre.name)
        noise_fields={}
        if noise is not None:
            from test_training_event_noise import draw
            noise_fields={row['key'][1]:row['fields'] for row in noise['rows'] if row['key'][0]=='new'}
            domain=bundle.provenance['scheduled_noise_domains'][syn.pre.name]
            raw=[row['edge'] for row in bundle.provenance['delay_queues'][paths[0]]['new'] if z[row['states'][0]]>.5]
            if anchors is not None:raw=anchors['raw'][tick][0]
            for edge in raw:
                for sample in noise['draws']:
                    z[noise_fields[edge][sample['name']]]=draw(sample['kind'],p['seed'],9,0,domain,edge,tick-1,None,sample['stream'])
        seeded=set();raw_tick=[]
        for stage_index,path in enumerate(paths):
            stage=trace['stages'][stage_index] if stage_index<len(trace['stages']) else None
            if stage is not None:
                tokens=set(stage['inputs'].values())
                if stage_index==0 and mode=='array':tokens.update(trace['initial'].values())
                for token in sorted(tokens):
                    source=trace['tokens'][token]['source']
                    if source is None or token in seeded:continue
                    seeded.add(token)
                    for edge,row in fields.items():
                        value=z[h[edge]] if source=='h' else weights[gain][edge] if source=='gain' else z[v[posts[edge]]] if source=='v_post' else z[u[posts[edge]]] if source=='u_post' else None
                        if source=='t':value=tick*float(data[3])
                        if source=='dt':value=float(data[3])
                        if source in noise_fields.get(edge,{}):value=z[noise_fields[edge][source]]
                        assert value is not None,source;z[row[token]]=value
            for row in bundle.provenance['event_callback_snapshots'].get(path,[]):z[row['cache']]=z[row['source']]
            old=z.copy();routes=bundle.provenance['delay_queues'][path]['new'];raw=[row['edge'] for row in routes if old[row['states'][0]]>.5]
            if anchors is not None:raw=anchors['raw'][tick][stage_index]
            raw_tick.append(list(raw))
            statement=ast.parse(stage['code']).body[0] if stage is not None else None
            if statement is not None and any(isinstance(node,ast.Call) for node in ast.walk(statement)) and raw:z[capture]+=.003
            for row in routes:
                edge=row['edge'];post=posts[edge];amplitude=old[row['states'][0]]
                if stage is None:
                    for name in ['h','v_post' if kind!='scalar' else 'u_post']:
                        if name not in trace['final']:continue
                        target=h[edge] if name=='h' else v[post] if name=='v_post' else u[post]
                        if name=='h' and kind=='value':continue
                        z[target]=old[target]+amplitude*(old[fields[edge][trace['final'][name]]]-old[target])
                    continue
                values={name:old[fields[edge][token]] for name,token in stage['inputs'].items()}
                if mode=='vectorised' and stage['target'] in ('v_post','u_post'):values[stage['target']]=z[v[post] if stage['target']=='v_post' else u[post]]
                def evaluate(node):
                    if isinstance(node,ast.Name):return values[node.id]
                    if isinstance(node,ast.Constant):return node.value
                    if isinstance(node,ast.BinOp):
                        a=evaluate(node.left);b=evaluate(node.right)
                        return a+b if isinstance(node.op,ast.Add) else a*b if isinstance(node.op,ast.Mult) else a/b if isinstance(node.op,ast.Div) else (_ for _ in ()).throw(AssertionError(ast.dump(node)))
                    if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='f':return .7*evaluate(node.args[0])
                    raise AssertionError(ast.dump(node))
                rhs=evaluate(statement.value);target_name=stage['target']
                if isinstance(statement,ast.AugAssign):rhs=values[target_name]+rhs if isinstance(statement.op,ast.Add) else values[target_name]*rhs
                if target_name=='v_post' and not flag[post]:rhs=values[target_name]
                for name,token in stage['outputs'].items():
                    assert name==target_name,'unexpected implicit callback output'
                    target=(h[edge] if name=='h' else v[post] if name=='v_post' else u[post] if name=='u_post' else fields[edge][token]) if mode=='vectorised' else fields[edge][token]
                    z[target]+=amplitude*(rhs-z[target])
        raw_rows.append(raw_tick)
        for queue in bundle.provenance['delay_queues'].values():
            for row in queue['new']:
                cells=row['states'];edge=row['edge'];z[cells[:-1]]=z[cells[1:]];z[cells[-1]]=float(tick==0 or tick==1 and edge==0 or tick==2 and edge==len(posts)-1)
        z[v]-=.5*event;z[counter]=np.where(hard,1.,z[counter])
    logits=np.array(spikes).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,dict(before=before,margins=margins,hard=hard_rows,free=free_rows,raw=raw_rows)


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('kind',list(CODES))
@pytest.mark.parametrize('alias',[False,True])
@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('ranks',[None,2])
def test_batch_caller_temporary_all_vjps(engine,mode,kind,alias,window,ranks):
    mpi(ranks);data=model(engine,ranks,mode,kind,alias);_,g,syn,_,bundle,x=data;p=copy.deepcopy(bundle.plan);p['tbptt_window']=window
    options={'noise_sequence':9} if kind.startswith('random_') else {}
    out=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER).gradients(x,[0],**options);loss,z,anchors=reference(data,mode,kind,alias,bundle.weights,window)
    cells=[*bundle.provenance['neuron_state_layout'][g.name]['v'],*bundle.provenance['neuron_state_layout'][g.name]['u'],*bundle.provenance['dynamic_state_layout'][syn.name]['h']]
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],z[cells],rtol=8e-5,atol=8e-6);assert out['loss']==pytest.approx(loss,abs=8e-6)
    for bank,row in enumerate(bundle.weights):
        for index in range(len(row)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
            fd=(reference(data,mode,kind,alias,hi,window,anchors=anchors)[0]-reference(data,mode,kind,alias,lo,window,anchors=anchors)[0])/2e-6
            assert out['gradients'][bank][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),(bank,index)
    for index in range(len(z)):
        if p['dynamic']['detached'][index] or index in p['dynamic']['integer_states']:assert out['initial_state_gradients'][0][index]==0;continue
        hi=np.array(bundle.initial_state);lo=hi.copy();hi[index]+=1e-6;lo[index]-=1e-6
        fd=(reference(data,mode,kind,alias,bundle.weights,window,hi,anchors)[0]-reference(data,mode,kind,alias,bundle.weights,window,lo,anchors)[0])/2e-6
        assert out['initial_state_gradients'][0][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),index
