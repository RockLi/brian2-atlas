"""Preserve local array lifetime and statement order around masked callbacks."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_training_capture_callbacks import add_selected,scale_selected_by_capture
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_native_training import RUNNER

CODES={'before':'h*=.8;v_post+=gain*f(h)',
       'twice':'v_post+=gain*f(h);v_post+=gain*f(h)',
       'after':'v_post+=gain*f(h);h*=.8'}


def model(backend,ranks,mode,kind,alias,*,self_argument=False):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    size=3 if mode=='vectorised' else 2
    source=b.SpikeGeneratorGroup(size,[*range(size),0,size-1],np.array([0]*size+[1,2])*dt,dt=dt)
    hidden=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt)
    g=b.NeuronGroup(2,'dv/dt=0/second:1 (unless refractory)\nu:1',threshold='v>.5',reset='v-=.5',refractory=.4*b.ms,method='euler',dt=dt)
    g.v=[.217,.719];g.u=[.017,.031];g.lastspike=np.array([-1.,-1.])*b.second
    syn=b.Synapses(source,g,'dh/dt=0/second:1 (clock-driven)\ngain:1 (constant)',on_pre=CODES[kind].replace('f(h)','f(v_post)') if self_argument else CODES[kind],method='euler',dt=dt)
    syn.connect(i=list(range(size)),j=[0,0,1] if size==3 else [0,1]);syn.h=[.033,.053,.047][:size];syn.gain=[.11,.19,.17][:size];syn.delay=dt
    if self_argument:g.v=[.217,.319]
    if mode=='array':
        assert len(set(np.asarray(syn.j[:],int)))==len(syn);syn.variables['_postsynaptic_idx'].unique=True
    syn.namespace['f']=b.Function((scale_selected_by_capture if self_argument else add_selected)(g.variables['v' if alias else 'u'].get_value()),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    net=b.Network(source,hidden,g,syn)
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=backend,mpi_ranks=ranks,detach_reset=False,trainable_synapse_parameters={syn.name:['h','gain']})
    for path in bundle.provenance['event_callback_stage_groups'][syn.pre.name][1:]:assert bundle.provenance['event_callback_modes'][path]['mode']==mode
    x=np.zeros((1,5,size));x[0,0,:]=1;x[0,1,0]=1;x[0,2,-1]=1
    return net,g,syn,dt,bundle,x


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('kind',list(CODES))
@pytest.mark.parametrize('alias',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
def test_masked_multistatement_original_restore(engine,mode,kind,alias,ranks,tmp_path):
    mpi(ranks);net,g,syn,dt,bundle,x=model(engine,ranks,mode,kind,alias)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights);t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    for tick in range(5):
        out=t.step(x[:,tick:tick+1],[0],initial='carry' if tick else None);net.run(dt,namespace={})
        for obj,key,names in [(g,'neuron_state_layout',['v','u']),(syn,'dynamic_state_layout',['h'])]:
            for name in names:
                cells=bundle.provenance[key][obj.name][name];np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],obj.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        path=tmp_path/str(tick);t.store(path);t=NativeLIFTrainer(t.plan,runner=RUNNER);t.restore(path)


def reference(data,kind,alias,weights,window,initial=None,anchors=None):
    _,g,syn,_,bundle,_=data;p=bundle.plan
    z=np.array(bundle.initial_state if initial is None else initial,float)
    if initial is None:
        for cell,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[cell]=weights[ref[0]][ref[1]]
    layout=bundle.provenance['neuron_state_layout'][g.name];v=layout['v'];u=layout['u'];counter=layout['__refractory_ticks']
    activity=bundle.provenance['refractory_activity_layout'][g.name];h=bundle.provenance['dynamic_state_layout'][syn.name]['h']
    gain=next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==syn.name and row['variables']==['gain'])
    paths=bundle.provenance['event_callback_stage_groups'][syn.pre.name][1:]
    roles={'before':['h','f'],'after':['f','h'],'twice':['f','f']}[kind]
    local_spec=bundle.provenance.get('event_callback_array_locals',{}).get(syn.pre.name)
    local_rows={} if local_spec is None else {row['key'][1]:row['fields'] for row in local_spec['rows'] if row['key'][0]=='new'}
    if local_spec is not None:roles=[*roles,'publish']
    assert len(paths)==len(roles)
    posts=np.asarray(syn.j[:],int);capture=v if alias else u
    before=[];margins=[];hard_rows=[];free_rows=[];selections=[];spikes=[]
    for tick in range(5):
        if anchors is not None and window and tick and tick%window==0:z=anchors['before'][tick].copy()
        before.append(z.copy());free=z[counter]<.5;z[counter]=np.maximum(z[counter]-1,0.)
        if anchors is not None:free=anchors['free'][tick]
        margin=z[v]-.5;hard=(margin>0)&free;event=hard.astype(float)
        if anchors is not None:
            hard=anchors['hard'][tick];old=anchors['margins'][tick]
            event=hard.astype(float)+free*p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
        margins.append(margin.copy());hard_rows.append(hard.copy());free_rows.append(free.copy());spikes.append(event.copy());flag=free&~hard;z[activity]=flag
        selected_tick=[]
        for fields in local_rows.values():
            for row in fields.values():z[row['local']]=z[row['source']]
        for stage,(path,role) in enumerate(zip(paths,roles)):
            for row in bundle.provenance['event_callback_snapshots'].get(path,[]):z[row['cache']]=z[row['source']]
            old=z.copy();rows=bundle.provenance['delay_queues'][path]['new']
            raw=[row['edge'] for row in rows if old[row['states'][0]]>.5]
            if anchors is not None:raw=anchors['selections'][tick][stage]
            selected_tick.append(list(raw));active=[edge for edge in raw if flag[posts[edge]]]
            if role=='f' and active:
                incoming=np.array([old[local_rows[edge]['h']['local'] if local_rows else h[edge]] for edge in active]);z[capture]+=incoming[0] if len(incoming)==1 else incoming
            for index,row in enumerate(rows):
                edge=row['edge'];amplitude=old[row['states'][0]]
                local_h=local_rows[edge]['h']['local'] if local_rows else h[edge]
                local_v=local_rows[edge]['v_post']['local'] if local_rows else v[posts[edge]]
                if role=='publish':
                    for name in local_spec['write_order']:
                        entry=local_rows[edge][name];target=entry['source']
                        z[target]=old[target]+amplitude*(old[entry['local']]-old[target])
                elif role=='h':z[local_h]=old[local_h]+amplitude*(.8*old[local_h]-old[local_h])
                else:
                    ordinal=sum(other['edge'] in active for other in rows[:index]);selected=active[0] if len(active)==1 else active[ordinal] if ordinal<len(active) else edge
                    selected_h=local_rows[selected]['h']['local'] if local_rows else h[selected]
                    if flag[posts[edge]]:z[local_v]+=amplitude*weights[gain][edge]*.7*old[selected_h]
        selections.append(selected_tick)
        for queue in bundle.provenance['delay_queues'].values():
            for row in queue['new']:
                cells=row['states'];edge=row['edge'];z[cells[:-1]]=z[cells[1:]];z[cells[-1]]=float(tick==0 or tick==1 and edge==0 or tick==2 and edge==len(posts)-1)
        z[v]-=.5*event;z[counter]=np.where(hard,1.,z[counter])
    logits=np.array(spikes).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,dict(before=before,margins=margins,hard=hard_rows,free=free_rows,selections=selections)


def check_all_vjps(engine,kind,alias,window,ranks,mode):
    mpi(ranks);data=model(engine,ranks,mode,kind,alias);_,g,syn,_,bundle,x=data
    p=copy.deepcopy(bundle.plan);p['tbptt_window']=window
    out=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER).gradients(x,[0]);loss,z,anchors=reference(data,kind,alias,bundle.weights,window)
    cells=[*bundle.provenance['neuron_state_layout'][g.name]['v'],*bundle.provenance['neuron_state_layout'][g.name]['u'],*bundle.provenance['dynamic_state_layout'][syn.name]['h']]
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],z[cells],rtol=8e-5,atol=8e-6);assert out['loss']==pytest.approx(loss,abs=8e-6)
    for bank,row in enumerate(bundle.weights):
        for index in range(len(row)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
            fd=(reference(data,kind,alias,hi,window,anchors=anchors)[0]-reference(data,kind,alias,lo,window,anchors=anchors)[0])/2e-6
            assert out['gradients'][bank][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),(bank,index)
    for index in range(len(z)):
        if p['dynamic']['detached'][index] or index in p['dynamic']['integer_states']:assert out['initial_state_gradients'][0][index]==0;continue
        hi=np.array(bundle.initial_state);lo=hi.copy();hi[index]+=1e-6;lo[index]-=1e-6
        fd=(reference(data,kind,alias,bundle.weights,window,hi,anchors)[0]-reference(data,kind,alias,bundle.weights,window,lo,anchors)[0])/2e-6
        assert out['initial_state_gradients'][0][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),index


@pytest.mark.parametrize('kind',list(CODES))
@pytest.mark.parametrize('alias',[False,True])
@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('ranks',[None,2])
def test_masked_multistatement_vector_all_vjps(engine,kind,alias,window,ranks):
    check_all_vjps(engine,kind,alias,window,ranks,'vectorised')


@pytest.mark.parametrize('kind',list(CODES))
@pytest.mark.parametrize('alias',[False,True])
@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('ranks',[None,2])
def test_masked_multistatement_array_all_vjps(engine,kind,alias,window,ranks):
    check_all_vjps(engine,kind,alias,window,ranks,'array')


@pytest.mark.parametrize('kind',['before','twice','after'])
@pytest.mark.parametrize('ranks',[None,2])
def test_array_nonlinear_local_lifetime(engine,kind,ranks,tmp_path):
    mpi(ranks);net,g,syn,dt,bundle,x=model(engine,ranks,'array',kind,True,self_argument=True)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights);t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    for tick in range(2):
        out=t.step(x[:,tick:tick+1],[0],initial='carry' if tick else None);net.run(dt,namespace={})
        cells=bundle.provenance['neuron_state_layout'][g.name]['v']
        np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],g.v[:],rtol=8e-5,atol=8e-6)
        path=tmp_path/str(tick);t.store(path);t=NativeLIFTrainer(t.plan,runner=RUNNER);t.restore(path)
