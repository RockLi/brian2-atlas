"""Complete physical-column compilation for NumPy neuron blocks."""
import ast
from .training_callback_vector import lower_vector_callback_draws
from .training_regular_whole import compile_whole_regular,cached_poisson_code
from .training_equations import compile_training_equation,NormalNoise,UniformNoise


def compile_whole_neuron_block(code,variables,functions,fields,parameters,*,rows,allocate,stream_offset=0,float_result=None,write_order=(),return_context=False,selected_length=None,selected_detached=False,selected_counterfactual=False,predicate_result=None):
    """Return ordered draw actions and one simultaneous block writeback.

    Each field holds its actual physical columns. Random domains may be wider
    than the caller; caches and closure writes retain every column.
    """
    text,params,records,_,trace=lower_vector_callback_draws(code,functions,variables,
        stream_offset=stream_offset,allow_scalar=True,return_trace=True)
    if float_result is not None:
        token=trace['tokens'][trace['final'][float_result]]
        if token['dtype']!='float':raise ValueError('whole ordered neuron thresholds require a floating margin')
    params.update(parameters);fields=dict(fields);actions=[]
    for record in records:
        if selected_length is not None and not selected_detached and not selected_counterfactual and record['kind']=='poisson' and not record['scalar'] and record['length'] is None:
            raise ValueError('selected Poisson reset requires actual-only checkpoint and cached counterfactual binding')
        count=1 if record['scalar'] else record['length'] if record['length'] is not None else rows
        dtype='integer' if record['kind']=='poisson' else 'float'
        slots=[allocate(dtype) for _ in range(max(count,1))]
        record['cache_name']='__regular_whole_draw_'+str(record['stream'])
        name=record['cache_name'] if record['kind']=='poisson' else record['name']
        fields[name]=dict(values=slots,dtype=dtype,array=not record['scalar'],writable=True)
        if selected_length is not None and not record['scalar'] and record['length'] is None:
            fields[name]['length']=selected_length;record['selected_cache']=name
        if count==0:
            fields[name]['length']=allocate('integer')
            record['empty_cache']=True
        params.pop(record['name'])
        if record['kind']!='poisson':
            noise=NormalNoise(record['stream']) if record['kind']=='randn' else UniformNoise(record['stream'])
            program=compile_training_equation('draw',parameters={'draw':noise})
            for j,cell in enumerate(slots[:count]):actions.append(dict(reads=[cell],writes=[cell],programs=[program],noise_entity=j,noise_streams=record['stream']+1))
    text,prefixes=cached_poisson_code(text,records)
    rate_cell=allocate('float')
    fields['__regular_whole_rate']=dict(values=[rate_cell],dtype='float',array=False,writable=True)
    def compiled(text,j,*,separate=False):return compile_whole_regular(text,fields,params,column=j,write_order=write_order,compact_eager_roots=True,separate_whole_eager=separate,predicate_outputs={} if predicate_result is None else {predicate_result[0]:predicate_result[1:]})
    for record,prefix in prefixes:
        for j,cell in enumerate(fields[record['cache_name']]['values'] if not record.get('empty_cache') else []):
            result=compiled(prefix,j);program=result['programs'][result['names']['__regular_whole_rate']]
            program=[*program,dict(op='poisson',rate=len(program)-1,stream=record['stream'])]
            if len(program)>128:raise ValueError('whole neuron Poisson rate exceeds128 nodes')
            reads=result['reads'];reads.extend([cell] if cell not in reads else [])
            action=dict(reads=reads,writes=[cell],programs=[program],noise_entity=j,noise_streams=record['stream']+1)
            if selected_length is not None and not record['scalar'] and record['length'] is None:
                # NumPy draws only the compact array's actual columns. A hard
                # guard must precede the sampler, so empty/unused columns own
                # neither a cached observation nor its likelihood gradient.
                guard=allocate('boolean')
                actions.append(dict(reads=[selected_length,guard],writes=[guard],programs=[[
                    dict(op='integer_state',index=0),dict(op='integer_constant',value=j),
                    dict(op='integer_compare',left=1,right=0,kind='lt')]]))
                reads.extend([guard] if guard not in reads else [])
                action.update(trigger=dict(external=False,state=True,index=guard),detach_trigger=True)
            actions.append(action)
    outputs={}
    for j in range(max([rows,*[len(field['values']) for field in fields.values() if field['array']]])):
        result=compiled(text,j,separate=True)
        if result['eager_programs']:
            context=list(result['reads']);context.extend([rate_cell] if rate_cell not in context else [])
            if len(context)>64:raise ValueError('whole neuron eager checks exceed64 context slots')
            for eager in result['eager_programs']:
                actions.append(dict(reads=context,writes=[rate_cell],programs=[eager]))
        for slot,program in result['programs'].items():outputs[result['reads'][slot]]=(result['reads'],program)
    if outputs:
        reads=list(next(iter(outputs.values()))[0])
        if not all(context==reads for context,_ in outputs.values()):raise ValueError('whole neuron columns require one common input snapshot')
        writes=list(outputs);reads.extend(cell for cell in writes if cell not in reads)
        if len(reads)>64:raise ValueError('whole neuron writeback exceeds64 context slots')
        actions.append(dict(reads=reads,writes=writes,programs=[program for _,program in outputs.values()]))
    return (actions,records,text,dict(fields=fields,parameters=params,poisson_prefixes=prefixes,selected_cache_fields=[row['selected_cache'] for row in records if 'selected_cache' in row])) if return_context else (actions,records,text)
