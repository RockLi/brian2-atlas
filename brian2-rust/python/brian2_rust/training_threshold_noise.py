"""Shared threshold samples and dependency-only scalar Poisson rate programs."""
import copy


def dependencies(node):
    fields=['arg','left','right','low','high','condition','yes','no','rate','slope','scale']
    if node['op'] in ('timed_parameter','parameter_gather','integer_parameter_gather'):
        fields.append('index')
        if node['op']=='timed_parameter':fields.append('time')
    return [field for field in fields if field in node]


def rate_program(program,root):
    """Retain every dependency, including lazy branches, in SSA source order."""
    needed=set()
    def visit(index):
        if index in needed:return
        needed.add(index)
        for field in dependencies(program[index]):visit(program[index][field])
    visit(root);remap={};nodes=[]
    for old in sorted(needed):
        node=copy.deepcopy(program[old])
        for field in dependencies(node):node[field]=remap[node[field]]
        remap[old]=len(nodes);nodes.append(node)
    return nodes


def program_signature(program):
    """Compare expression structure independently of SSA allocation/sharing."""
    memo={}
    def visit(index):
        if index not in memo:
            node=program[index];fields=dependencies(node)
            memo[index]=({key:value for key,value in node.items() if key not in fields},
                         tuple((field,visit(node[field])) for field in fields))
        return memo[index]
    return visit(len(program)-1)


def shared_threshold_records(programs,records):
    result=[]
    for record in records:
        if not record.get('scalar',False):continue
        row=dict(record)
        if row['kind']=='poisson':
            candidates=[rate_program(program,node['rate']) for program in programs for node in program
                        if node['op']=='poisson' and node['stream']==row['stream']]
            if not candidates or any(program_signature(candidate)!=program_signature(candidates[0]) for candidate in candidates[1:]):
                raise ValueError('shared threshold Poisson requires one consistent scalar rate')
            row['rate_program']=candidates[0]
            if any(node['op'] in ('noise','uniform_noise','poisson') and node['stream']>=row['stream'] for node in row['rate_program']):
                raise ValueError('shared threshold rate has a forward random dependency')
        result.append(row)
    return result


def bind_shared_threshold_noise(program,slots):
    nodes=copy.deepcopy(program)
    for index,node in enumerate(nodes):
        if node['op'] in ('noise','uniform_noise','poisson') and node['stream'] in slots:
            nodes[index]=dict(op='integer_state' if node['op']=='poisson' else 'state',index=slots[node['stream']])
    return nodes
