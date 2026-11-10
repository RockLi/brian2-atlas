"""Lift authenticated callback random bodies into the actual caller scope."""
import ast
import copy
from dataclasses import replace


def lift_callback_noise(code,functions,*,scalar_call=False):
    tree=ast.parse(code);occupied={n.id for n in ast.walk(tree) if isinstance(n,ast.Name)}
    if any(n.startswith('_b2callback_') or n in ('_b2_scalar_value','_b2_shape_noise','_b2_scalar_noise','_b2_scalar_rate','_b2_size_rate','_b2_rate_size_element','_b2_full_slice','_b2_fixed_shape') for n in occupied):raise ValueError('reserved callback local name collision')
    output=[];groups=[];getters={};sites={};serial=0;group=0
    def fresh():
        nonlocal serial
        serial+=1
        if serial>256:raise ValueError('callback local budget exceeded')
        return '_b2callback_'+str(serial)
    def emit(statement):output.append(statement);groups.append(group)
    def save(value):
        name=fresh();emit(ast.Assign(targets=[ast.Name(id=name,ctx=ast.Store())],value=value));return ast.Name(id=name,ctx=ast.Load())
    def contains(node):return any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and getattr(functions.get(n.func.id),'random_sites',()) for n in ast.walk(node))
    def expression(node):
        node=copy.deepcopy(node)
        if isinstance(node,ast.BinOp):
            left=expression(node.left)
            if contains(node.right):left=save(left)
            node.left=left;node.right=expression(node.right);return node
        if isinstance(node,ast.Compare):
            left=expression(node.left)
            if any(contains(value) for value in node.comparators):left=save(left)
            node.left=left;comparators=[]
            for index,value in enumerate(node.comparators):
                value=expression(value)
                if any(contains(later) for later in node.comparators[index+1:]):value=save(value)
                comparators.append(value)
            node.comparators=comparators;return node
        if isinstance(node,ast.Call):
            arguments=[]
            for index,arg in enumerate(node.args):
                value=expression(arg)
                if any(contains(later) for later in node.args[index+1:]):value=save(value)
                arguments.append(value)
            node.args=arguments;descriptor=functions.get(node.func.id) if isinstance(node.func,ast.Name) else None
            if not getattr(descriptor,'random_sites',()):return node
            if node.keywords or len(arguments)!=len(descriptor.arguments):raise ValueError('callback random arity mismatch')
            names={name:save(value).id for name,value in zip(descriptor.arguments,arguments)}
            for name,array in descriptor.captured_arrays:
                getter=fresh();getters[getter]=replace(descriptor,arguments=(),body='return '+name,constants=(),scalarize_arguments=(),array_return=False,random_sites=(),random_domains=(),random_argument_orders=())
                names[name]=save(ast.Call(func=ast.Name(id=getter,ctx=ast.Load()),args=[],keywords=[])).id
            constants=dict(descriptor.constants);random={}
            for name,kind in descriptor.random_sites:
                alias=fresh();random[name]=alias;sites[alias]=dict(kind=kind)
            for name,domain in descriptor.random_domains:
                sites[random[name]]['domain']=names[domain] if domain in names else names.setdefault(domain,fresh())
            orders=dict(descriptor.random_argument_orders)
            class Rename(ast.NodeTransformer):
                def __init__(self,returning=False):self.returning=returning
                def visit_Call(self,n):
                    original=n.func.id if isinstance(n.func,ast.Name) else None
                    n=self.generic_visit(n)
                    if scalar_call and original in random and sites[random[original]]['kind']=='poisson' and original not in orders:
                        n.args=[ast.Call(func=ast.Name(id='_b2_scalar_rate',ctx=ast.Load()),args=[n.args[0]],keywords=[])]
                    if original not in orders:return n
                    order=orders[original];values={}
                    for label,value in zip(order,n.args):values[label]=save(value)
                    alias=random[original];kind=sites[alias]['kind']
                    if kind=='poisson':
                        n.args=[ast.Call(func=ast.Name(id='_b2_size_rate',ctx=ast.Load()),args=[values['lam'],values['size']],keywords=[])]
                        return save(n) if self.returning else n
                    sites[alias]['kind']='randn' if kind=='normal' else 'rand'
                    helper=fresh()
                    if scalar_call and 'size' in values:raise ValueError('scalar callback array random sizes require whole-array sampling')
                    shaped_noise=('_b2_scalar_noise(first,second,noise)' if scalar_call else '_b2_shape_noise(first,second,noise)') if 'size' not in values else 'noise'
                    body=('check=second**.5\nreturn first+second*'+shaped_noise if kind=='normal' else 'return first+(second-first)*'+shaped_noise)
                    getters[helper]=replace(descriptor,arguments=('first','second','noise'),body=body,constants=(),scalarize_arguments=(),array_return=False,random_sites=(),random_domains=(),random_argument_orders=())
                    first=values['loc' if kind=='normal' else 'low'];second=values['scale' if kind=='normal' else 'high']
                    domains=[values['size']] if 'size' in values else [first,second]
                    sites[alias]['domain']=domains[0].id if 'size' in values else tuple(value.id for value in domains)
                    noise=save(ast.Call(func=ast.Name(id=alias,ctx=ast.Load()),args=domains,keywords=[]))
                    result=ast.Call(func=ast.Name(id=helper,ctx=ast.Load()),args=[first,second,noise],keywords=[])
                    return save(result) if self.returning and 'size' not in values else result

                def visit_Name(self,n):
                    if n.id in ('_b2_full_slice','_b2_fixed_shape'):return n
                    if n.id in constants:return ast.Constant(constants[n.id])
                    if n.id in random:return ast.Name(id=random[n.id],ctx=n.ctx)
                    if n.id not in names:names[n.id]=fresh()
                    return ast.Name(id=names[n.id],ctx=n.ctx)
            body=ast.parse(descriptor.body)
            for statement in body.body:
                if isinstance(statement,ast.Expr) and isinstance(statement.value,ast.Constant) and isinstance(statement.value.value,str):continue
                if isinstance(statement,ast.Return):
                    if statement.value is None:raise ValueError('callback return requires value')
                    return Rename(returning=True).visit(statement.value)
                if not isinstance(statement,(ast.Assign,ast.AugAssign)):raise ValueError('callback random body requires straight-line assignments')
                emit(Rename().visit(statement))
            raise ValueError('callback random body missing return')
        if isinstance(node,ast.Subscript):
            value=expression(node.value)
            if contains(node.slice):value=save(value)
            node.value=value;node.slice=expression(node.slice)
            return node
        if isinstance(node,ast.UnaryOp):node.operand=expression(node.operand)
        return node
    for group,statement in enumerate(tree.body):
        if not isinstance(statement,(ast.Assign,ast.AugAssign)):raise ValueError('callback caller requires assignments')
        staged=copy.deepcopy(statement);staged.value=expression(statement.value);emit(staged)
    return ast.unparse(ast.fix_missing_locations(ast.Module(body=output,type_ignores=[]))),getters,sites,groups
