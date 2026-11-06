"""Expose ordered Poisson array sites as caller-local statements."""
import ast
import copy


def stage_batch_poisson(code,*,statement_groups=None,input_statement_groups=None,poisson_names=('poisson',)):
    """Keep arguments and preceding operands alive before each array draw.

    Assignment retains an array reference. This is intentional: a later
    callback may mutate that same object while evaluating the right operand.
    The caller-lifetime planner decides whether a value needs a copied lane or
    a borrowed whole-vector binding; this pass never executes model functions.
    """
    tree=ast.parse(code)
    def contains(node):
        return any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in poisson_names for n in ast.walk(node))
    if not contains(tree):
        if statement_groups is not None and input_statement_groups is not None:statement_groups.extend(input_statement_groups)
        return code
    names={n.id for n in ast.walk(tree) if isinstance(n,ast.Name)}
    if any(name.startswith(('_b2poisson_local_','_b2poisson_anchor')) for name in names):raise ValueError('reserved Poisson local name collision')
    output=[];serial=0;group=0
    def save(value):
        nonlocal serial
        name='_b2poisson_local_'+str(serial);serial+=1
        if serial>256:raise ValueError('Poisson caller temporary budget exceeded')
        output.append(ast.Assign(targets=[ast.Name(id=name,ctx=ast.Store())],value=value))
        if statement_groups is not None:statement_groups.append(group)
        return ast.Name(id=name,ctx=ast.Load())
    def expression(node):
        node=copy.deepcopy(node)
        if isinstance(node,ast.BinOp):
            left=expression(node.left)
            if contains(node.right):left=save(left)
            node.left=left;node.right=expression(node.right)
        elif isinstance(node,ast.Subscript):
            value=expression(node.value)
            if contains(node.slice):value=save(value)
            node.value=value;node.slice=expression(node.slice)
        elif isinstance(node,ast.UnaryOp):node.operand=expression(node.operand)
        elif isinstance(node,ast.Compare):
            if len(node.ops)!=1:raise ValueError('Poisson caller chained comparisons are unsupported')
            left=expression(node.left)
            if contains(node.comparators[0]):left=save(left)
            node.left=left;node.comparators=[expression(node.comparators[0])]
        elif isinstance(node,ast.Call):
            if node.keywords:raise ValueError('Poisson caller requires positional calls')
            arguments=[]
            for i,arg in enumerate(node.args):
                value=expression(arg)
                if any(contains(later) for later in node.args[i+1:]):value=save(value)
                arguments.append(value)
            node.args=arguments
            if isinstance(node.func,ast.Name) and node.func.id in poisson_names:
                if len(arguments)!=1:raise ValueError('invalid random function arguments')
                if any(isinstance(n,ast.Call) for n in ast.walk(arguments[0])):node.args=[save(arguments[0])]
                return save(node)
        elif not isinstance(node,(ast.Name,ast.Constant)):
            raise ValueError('unsupported Poisson caller expression: '+ast.dump(node))
        return node
    if input_statement_groups is not None and len(input_statement_groups)!=len(tree.body):raise ValueError('invalid callback statement groups')
    for number,statement in enumerate(tree.body):
        group=number if input_statement_groups is None else input_statement_groups[number]
        if not isinstance(statement,(ast.Assign,ast.AugAssign)):raise ValueError('Poisson caller requires assignments')
        staged=copy.deepcopy(statement);staged.value=expression(statement.value);output.append(staged)
        if statement_groups is not None:statement_groups.append(group)
    if len(output)>256:raise ValueError('Poisson caller statement budget exceeded')
    return ast.unparse(ast.fix_missing_locations(ast.Module(body=output,type_ignores=[])))
