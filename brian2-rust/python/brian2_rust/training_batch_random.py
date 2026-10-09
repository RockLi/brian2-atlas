"""Cache each event's random arrays before evaluating a whole capture callback."""
import ast
from brian2.core.functions import DEFAULT_FUNCTIONS


def cached_batch_draws(code,variables):
    """Return rewritten statements and draws in Python expression order.

    The caller allocates one persistent field per arrival lane and evaluates
    these streams with that lane's emission/pending identity. Later whole-array
    actions read fields, so they never sample from a synthetic owner identity.
    """
    tree=ast.parse(code);names={n.id for n in ast.walk(tree) if isinstance(n,ast.Name)};draws=[]
    class Cache(ast.NodeTransformer):
        def visit_Call(self,node):
            if isinstance(node.func,ast.Name) and node.func.id in ('rand','randn'):
                if node.args or node.keywords:raise ValueError('invalid random function arguments')
                if variables.get(node.func.id) is not DEFAULT_FUNCTIONS[node.func.id]:
                    raise ValueError('custom random functions are unsupported')
                if len(draws)>=16:raise ValueError('synaptic code exceeds 16 random streams')
                name='_b2_batch_random_'+str(len(draws))
                if name in names:raise ValueError('reserved batch random name collision')
                draws.append(dict(name=name,kind=node.func.id,stream=len(draws)))
                return ast.copy_location(ast.Name(id=name,ctx=ast.Load()),node)
            return self.generic_visit(node)
    return ast.unparse(Cache().visit(tree)),draws
