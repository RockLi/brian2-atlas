"""Cache each event's random arrays before evaluating a whole capture callback."""
import ast
from brian2.core.functions import DEFAULT_FUNCTIONS


def cached_batch_draws(code,variables,*,poisson_sites=None,callback_poisson_names=(),callback_normal_sites=None):
    """Return rewritten statements and draws in Python expression order.

    The caller allocates one persistent field per arrival lane and evaluates
    these streams with that lane's emission/pending identity. Later whole-array
    actions read fields, so they never sample from a synthetic owner identity.
    """
    callback_normal_sites={} if callback_normal_sites is None else callback_normal_sites
    tree=ast.parse(code);names={n.id for n in ast.walk(tree) if isinstance(n,ast.Name)};draws=[];stream=0
    class Cache(ast.NodeTransformer):
        def visit_Call(self,node):
            nonlocal stream
            if poisson_sites is not None and isinstance(node.func,ast.Name) and node.func.id in {'poisson',*callback_poisson_names}:
                if len(node.args)!=1 or node.keywords:raise ValueError('invalid random function arguments')
                if variables.get(node.func.id) is not DEFAULT_FUNCTIONS['poisson']:raise ValueError('custom random functions are unsupported')
                node=self.generic_visit(node)
                if stream>=16:raise ValueError('synaptic code exceeds 16 random streams')
                alias='_b2_batch_poisson_'+str(stream)
                if alias in names:raise ValueError('reserved batch random name collision')
                poisson_sites[alias]=dict(kind='poisson',stream=stream,**({'callback':True} if node.func.id in callback_poisson_names else {}));stream+=1
                node.func=ast.Name(id=alias,ctx=ast.Load());return node
            if isinstance(node.func,ast.Name) and node.func.id in {'rand','randn',*callback_normal_sites}:
                site=callback_normal_sites.get(node.func.id);kind=site['kind'] if site else node.func.id
                if site:
                    domains=site['domain'] if isinstance(site['domain'],tuple) else (site['domain'],)
                    if len(node.args)!=len(domains) or any(not isinstance(arg,ast.Name) or arg.id!=domain for arg,domain in zip(node.args,domains)):raise ValueError('invalid callback random domain')
                elif node.args:raise ValueError('invalid random function arguments')
                if node.keywords:raise ValueError('invalid random function arguments')
                if variables.get(node.func.id) is not DEFAULT_FUNCTIONS[kind]:
                    raise ValueError('custom random functions are unsupported')
                if stream>=16:raise ValueError('synaptic code exceeds 16 random streams')
                name='_b2_batch_random_'+str(len(draws))
                if name in names:raise ValueError('reserved batch random name collision')
                draws.append(dict(name=name,kind=kind,stream=stream,**({'callback_domain':site['domain']} if site else {})));stream+=1
                return ast.copy_location(ast.Name(id=name,ctx=ast.Load()),node)
            return self.generic_visit(node)
    return ast.unparse(Cache().visit(tree)),draws
