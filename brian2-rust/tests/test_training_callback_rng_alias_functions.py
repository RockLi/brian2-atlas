"""Readable callbacks for standard RNG module and direct function aliases."""
import numpy as np
import numpy.random as rng
from numpy.random import normal as imported_normal, uniform as imported_uniform
from numpy.random import randn as imported_randn, rand as imported_rand, poisson as imported_poisson


def module_normal(x,mean,spread):
    return rng.normal(mean,spread)


def module_uniform(x,mean,spread):
    return rng.uniform(mean,mean+spread,size=None)


def function_normal(x,mean,spread):
    return imported_normal(mean,spread)


def function_uniform(x,mean,spread):
    return imported_uniform(mean,mean+spread,size=None)


def module_randn(x):
    x*=.7
    return x+.1*rng.randn(len(x))


def function_randn(x):
    x*=.7
    return x+.1*imported_randn(len(x))


def qualified_randn(x):
    x*=.7
    return x+.1*np.random.randn(len(x))


def module_rand(x):
    x*=.7
    return x+.1*rng.rand(len(x))


def function_rand(x):
    x*=.7
    return x+.1*imported_rand(len(x))


def qualified_rand(x):
    x*=.7
    return x+.1*np.random.rand(len(x))


def module_poisson(x):
    x*=.7
    return x+.1*rng.poisson(x)


def function_poisson(x):
    x*=.7
    return x+.1*imported_poisson(x)


def qualified_poisson(x):
    x*=.7
    return x+.1*np.random.poisson(x)


def shadowed_function(x,imported_normal):
    x*=.7
    return imported_normal(x)


def shadowed_module(x,rng):
    x*=.7
    return rng.normal(x)


def shadowed_numpy(x,np):
    x*=.7
    return np.random.normal(x)


def closure_normal():
    sampler=imported_normal
    def callback(x,mean,spread):
        return sampler(mean,spread)
    return callback


def keyword_poisson(x):
    x*=.7
    return x+.1*rng.poisson(lam=x)


def keyword_none_poisson(x):
    x*=.7
    return x+.1*imported_poisson(size=None,lam=x)


def positional_none_poisson(x):
    x*=.7
    return x+.1*np.random.poisson(x,None)


def sized_poisson(x):
    x*=.7
    return x+.1*rng.poisson(lam=x,size=len(x))


def duplicate_poisson(x):
    x*=.7
    return x+.1*rng.poisson(x,lam=x)


def keyword_captured_poisson(array):
    def callback(x):
        x*=.7
        counts=rng.poisson(lam=x,size=None)
        captured=array
        captured+=.03*counts
        return x+.1*counts
    return callback


def imported_captured_poisson(array):
    def callback(x):
        x*=.7
        counts=imported_poisson(size=None,lam=x)
        captured=array
        captured+=.03*counts
        return x+.1*counts
    return callback
