"""Stable authenticated sources for callback-internal random sites."""
import numpy as np


def sample_selected(array):
    def f(x):
        x*=.7
        counts=np.random.poisson(x)
        captured=array
        captured+=.03*counts
        return x+.1*counts
    return f


def replay_domain_selected(array):
    def f(x):
        x*=.7
        counts=np.random.poisson(x)
        captured=array
        captured+=.03*counts
        value=.1/(1.-counts)
        return x+value
    return f


def normal_selected(array):
    def f(x):
        x*=.7
        values=np.random.randn(len(x))
        captured=array
        captured+=.03*values
        return x+.1*values
    return f


def uniform_selected(array):
    def f(x):
        x*=.7
        values=np.random.rand(len(x))
        captured=array
        captured+=.03*values
        return x+.1*values
    return f


def mixed_selected(array):
    def f(x):
        x*=.7
        normal=np.random.randn(len(x))
        uniform=np.random.rand(len(x))
        counts=np.random.poisson(x+.02*normal)
        captured=array
        captured+=.03*normal+.02*uniform+.01*counts
        return x+.1*normal+.1*uniform+.05*counts
    return f


def normal_domain_selected(array):
    def f(x):
        x*=.7
        values=np.random.randn(len(x))
        captured=array
        captured+=.03*values
        value=(x+.1)**.5
        return value+.1*values
    return f


def uniform_domain_selected(array):
    def f(x):
        x*=.7
        values=np.random.rand(len(x))
        captured=array
        captured+=.03*values
        value=(x+.1)**.5
        return value+.1*values
    return f


def affine_normal_selected(array):
    def f(x,spread):
        x*=.7
        values=np.random.normal(x,spread,size=len(x))
        captured=array
        captured+=.03*values
        return x+.1*values
    return f


def affine_uniform_selected(array):
    def f(x,spread):
        x*=.7
        values=np.random.uniform(x,x+spread,size=len(x))
        captured=array
        captured+=.03*values
        return x+.1*values
    return f


def keyword_normal_selected(array):
    def f(x,spread):
        x*=.7
        values=np.random.normal(scale=spread,size=len(x),loc=x)
        captured=array
        captured+=.03*values
        return x+.1*values
    return f


def keyword_uniform_selected(array):
    def f(x,spread):
        x*=.7
        values=np.random.uniform(high=x+spread,size=len(x),low=x)
        captured=array
        captured+=.03*values
        return x+.1*values
    return f


def default_normal_selected(array):
    def f(x,spread):
        x*=.7
        values=np.random.normal(size=len(x))
        captured=array
        captured+=.03*values
        return x+.1*values
    return f


def default_uniform_selected(array):
    def f(x,spread):
        x*=.7
        values=np.random.uniform(size=len(x))
        captured=array
        captured+=.03*values
        return x+.1*values
    return f


def inferred_normal_selected(array):
    def f(x,spread):
        x*=.7
        values=np.random.normal(x,spread)
        captured=array
        captured+=.03*values
        return x+.1*values
    return f


def inferred_uniform_selected(array):
    def f(x,spread):
        x*=.7
        values=np.random.uniform(x,x+spread,size=None)
        captured=array
        captured+=.03*values
        return x+.1*values
    return f


def inferred_scale_normal_selected(array):
    def f(x,spread):
        x*=.7
        values=np.random.normal(scale=spread)
        captured=array
        captured+=.03*values
        return x+.1*values
    return f


def inferred_high_uniform_selected(array):
    def f(x,spread):
        x*=.7
        values=np.random.uniform(high=spread,size=None)
        captured=array
        captured+=.03*values
        return x+.1*values
    return f


def scalar_normal_selected(array):
    def f(x,spread):
        x*=.7
        values=np.random.normal(.04,.13)
        captured=array
        captured+=.03*values
        return x+.1*values
    return f


def scalar_uniform_selected(array):
    def f(x,spread):
        x*=.7
        values=np.random.uniform(.04,.17,size=None)
        captured=array
        captured+=.03*values
        return x+.1*values
    return f


def scalar_randn_selected(array):
    def f(x,spread):
        x*=.7
        values=np.random.randn()
        captured=array
        captured+=.03*values
        return x+.1*values
    return f


def scalar_rand_selected(array):
    def f(x,spread):
        x*=.7
        values=np.random.rand()
        captured=array
        captured+=.03*values
        return x+.1*values
    return f


def scalar_parameter_normal_selected(array):
    def f(x,mean,spread):
        x*=.7
        values=np.random.normal(mean,spread)
        captured=array
        captured+=.03*values
        return x+.1*values
    return f


def scalar_parameter_uniform_selected(array):
    def f(x,mean,spread):
        x*=.7
        values=np.random.uniform(mean,mean+spread,size=None)
        captured=array
        captured+=.03*values
        return x+.1*values
    return f


def scalar_alias_normal_selected(array):
    def f(x,spread):
        x*=.7
        values=np.random.normal(.04,.13)
        alias=values
        values*=.8
        captured=array
        captured+=.03*alias
        return x+.1*values
    return f
