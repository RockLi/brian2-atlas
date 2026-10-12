"""Scalar FIFO callback factories with authenticated legacy sampler bodies."""
import numpy as np


def normal(array):
    def f(x,mean,spread):
        captured=array
        captured*=.8
        x*=.7
        return x+.1*np.random.normal(mean,spread)
    return f


def uniform(array):
    def f(x,mean,spread):
        captured=array
        captured*=.8
        x*=.7
        return x+.1*np.random.uniform(mean,mean+spread,size=None)
    return f


def poisson(array):
    def f(x,mean,spread):
        captured=array
        captured*=.8
        x*=.7
        return x+.1*np.random.poisson(mean)
    return f


def pure_normal(x,mean,spread):
    return x+.1*np.random.normal(mean,spread)


def pure_uniform(x,mean,spread):
    return x+.1*np.random.uniform(mean,mean+spread,size=None)


def pure_poisson(x,mean,spread):
    return x+.1*np.random.poisson(mean)


def twice_normal(x,mean,spread):
    return x+.1*np.random.normal(mean,spread)+.1*np.random.normal(mean,spread)


def mixed(array):
    def f(x,mean,spread):
        captured=array
        captured*=.8
        x*=.7
        return x+.1*np.random.normal(mean,spread)+.1*np.random.uniform(mean,mean+spread)+.1*np.random.poisson(mean)
    return f


def pure_mixed(x,mean,spread):
    return x+.1*np.random.normal(mean,spread)+.1*np.random.uniform(mean,mean+spread)+.1*np.random.poisson(mean)
