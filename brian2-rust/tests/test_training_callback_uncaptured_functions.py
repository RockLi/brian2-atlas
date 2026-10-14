"""Authenticated no-capture stochastic functions, including pure/direct returns."""
import numpy as np


def stored_normal(x,mean,spread):
    x*=.7
    values=np.random.normal(mean,spread)
    return x+.1*values


def stored_uniform(x,mean,spread):
    x*=.7
    values=np.random.uniform(mean,mean+spread,size=None)
    return x+.1*values


def direct_normal(x,mean,spread):
    x*=.7
    return x+.1*np.random.normal(mean,spread)


def direct_uniform(x,mean,spread):
    x*=.7
    return x+.1*np.random.uniform(mean,mean+spread,size=None)


def pure_normal(x,mean,spread):
    return x+.1*np.random.normal(mean,spread)


def pure_uniform(x,mean,spread):
    return x+.1*np.random.uniform(mean,mean+spread,size=None)


def scalar_normal(x,mean,spread):
    return np.random.normal(mean,spread)


def scalar_uniform(x,mean,spread):
    return np.random.uniform(mean,mean+spread,size=None)
