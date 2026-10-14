"""Scalar legacy Poisson calls have one count per Python callback."""
import numpy as np


def direct_poisson(x,rate,unused):
    return np.random.poisson(lam=rate,size=None)


def stored_poisson(x,rate,unused):
    counts=np.random.poisson(rate)
    old=counts
    counts+=1
    return old


def response_poisson(x,rate,unused):
    x*=.7
    counts=np.random.poisson(rate)
    return x+.1*counts


def twice_poisson(x,rate,unused):
    return np.random.poisson(rate)+np.random.poisson(rate)


def default_poisson(x,rate,unused):
    return np.random.poisson()


def derived_poisson(x,rate,unused):
    return np.random.poisson(2.*rate)
