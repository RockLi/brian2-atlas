"""Stable callback source locations for independent training references."""


def add_selected(array):
    def f(x):
        captured=array
        captured+=x
        x*=.7
        return x
    return f


def discard_selected(array):
    def f(x):
        temporary=x+array
        temporary*=.8
        x*=.7
        return x
    return f


def add_two_selected(a,b):
    def f(x):
        first=a
        second=b
        first+=x
        second+=x
        x*=.7
        return x
    return f


def invalid_guarded_capture(array):
    def f(x):
        captured=array
        captured+=x
        temporary=1./0.
        return x+temporary
    return f


def scale_selected_by_capture(array):
    def f(x):
        captured=array
        captured*=.9
        x*=captured
        return x
    return f


def add_constant_selected(array):
    def f(x):
        captured=array
        captured+=.003
        x*=.7
        return x
    return f
