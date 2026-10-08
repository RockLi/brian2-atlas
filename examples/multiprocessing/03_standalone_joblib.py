"""
This example uses C++ standalone mode by default for the simulation and the
`joblib library <https://joblib.readthedocs.io>`_
to parallelize the code. See the previous example (``02_using_standalone.py``)
for more explanations, including how to select an optional standalone device.
"""
import importlib
import os
from time import time as wall_time

from brian2 import *
from joblib import Parallel, delayed


standalone_device = os.environ.get("BRIAN2_STANDALONE_DEVICE", "cpp_standalone")
standalone_module = os.environ.get("BRIAN2_STANDALONE_MODULE")


def run_sim(tau):
    pid = os.getpid()
    directory = f"standalone{pid}"
    # loky starts fresh Python interpreters, so optional Devices have to be
    # imported and registered inside each worker.
    if standalone_module:
        importlib.import_module(standalone_module)
    set_device(standalone_device, directory=directory)
    print(f'RUNNING {pid}')

    G = NeuronGroup(1, 'dv/dt = -v/tau : 1', method='euler')
    G.v = 1

    mon = StateMonitor(G, 'v', record=0)
    net = Network()
    net.add(G, mon)
    net.run(100 * ms)
    res = (mon.t/ms, mon.v[0])

    device.reinit()

    print(f'FINISHED {pid}')
    return res


if __name__ == "__main__":
    start_time = wall_time()

    n_jobs = int(os.environ.get("BRIAN2_EXAMPLE_PROCESSES", "4"))
    num_simulations = int(os.environ.get("BRIAN2_EXAMPLE_SIMULATIONS", "10"))
    tau_values = np.arange(num_simulations)*ms + 5*ms

    results = Parallel(n_jobs=n_jobs)(map(delayed(run_sim), tau_values))

    print(f"Done in {wall_time() - start_time:10.3f}")

    for tau_value, (t, v) in zip(tau_values, results):
        plt.plot(t, v, label=str(tau_value))
    plt.legend()
    plt.show()
