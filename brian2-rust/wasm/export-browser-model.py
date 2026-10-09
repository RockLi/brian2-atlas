"""Export two connected populations without simulating them in Python.

Run from this repository with brian2-rust/python on PYTHONPATH and a built native
b2-runner validator. The browser receives a model bundle, not a Python program.
"""
import argparse
from pathlib import Path
import brian2 as b
from brian2_rust import export_network, export_wasm_bundle


def export_example(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    previous = b.get_device()
    b.set_device('rust_standalone', build_on_run=False)
    try:
        source = b.NeuronGroup(24, 'dv/dt=(drive-v)/(20*ms):1\ndrive:1 (constant)',
                              threshold='v>1', reset='v=0', method='euler',
                              dt=0.1*b.ms, name='input_cells')
        source.drive = [1.25 + 0.035*i for i in range(24)]
        target = b.NeuronGroup(12, 'dvm/dt=(-65*mV-vm)/(10*ms):volt',
                              threshold='vm>-55*mV', reset='vm=-65*mV',
                              refractory=2*b.ms, method='euler', dt=0.1*b.ms,
                              name='output_cells')
        target.vm = -65*b.mV
        synapses = b.Synapses(source, target, on_pre='vm_post += 12*mV', clock=source.clock,
                             delay=0.5*b.ms, name='feedforward')
        synapses.connect(i=list(range(24)), j=[i % 12 for i in range(24)])
        monitors = [b.SpikeMonitor(source), b.SpikeMonitor(target),
                    b.StateMonitor(source, 'v', record=[0, 6, 12, 18]),
                    b.StateMonitor(target, 'vm', record=[0, 3, 6, 9])]
        network = b.Network(source, target, synapses, *monitors)
        model = export_network(network, 300*b.ms, directory/'model.json')
        export_wasm_bundle(model, directory/'import-example.browser.json')
        return directory/'import-example.browser.json'
    finally:
        b.get_device().reinit()
        b.set_device(previous)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('browser-export'))
    args = parser.parse_args()
    print(export_example(args.output))
