# Simulation and native training quickstart

Install Atlas using the [installation guide](INSTALLATION.md). These examples
use the core package and NumPy, with no plotting or vendor SDK dependencies.
Save each example as a separate Python script and run it in the installed
environment outside the source checkout.

## Run a spiking neuron

```python
import brian2 as b
import brian2_atlas

b.set_device("atlas", engine="reference")
neurons = b.NeuronGroup(
    1, "dv/dt = (1.5-v)/(10*ms) : 1",
    threshold="v>1", reset="v=0", method="euler", dt=0.1*b.ms,
)
spikes = b.SpikeMonitor(neurons)
network = b.Network(neurons, spikes)
network.run(100*b.ms)
assert spikes.num_spikes == 9
print("spikes:", spikes.num_spikes)
```

Importing `brian2_atlas` registers the `atlas` Device. `engine="reference"` uses
the bundled native CPU runner. `engine="aot"` selects model-specialized CPU
execution and needs the pinned Rust toolchain. Unsupported model features raise
capability errors; see the [backend guide](../brian2-rust/README.md).

## Compute gradients and take a training step

Native training has its own versioned plans. This small two-class example uses
two input channels, two hidden neurons and two output neurons. It checks gradient
execution and checkpoint creation; it does not measure classification accuracy.

```python
from pathlib import Path
import numpy as np
import brian2_atlas as atlas

inputs = np.zeros((4, 24, 2))  # batch, time steps, input channels
inputs[:2, :, 0] = 1
inputs[2:, :, 1] = 1
labels = [0, 0, 1, 1]
plan = atlas.lif_training_plan([2, 2, 2], beta=0.8)
trainer = atlas.NativeLIFTrainer(
    plan,
    weights=[[1.2, 0.4, 0.4, 1.2], [0.2, 0.8, 0.8, 0.2]],
)
result = trainer.gradients(inputs, labels)
assert np.isfinite(result["loss"])
assert any(np.any(np.asarray(g) != 0) for g in result["gradients"])
trainer.step(inputs, labels)
checkpoint = Path("atlas-quickstart-checkpoint.json")
trainer.store(checkpoint)

restored = atlas.NativeLIFTrainer(plan)
restored.restore(checkpoint)
assert restored.state == trainer.state
print("finite loss:", result["loss"], "checkpoint:", checkpoint)
```

The example writes a checkpoint in the working directory. Use the
[training guide](../brian2-rust/TRAINING.md) for supported neuron models, Brian
lowering, optimizers and GPU/MPI training. General training plans are separate
from the frozen `atlasir-v1` simulation ABI.

## Continue with supported examples

The [source example guide](../brian2-rust/examples/README.md) covers simulation,
training and MPI checkpoints. Examples live in the repository, so use an example
from the same tag or commit as the installed code. Development conversion
examples have additional SDK requirements and bounded model support; consult
the [compatibility checklist](../brian2-rust/examples/neuromorphic/COMPATIBILITY.md).
