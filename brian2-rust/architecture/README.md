# Atlas architecture

Atlas has three jobs: simulate Brian2 models, train supported networks, and
convert admitted model profiles to neuromorphic software targets. They share the
modelling frontend, but have different execution contracts.

![Atlas architecture](atlas-architecture.svg)

[English PNG](atlas-architecture.png) · [中文 PNG](atlas-architecture-zh.png) ·
[中文 SVG](atlas-architecture-zh.svg)

## The three paths

| Capability | Intermediate contract | Executor | Result |
|---|---|---|---|
| Simulation | AtlasIR, frozen `b2ir-v1` wire format | Native CPU/AOT, CUDA, Metal, browser WASM/WebGPU, MPI | State trajectories, events, monitor records |
| Training | Dedicated versioned training plans | `b2-train` on CPU, CUDA, Metal or MPI | Learned weights, optimizer and physical-state checkpoints |
| Conversion | Validated AtlasIR, then a target-specific private plan | Optional vendor/toolchain SDK on CPU | Target outputs, states or spike times for comparison |

### Simulation

`AtlasDevice` lowers Brian2 models to AtlasIR and selects an admitted execution
backend. Browser deployment uses the separate `export_wasm_bundle` API. The
simulation representation describes typed state, topology, clocks, scheduling,
events and ordering; backend capability checks reject unsupported constructs.

Heterogeneous MPI execution has a specific scope: CPU ranks retain canonical
event and queue ordering and f64 storage; selected population state-update nodes
can execute on CUDA or Metal in f32. This is not a claim that arbitrary model
operations can move freely between CPU and GPU.

### Training

`NativeLIFTrainer` and the supported Brian network lowering APIs build dedicated
training plans. Native execution performs forward evaluation, surrogate-gradient
BPTT/TBPTT and optimizer updates. Evaluation and gradient calls have different
state-commit rules from training steps; checkpoints include optimizer and physical
state as specified by the training contract.

The training plans are versioned independently of the frozen AtlasIR simulation
ABI. A shared frontend does not mean that an arbitrary simulation model can be
trained or exported to a chip. The first [native training bridge](../examples/neuromorphic/native_training/README.md)
now freezes a bounded dense LIF model, applies explicit quantization, constructs
an inference-equivalent Brian2 model, and exports through AtlasIR to SpiNNaker2.
It admits beta=0.5, subtract reset and fresh binary-input sequences of at most
16 ticks. General training plans and live checkpoint continuation are not
supported by this bridge.

### Neuromorphic conversion

`convert_ir` / `export_ir` start with AtlasIR, run native validation and target
profile checks, and produce a backend-specific plan. `run_simulator` executes it
through an optional SDK. Each target has its own supported equations, numeric
ranges and resource constraints; an unsupported combination must be rejected.

The current targets are Akida, Lava, Specksim, py-spinnaker2, XyloSim and DynapSim.
They do not all accept the same models:

- Akida, Lava and SpiNNaker2 have evidence for the original frozen integer
  784→256→10 TensorFlow MNIST model at 98.74% on 10,000 test images.
- A separate binary count classifier converts to five targets at 90.03%.
  DynapSim does not admit that static classifier profile.
- Temporal LIF, integer recurrent and analog DPI profiles have separate
  state/spike comparisons. Their support does not follow from static MNIST
  conversion, and static encodings do not establish general SNN timing
  equivalence.

The external TensorFlow path imports frozen parameters into an explicitly
constructed Brian2 model. It is not a general Keras/SavedModel graph importer,
and it does not train on the target chip. Native Atlas training is a second
source path: its bounded LIF exporter joins the same AtlasIR conversion API,
with separate float-versus-quantized and quantized-versus-SDK checks.

## What the validation means

Native CPU/GPU execution and same-host MPI have bounded validation recorded in
the migration evidence. Cross-host and multiple-physical-GPU qualification remain
open. The heterogeneous offload profile does not expand those evidence claims.

Neuromorphic checks compare the Atlas reference with SDK results: integer
outputs, intermediate states and/or spike ticks, using exact equality or the
profile's declared tolerance. The common reference is the link between the
simulation and conversion paths in the figure.

All neuromorphic execution evidence here is software-only. In particular,
py-spinnaker2 uses the upstream Brian2 backend rather than an independent silicon
emulator, and DynapSim models analog circuit dynamics rather than bit-exact chip
behavior. Simulator admission does not guarantee hardware fit: the Xylo count
MNIST encoding exceeds the physical chip's input-channel limit. Hardware
placement, physical execution, power and hardware latency are not qualified.

## Implementation and evidence

- [Simulation/backend guide](../../brian2-rust/README.md)
- [AtlasIR contract](../../brian2-rust/AtlasIR.md)
- [Native training guide](../../brian2-rust/TRAINING.md)
- [Training-plan implementation](../../brian2-rust/python/brian2_rust/training.py)
- [Heterogeneous MPI profile](../../brian2-rust/python/brian2_rust/mpi_gpu.py)
- [Neuromorphic compatibility checklist](../../brian2-rust/examples/neuromorphic/COMPATIBILITY.md)
- [Full-model SpiNNaker2 evidence](../../brian2-rust/examples/neuromorphic/spinnaker2_dense/README.md)
- [Migration qualification](../../migration/README.md)

## Regenerate the figures

The SVGs contain vector paths and do not require the reader to have the rendering
fonts installed. PNGs are provided for presentations and sharing. Regenerate with
Matplotlib using [the renderer](../../dev/atlas/render_architecture.py):

```sh
MPLCONFIGDIR=/tmp/atlas-mpl python dev/atlas/render_architecture.py
```

English always renders. Chinese also renders when `/Library/Fonts/Arial Unicode.ttf`
is present; on other systems, pass `--cjk-font /path/to/a/CJK-font.ttf`. The renderer
is a documentation tool and adds no dependency to the Atlas runtime.
