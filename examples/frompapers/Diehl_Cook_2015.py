"""
Unsupervised learning using STDP
--------------------------------
Diehl, P. U., & Cook, M. (2015). Unsupervised learning of digit
recognition using spike-timing-dependent plasticity. Frontiers in
computational neuroscience, 9, 99.

This script replicates the small 2x400-model. You can either change the
constants below or use the command line parameters. Run the script in
``train`` mode, which
(eventually) creates the files theta.npy and weights.npy in the
DATA_PATH directory. Rerun it with MODE set to "observe" to create the
assign.npy file in the same directory. Then, run "test" to create a
confusion matrix in confusion.npy. Finally, you can use "plot" to
plot the confusion matrix. The script also creates a few auxilliary
.npy files useful for analysis. The script requires the progressbar2
library.

MNIST_PATH should point to the directory storing either compressed or
unzipped IDX MNIST files. Pass ``--download-mnist`` to download and verify
the four files from the official PyTorch MNIST mirror.
For reasonable accuracy, N_TRAIN should be 50,000+ and N_OBSERVE 1,000+.

Written in 2024 by Björn A. Lindqvist <bjourne@gmail.com>
"""
from brian2 import *
import argparse
from collections import defaultdict
import gzip
import hashlib
from pathlib import Path
from progressbar import progressbar
from random import randrange, seed as rseed
from struct import unpack
import urllib.request
import numpy as np

# Switch between "train", "observe", and "test" to tune parameters,
# observe excitatory spiking, and test accuracy, respectively.
# Use "plot" to plot the confusion matrix.
MODE = 'test'

# Number of training, observation, and testing samples
N_TRAIN = 25_000
N_OBSERVE = 2_000
N_TEST = 1_000

# Random seed value
SEED = 42

# Storage paths
MNIST_PATH = Path('../mnist')
DATA_PATH = Path('data')

# Number of weight save points
N_SAVE_POINTS = 100

# Presentation settings. Command line overrides are useful for quick
# compatibility checks; the defaults retain the published behaviour.
PRESENTATION_DURATION = 350 * ms
REST_DURATION = 150 * ms
MIN_SPIKES = 5

# Don't change these values unless you know what you're doing.
N_INP = 784
N_NEURONS = 400
V_EXC_REST = -65 * mV
V_INH_REST = -60 * mV
INTENSITY = 2

# Weights of exc->inh and inh->exc synapses
W_EXC_INH = 10.4
W_INH_EXC = 17.0

MNIST_URL = 'https://ossci-datasets.s3.amazonaws.com/mnist/'
MNIST_FILES = {
    'train-images-idx3-ubyte.gz': 'f68b3c2dcbeaaa9fbdd348bbdeb94873',
    'train-labels-idx1-ubyte.gz': 'd53e105ee54ea40749a09fcbcd1e9432',
    't10k-images-idx3-ubyte.gz': '9fb629c4189551a2d022fa330f9573f3',
    't10k-labels-idx1-ubyte.gz': 'ec29112dd5afa0611ce80d1b7f02629c',
}

def save_npy(arr, path):
    arr = np.array(arr)
    print('%-9s %-15s => %-30s' % ('Saving', arr.shape, path))
    np.save(path, arr)

def load_npy(path):
    arr = np.load(path)
    print('%-9s %-30s => %-15s' % ('Loading', path, arr.shape))
    return arr

def download_mnist(path=None):
    """Download the original MNIST IDX files with checksum verification."""
    path = MNIST_PATH if path is None else Path(path)
    path.mkdir(parents=True, exist_ok=True)
    for filename, expected_md5 in MNIST_FILES.items():
        destination = path / filename
        if destination.exists():
            digest = hashlib.md5(destination.read_bytes()).hexdigest()
            if digest == expected_md5:
                continue
            raise ValueError(f'Checksum mismatch for existing file {destination}')
        print(f'Downloading {MNIST_URL + filename} => {destination}')
        with urllib.request.urlopen(MNIST_URL + filename, timeout=60) as response:
            data = response.read()
        if hashlib.md5(data).hexdigest() != expected_md5:
            raise ValueError(f'Checksum mismatch for downloaded file {filename}')
        temporary = destination.with_suffix(destination.suffix + '.partial')
        temporary.write_bytes(data)
        temporary.replace(destination)


def _read_idx(path):
    """Read a uint8 IDX array and reject malformed or truncated files."""
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rb') as handle:
        header = handle.read(4)
        if len(header) != 4:
            raise ValueError(f'Truncated IDX header in {path}')
        zero, dtype, ndim = unpack('>HBB', header)
        if zero != 0 or dtype != 0x08 or ndim == 0:
            raise ValueError(f'Unsupported IDX header in {path}')
        shape_data = handle.read(4 * ndim)
        if len(shape_data) != 4 * ndim:
            raise ValueError(f'Truncated IDX dimensions in {path}')
        shape = unpack('>' + 'I' * ndim, shape_data)
        payload = handle.read()
    expected = int(np.prod(shape))
    if expected != len(payload):
        raise ValueError(
            f'IDX payload size mismatch in {path}: expected {expected}, '
            f'got {len(payload)}'
        )
    return np.frombuffer(payload, dtype=np.uint8).reshape(shape).copy()


def _mnist_path(tag, kind):
    dimension = '3' if kind == 'images' else '1'
    basename = f'{tag}-{kind}-idx{dimension}-ubyte'
    for suffix in ('', '.gz'):
        candidate = MNIST_PATH / (basename + suffix)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f'Missing {basename}[.gz] in {MNIST_PATH}. '
        'Run this script with --download-mnist.'
    )


def read_mnist(training):
    tag = 'train' if training else 't10k'
    images = _read_idx(_mnist_path(tag, 'images'))
    labels = _read_idx(_mnist_path(tag, 'labels'))
    if images.ndim != 3 or images.shape[1:] != (28, 28):
        raise ValueError(f'Unexpected MNIST image dimensions: {images.shape}')
    if labels.ndim != 1 or len(images) != len(labels):
        raise ValueError('MNIST image and label counts do not match')
    return images.reshape(len(images), -1) / 8.0, labels

def build_network(training):
    eqs = '''
    dv/dt = (v_rest - v + i_exc + i_inh) / tau_mem  : volt (unless refractory)
    i_exc = ge * -v                         : volt
    i_inh = gi * (v_inh_base - v)           : volt
    dge/dt = -ge/(1 * ms)                   : 1
    dgi/dt = -gi/(2 * ms)                   : 1
    dtimer/dt = 1                           : second
    '''
    reset = 'v = %r; timer = 0 * ms' % V_EXC_REST
    if training:
        exc_eqs = eqs + '''
        dtheta/dt = -theta / (1e7 * ms)         : volt
        '''
        arr_theta = np.ones(N_NEURONS) * 20 * mV
        reset += '; theta += 0.05 * mV'
    else:
        exc_eqs = eqs + '''
        theta                                   : volt
        '''
        arr_theta = load_npy(DATA_PATH / 'theta.npy') * volt
    exc_eqs = Equations(exc_eqs,
                        tau_mem = 100 * ms,
                        v_rest = V_EXC_REST,
                        v_inh_base = -100 * mV)
    # Note that this neuron has a bit of un unusual refractoriness mechanism:
    # The membrane potential is clamped for 5ms, but spikes are prevented for 50ms
    # This has been taken from the original code.
    ng_exc = NeuronGroup(
        N_NEURONS, exc_eqs,
        threshold = 'v > (theta - 72 * mV) and (timer > 50 * ms)',
        refractory = 5 * ms,
        reset = reset,
        method = 'euler',
        name = 'exc')
    ng_exc.v = V_EXC_REST
    ng_exc.theta = arr_theta

    inh_eqs = Equations(eqs,
                        tau_mem = 10 * ms,
                        v_rest = V_INH_REST,
                        v_inh_base = -85 * mV)
    ng_inh = NeuronGroup(N_NEURONS, inh_eqs,
                         threshold = 'v > -40 * mV',
                         refractory = 2 * ms,
                         reset = 'v = -45 * mV',
                         method = 'euler',
                         name = 'inh')
    ng_inh.v = V_INH_REST

    syns_exc_inh = Synapses(ng_exc, ng_inh,
                            on_pre = 'ge_post += %f' % W_EXC_INH)
    syns_exc_inh.connect(j = 'i')

    syns_inh_exc = Synapses(ng_inh, ng_exc,
                            on_pre = 'gi_post += %f' % W_INH_EXC)
    syns_inh_exc.connect("i != j")

    pg_inp = PoissonGroup(N_INP, 0 * Hz, name = 'inp')

    # During training, inp->exc synapse weights are plastic.
    model = 'w : 1'
    on_post = ''
    on_pre = 'ge_post += w'
    if training:
        on_pre += '; pre = 1.; w = clip(w - 0.0001 * post1, 0, 1.0)'
        on_post += 'post2bef = post2; w = clip(w + 0.01 * pre * post2bef, 0, 1.0); post1 = 1.; post2 = 1.'
        model += '''
        post2bef                        : 1
        dpre/dt   = -pre/(20 * ms)      : 1 (event-driven)
        dpost1/dt = -post1/(20 * ms)    : 1 (event-driven)
        dpost2/dt = -post2/(40 * ms)    : 1 (event-driven)
        '''
        weights = (np.random.random(N_INP * N_NEURONS) + 0.01) * 0.3
    else:
        weights = load_npy(DATA_PATH / 'weights.npy')

    syns_inp_exc = Synapses(
        pg_inp, ng_exc,
        model = model,
        on_pre = on_pre,
        on_post = on_post,
        name = 'inp_exc'
    )
    syns_inp_exc.connect(True)
    syns_inp_exc.delay = 'rand() * 10 * ms'
    syns_inp_exc.w = weights

    exc_mon = SpikeMonitor(ng_exc, name = 'sp_exc')
    net = Network([pg_inp, ng_exc, ng_inh,
                   syns_inp_exc, syns_exc_inh, syns_inh_exc,
                   exc_mon])
    # Initialize
    net.run(0 * ms)
    return net

def show_sample(net, sample, intensity):
    exc_mon = net['sp_exc']
    prev = exc_mon.count[:]
    net['inp'].rates = sample * intensity * Hz
    net.run(PRESENTATION_DURATION)
    # Don't count spikes occuring during the 150 ms rest.
    next = exc_mon.count[:]
    net['inp'].rates = 0 * Hz
    net.run(REST_DURATION)
    pat = next - prev
    cnt = np.sum(pat)
    if cnt < MIN_SPIKES:
        return show_sample(net, sample, intensity + 1)
    return pat

def predict(groups, rates):
    scores = [rates[grp].mean() if len(grp) else -np.inf for grp in groups]
    return np.argmax(scores)

def test():
    conf = np.zeros((10, 10))
    assign = np.load(DATA_PATH / 'assign.npy')
    groups = [np.where(assign == i)[0] for i in range(10)]

    X, Y = read_mnist(False)
    net = build_network(False)
    for i in progressbar(range(N_TEST)):
        ix = randrange(len(X))
        exc = show_sample(net, X[ix], INTENSITY)
        guess = predict(groups, exc)
        real = Y[ix]
        conf[real, guess] += 1

    print('Accuracy: %6.3f' % (np.trace(conf) / np.sum(conf)))
    row_sums = conf.sum(axis=1)[:, None]
    conf = np.divide(conf, row_sums, out=np.zeros_like(conf), where=row_sums != 0)
    print(np.around(conf, 2))
    save_npy(conf, DATA_PATH / 'confusion.npy')

def normalize_plastic_weights(syns):
    conns = np.reshape(syns.w, (N_INP, N_NEURONS))
    col_sums = np.sum(conns, axis = 0)
    factors = 78./ col_sums
    conns *= factors
    syns.w = conns.reshape(-1)

def stats(net):
    tick = defaultclock.timestep[:]
    cnt = np.sum(net['sp_exc'].count[:])

    inp_exc = net['inp_exc']
    w_mu = np.mean(inp_exc.w)
    w_std = np.std(inp_exc.w)

    exc = net['exc']
    theta = exc.theta / mV
    theta_mu = np.mean(theta)
    theta_sig = np.std(theta)
    return [tick, cnt, w_mu, w_std, theta_mu, theta_sig]

def train():
    X, Y = read_mnist(True)
    n_samples = X.shape[0]
    net = build_network(True)
    rows = [stats(net) + [-1]]
    w_hist = [np.array(net['inp_exc'].w)]

    ratio = max(N_TRAIN // N_SAVE_POINTS, 1)
    for i in progressbar(range(N_TRAIN)):
        ix = i % n_samples
        normalize_plastic_weights(net['inp_exc'])
        show_sample(net, X[ix], INTENSITY)
        rows.append(stats(net) + [Y[ix]])
        if i % ratio == 0:
            w_hist.append(np.array(net['inp_exc'].w))

    save_npy(rows, DATA_PATH / 'train_stats.npy')
    save_npy(w_hist, DATA_PATH / 'train_w_hist.npy')
    save_npy(net['inp_exc'].w, DATA_PATH / 'weights.npy')
    save_npy(net['exc'].theta, DATA_PATH / 'theta.npy')

def observe():
    X, Y = read_mnist(True)
    n_samples = X.shape[0]
    net = build_network(False)
    rows = [stats(net) + [-1]]
    responses = defaultdict(list)

    for i in progressbar(range(N_OBSERVE)):
        ix = i % n_samples
        sample = X[ix]
        cls = Y[ix]
        exc = show_sample(net, sample, INTENSITY)
        rows.append(stats(net) + [Y[ix]])
        responses[cls].append(exc)

    res = np.zeros((10, N_NEURONS))
    for cls, vals in responses.items():
        res[cls] = np.array(vals).mean(axis = 0)

    assign = np.argmax(res, axis = 0)
    save_npy(assign, DATA_PATH / 'assign.npy')
    save_npy(rows, DATA_PATH / 'observe_stats.npy')

def plot():
    conf = np.load(DATA_PATH / "confusion.npy")

    import matplotlib.pyplot as plt

    plt.imshow(100*conf, interpolation="nearest", cmap=plt.cm.Blues)
    for i, j in itertools.product(range(conf.shape[0]), range(conf.shape[1])):
        if conf[i, j] == 0:
            continue
        plt.text(
            j,
            i,
            f"{round(100*conf[i, j])}%",
            horizontalalignment="center",
            verticalalignment="center",
            color="white" if conf[i, j] > 0.5 else "black",
        )
    plt.colorbar()
    plt.xticks(range(10))
    plt.yticks(range(10))
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.show()


def main(argv=None):
    global MODE, MNIST_PATH, DATA_PATH, N_TRAIN, N_OBSERVE, N_TEST
    global N_NEURONS, N_SAVE_POINTS, PRESENTATION_DURATION, REST_DURATION
    global MIN_SPIKES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', nargs='?',
                        choices=('train', 'observe', 'test', 'plot'),
                        default=MODE)
    parser.add_argument('--mnist-path', type=Path, default=MNIST_PATH)
    parser.add_argument('--data-path', type=Path, default=DATA_PATH)
    parser.add_argument('--download-mnist', action='store_true')
    parser.add_argument('--n-train', type=int, default=N_TRAIN)
    parser.add_argument('--n-observe', type=int, default=N_OBSERVE)
    parser.add_argument('--n-test', type=int, default=N_TEST)
    parser.add_argument('--neurons', type=int, default=N_NEURONS)
    parser.add_argument('--save-points', type=int, default=N_SAVE_POINTS)
    parser.add_argument('--presentation-ms', type=float,
                        default=float(PRESENTATION_DURATION / ms))
    parser.add_argument('--rest-ms', type=float,
                        default=float(REST_DURATION / ms))
    parser.add_argument('--min-spikes', type=int, default=MIN_SPIKES)
    parser.add_argument('--device', default=None,
                        help='Brian device, e.g. cpp_standalone or rust_standalone')
    parser.add_argument('--engine', default='reference',
                        help='rust_standalone execution engine')
    parser.add_argument('--runner', type=Path, default=None,
                        help='existing b2-runner executable for rust_standalone')
    parser.add_argument('--build-directory', type=Path, default=None)
    args = parser.parse_args(argv)

    MODE = args.mode
    MNIST_PATH = args.mnist_path
    DATA_PATH = args.data_path
    N_TRAIN = args.n_train
    N_OBSERVE = args.n_observe
    N_TEST = args.n_test
    N_NEURONS = args.neurons
    N_SAVE_POINTS = args.save_points
    PRESENTATION_DURATION = args.presentation_ms * ms
    REST_DURATION = args.rest_ms * ms
    MIN_SPIKES = args.min_spikes

    if min(N_TRAIN, N_OBSERVE, N_TEST, N_NEURONS, N_SAVE_POINTS) < 1:
        parser.error('sample counts, neurons, and save points must be positive')
    if args.presentation_ms < 0 or args.rest_ms < 0 or MIN_SPIKES < 0:
        parser.error('durations and minimum spike count must be non-negative')

    if args.device:
        device_options = {}
        if args.build_directory is not None:
            device_options['directory'] = args.build_directory
        if args.device == 'rust_standalone':
            import brian2_rust  # noqa: F401 -- registers the device
            device_options['engine'] = args.engine
            if args.runner is not None:
                device_options['runner'] = args.runner
        set_device(args.device, **device_options)

    seed(SEED)
    rseed(SEED)
    DATA_PATH.mkdir(parents = True, exist_ok = True)
    if args.download_mnist:
        download_mnist()
    cmds = dict(train=train, observe=observe, test=test, plot=plot)
    cmds[MODE]()


if __name__ == '__main__':
    main()
