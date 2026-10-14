"""Atlas-native integer recurrent training on ten-class block-sequential MNIST."""
import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path
import time
import numpy as np
from brian2_atlas import NativeXyloTrainer, xylo_recurrent_training_plan
from brian2_atlas.neuromorphic import freeze_native_xylo
from brian2_rust.protocol import canonical_bytes

DATASET_SHA256 = '731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1'
SEED = 20261014
CODES = np.array([[int(i in pair) for i in range(5)] for pair in itertools.combinations(range(5), 2)])


def class_codes(name):
    if name == 'pairs5':
        return CODES.copy()
    if name == 'hadamard8':
        h = np.array([[1]], dtype=int)
        for _ in range(3):
            h = np.block([[h,h],[h,-h]])
        # Ten balanced codes, minimum pairwise Hamming distance four.
        return (np.concatenate([h[1:], -h[1:4]]) > 0).astype(int)
    raise ValueError('unknown class codebook')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encode(images, encoding='blocks', repeats=1):
    """Fixed, label-free spike counts. Every image position contributes to a scan.

    blocks: 49 row-major 4x4 patches from the original 28x28 image.
    rows: 14 rows of 2x2 average-pooled pixels, four pulse-density ticks per row.
    rows-counts: one pooled grayscale row per tick, with counts in 0..15.
    The network receives only the current row/patch, never the whole frame.
    """
    images = np.asarray(images)
    if images.ndim != 3 or images.shape[1:] != (28, 28) or repeats < 1:
        raise ValueError('expected N x 28 x 28 images and positive repeats')
    if encoding == 'blocks':
        x = (images.reshape(-1, 7, 4, 7, 4).transpose(0, 1, 3, 2, 4).reshape(-1, 49, 16) >= 80).astype(np.uint8)
    elif encoding == 'rows-counts':
        x = np.rint(images.reshape(-1,14,2,14,2).mean((2,4))*15/255).astype(np.uint8)
    elif encoding in ('rows', 'rows-query'):
        levels = np.rint(images.reshape(-1, 14, 2, 14, 2).mean((2, 4)) * 4 / 255).astype(int)
        t = np.arange(4)[None, None, :, None]
        v = levels[:, :, None, :]
        x = ((t + 1) * v // 4 - t * v // 4).reshape(-1, 56, 14).astype(np.uint8)
    else:
        raise ValueError('unknown encoding')
    x = np.tile(x, (1, repeats, 1))
    if x.shape[1] + 8 > 128:
        raise ValueError('encoding exceeds admitted 128-tick horizon')
    x = np.pad(x, ((0, 0), (0, 8), (0, 0)))
    if encoding in ('rows-query', 'rows-counts'):
        x = np.pad(x, ((0,0),(0,0),(0,2)))
        x[:,:,14] = 1  # Constant tonic input, independent of image and label.
        x[:,-8:,15] = 1  # Common readout query after the final image row.
    return x


def evaluate(trainer, x, y):
    correct = 0
    for i in range(0, len(x), 64):
        result = trainer.evaluate(x[i:i+64], y[i:i+64])
        correct += np.count_nonzero(np.argmax(result['logits'], axis=1) == y[i:i+64])
    return float(correct / len(y))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('dataset', 'runner', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--resume', type=Path)
    p.add_argument('--updates', type=int, default=3000)
    p.add_argument('--encoding', choices=('blocks', 'rows', 'rows-query', 'rows-counts'), default='rows')
    p.add_argument('--codebook', choices=('pairs5','hadamard8'), default='pairs5')
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--hidden', type=int, default=63)
    p.add_argument('--threshold', type=int, default=16)
    p.add_argument('--output-threshold', type=int, default=None)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--rec-init', type=float, default=.5)
    p.add_argument('--readout-ticks', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--validate-every', type=int, default=300)
    p.add_argument('--seed', type=int, default=SEED)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    source_bytes = Path(__file__).read_bytes()
    (a.output/'training-script.py').write_bytes(source_bytes)
    assert sha(a.dataset) == DATASET_SHA256
    with np.load(a.dataset, allow_pickle=False) as data:
        # Intentionally never index either official test member during training.
        images, labels = data['x_train'], data['y_train']
    x = encode(images[:54000], a.encoding, a.repeats); y = labels[:54000]
    vx = encode(images[54000:], a.encoding, a.repeats); vy = labels[54000:]
    rng = np.random.default_rng(a.seed); ni = x.shape[2]; nh = a.hidden
    codes = class_codes(a.codebook); no = codes.shape[1]
    plan = xylo_recurrent_training_plan([ni, nh, no], readout_start=x.shape[1]-a.readout_ticks,
        thresholds=(a.threshold, a.output_threshold or a.threshold), weight_scale=32., learning_rate=a.lr,
        logit_scale=12., seed=a.seed, class_codes=codes.tolist(),
        input_max=15 if a.encoding=='rows-counts' else 1)
    initial = [rng.normal(.03, .2, (ni, nh)).ravel(), rng.normal(.02, .1, (nh, no)).ravel(),
               (np.eye(nh)*a.rec_init + rng.normal(0, .015, (nh, nh))).ravel()]
    resume_receipt = None
    trainer = NativeXyloTrainer(plan, runner=a.runner, weights=initial)
    if a.resume:
        prior = json.loads((a.resume/'frozen.json').read_text())
        prior_report = json.loads((a.resume/'training.json').read_text())
        assert sha(a.resume/'frozen.json') == prior_report['frozen_file_sha256']
        assert prior_report['encoding'] == a.encoding and prior_report['repeats'] == a.repeats
        compatible = copy.deepcopy(plan)
        compatible['optimizer']['learning_rate'] = prior['plan']['optimizer']['learning_rate']
        compatible['seed'] = prior['plan']['seed']
        assert compatible == prior['plan'], 'resume changes more than the learning rate or sampling seed'
        restored = NativeXyloTrainer(prior['plan'], runner=a.runner)
        restored.restore(a.resume/'checkpoint.json')
        trainer.state = copy.deepcopy(restored.state)
        initial = [np.asarray(w).copy() for w in trainer.state['weights']]
        resume_receipt = dict(checkpoint_sha256=sha(a.resume/'checkpoint.json'),
            frozen_file_sha256=sha(a.resume/'frozen.json'), optimizer_step=trainer.state['step'],
            optimizer_moments_preserved=True, prior_validation_accuracy=prior_report['selected']['validation_accuracy'])
    start = time.monotonic(); history = []; best = -1.
    for step in range(1, a.updates+1):
        ix = rng.integers(0, len(x), a.batch_size)
        result = trainer.step(x[ix], y[ix])
        if step % 50 == 0:
            print(json.dumps(dict(step=step, loss=result['loss'], seconds=time.monotonic()-start)), flush=True)
        if step % a.validate_every and step != a.updates:
            continue
        score = evaluate(trainer, vx, vy)
        item = dict(step=step, validation_accuracy=score, loss=result['loss'])
        history.append(item); print(json.dumps(item), flush=True)
        if score > best:
            best = score; trainer.store(a.output/'checkpoint.json')
            frozen = freeze_native_xylo(trainer)
            (a.output/'frozen.json').write_bytes(canonical_bytes(frozen))
            selected = dict(item, changed_latent_weights=[int(np.count_nonzero(np.asarray(w) != old))
                for w, old in zip(trainer.state['weights'], initial)])
        report = dict(schema='atlas.sequential-mnist.training.v1', dataset_sha256=DATASET_SHA256,
            task='Ten-class block-sequential MNIST', encoding=a.encoding, repeats=a.repeats,
            steps=x.shape[1], readout_start=plan['readout_start'], sizes=plan['sizes'],
            classes=list(range(10)), class_codes=codes.tolist(), class_decoder_execution='host-fixed-integer-codebook',
            training_samples=len(y), validation_samples=len(vy), train_original_index_range=[0, 54000],
            validation_original_index_range=[54000, 60000], test_arrays_read=False,
            selected=selected, history=history, resume=resume_receipt, updates_completed=step, requested_updates=a.updates,
            selection='highest native validation accuracy, earliest tie; no test or SDK selection',
            training='native Rust integer forward, full-sequence surrogate BPTT/STE, Adam; all three banks trained',
            external_training_framework=False, seed=a.seed, arguments={k:str(v) if isinstance(v, Path) else v for k,v in vars(a).items()},
            seconds=time.monotonic()-start, script_sha256=hashlib.sha256(source_bytes).hexdigest(), runner_sha256=sha(a.runner),
            checkpoint_sha256=sha(a.output/'checkpoint.json'), frozen_file_sha256=sha(a.output/'frozen.json'))
        (a.output/'training.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
