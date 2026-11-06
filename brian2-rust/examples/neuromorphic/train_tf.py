"""External TF training, validation-only integer selection, frozen TF export.

A deliberately small shared simulator profile, distinct from the older 98.74%
MLP. Test data is never accessed by this training program.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import tensorflow as tf


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=20)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.keras.utils.set_random_seed(42)
    with np.load(a.dataset, allow_pickle=False) as data:
        x = (data['x_train'].reshape(-1, 784) >= 128).astype(np.float32)
        y = data['y_train'].astype(np.int64)
    w = tf.Variable(tf.keras.initializers.GlorotUniform(seed=42)((784, 10)))
    opt = tf.keras.optimizers.Adam(.003)
    @tf.function
    def step(inputs, labels):
        with tf.GradientTape() as tape:
            logits = inputs @ w
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        opt.apply_gradients([(tape.gradient(loss, w), w)])
    rng = np.random.default_rng(42)
    best = -1
    history = []
    for epoch in range(1, a.epochs+1):
        order = rng.permutation(54000)
        for start in range(0, len(order), 256):
            ix = order[start:start+256]
            step(x[ix], y[ix])
        raw = w.numpy()
        positive = raw-raw.min(axis=1, keepdims=True)
        # Common per-feature class offset preserves float argmax. Quantization
        # and floor can change it; selection measures the actual frozen model.
        scale = min(63/max(float(positive.max()), 1e-9), 8191/max(float(positive.sum(0).max()), 1e-9))
        q = np.floor(positive*scale).astype(np.int64)
        for threshold in (16, 32, 64):
            if q.max() > threshold or (q.sum(0)//threshold > 127).any():
                continue
            counts = x[54000:].astype(np.int64) @ q // threshold
            correct = int((counts.argmax(1) == y[54000:]).sum())
            row = dict(epoch=epoch, threshold=threshold, validation_correct=correct, validation_samples=6000)
            history.append(row)
            if correct > best:
                best = correct
                selection = row
                np.savez_compressed(a.output/'weights.npz', weights=q, threshold=np.array(threshold))
                np.savez_compressed(a.output/'float-weights.npz', weights=raw)
        print(json.dumps(dict(epoch=epoch, best_validation_correct=best)), flush=True)
    with np.load(a.output/'weights.npz', allow_pickle=False) as d:
        q, threshold = d['weights'], int(d['threshold'])
    class Frozen(tf.Module):
        def __init__(self):
            super().__init__()
            self.weights = tf.Variable(q, trainable=False, dtype=tf.int64)
        @tf.function(input_signature=[tf.TensorSpec([None,784], tf.int64, name='inputs')])
        def __call__(self, inputs):
            tf.debugging.assert_greater_equal(inputs, tf.constant(0,tf.int64))
            tf.debugging.assert_less_equal(inputs, tf.constant(1,tf.int64))
            return {'counts': tf.math.floordiv(tf.matmul(inputs,self.weights),threshold)}
    tf.saved_model.save(Frozen(), str(a.output/'saved_model'))
    report = dict(tensorflow_version=tf.__version__, architecture='binary-input bias-free 784-to-10 count classifier',
                  semantics='floor(binary_inputs @ integer_weights / threshold)', selection=selection,
                  history=history, seed=42, training_samples=54000, validation_samples=6000,
                  test_read_during_training=False, preprocessing='uint8 pixel >= 128',
                  dataset_sha256=hashlib.sha256(a.dataset.read_bytes()).hexdigest(),
                  weights_sha256=hashlib.sha256((a.output/'weights.npz').read_bytes()).hexdigest(),
                  note='Frozen quantized TF model is the conversion source, not the unquantized float model.')
    (a.output/'training.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__ == '__main__':
    main()
