"""Data and scoring contract for the bounded character language-model pilot."""
import hashlib
from pathlib import Path
import numpy as np

DATASET_URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
DATASET_SHA256 = '86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed'
VOCAB = " \nabcdefghijklmnopqrstuvwxyz.,'?"
CONTEXT = 4
TICKS = 16
SEED = 20261013
SPLITS = {'train': (0, 180000), 'validation': (180000, 190000), 'test': (190000, 200000)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(text):
    """Fixed vocabulary: lowercase ASCII letters; unsupported characters → space.

    No whitespace collapse, learned vocabulary or test-dependent normalization.
    """
    lookup = {c: i for i, c in enumerate(VOCAB)}
    return np.asarray([lookup.get(c, 0) for c in text.lower()], dtype=np.uint8)


def corpus(path):
    if sha(path) != DATASET_SHA256:
        raise ValueError('unexpected Tiny Shakespeare corpus SHA-256')
    tokens = normalize(Path(path).read_text(encoding='utf-8'))[:200000]
    if len(tokens) != 200000:
        raise ValueError('corpus is shorter than the declared split')
    return tokens


def windows(tokens, split):
    """All contexts and targets stay inside their split; no future token input."""
    start, stop = SPLITS[split]
    positions = np.arange(start + CONTEXT, stop)
    contexts = tokens[positions[:, None] + np.arange(-CONTEXT, 0)]
    return contexts, tokens[positions], positions


def raster(contexts):
    c = np.asarray(contexts)
    if c.ndim != 2 or c.shape[1] != CONTEXT or not np.issubdtype(c.dtype, np.integer) or np.any((c < 0) | (c >= len(VOCAB))):
        raise ValueError('expected batch × 4 integer context token IDs')
    x = np.zeros((len(c), CONTEXT * len(VOCAB) + 1), dtype=np.uint8)
    x[np.arange(len(c))[:, None], np.arange(CONTEXT) * len(VOCAB) + c] = 1
    x[:, -1] = 1  # Explicit constant spike input; not an implicit backend bias.
    return np.repeat(x[:, None, :], TICKS, axis=1)


def log_probs(logits):
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim != 2 or z.shape[1] != len(VOCAB) or not np.isfinite(z).all():
        raise ValueError('expected finite batch × vocabulary logits')
    shifted = z - z.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def metrics(logits, labels):
    lp = log_probs(logits)
    y = np.asarray(labels)
    if y.shape != (len(lp),) or not np.issubdtype(y.dtype, np.integer) or np.any((y < 0) | (y >= len(VOCAB))):
        raise ValueError('invalid next-token labels')
    nll = -lp[np.arange(len(y)), y]
    return dict(tokens=len(y), cross_entropy_nats=float(nll.mean()),
                perplexity=float(np.exp(nll.mean())), bits_per_character=float(nll.mean()/np.log(2)),
                top1_accuracy=float(np.mean(np.argmax(lp, axis=1) == y)))


def baselines(tokens, contexts, labels):
    """Add-0.1 unigram/bigram, fit only on the training split."""
    train = tokens[slice(*SPLITS['train'])]
    v = len(VOCAB)
    uni = np.bincount(train, minlength=v).astype(float) + .1
    bi = np.full((v, v), .1)
    np.add.at(bi, (train[:-1], train[1:]), 1)
    bi /= bi.sum(axis=1, keepdims=True)
    return dict(uniform=metrics(np.zeros((len(labels), v)), labels),
                unigram=metrics(np.broadcast_to(np.log(uni/uni.sum()), (len(labels), v)), labels),
                bigram=metrics(np.log(bi[contexts[:, -1]]), labels))


def evaluate(trainer, contexts, labels, batch_size=128):
    parts=[]
    for k in range(0, len(labels), batch_size):
        r=trainer.evaluate(raster(contexts[k:k+batch_size]), labels[k:k+batch_size])
        parts.append(np.asarray(r['logits']))
    return np.concatenate(parts)
