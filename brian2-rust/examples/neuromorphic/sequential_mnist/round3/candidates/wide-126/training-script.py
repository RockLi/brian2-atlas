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


def deskew_images(images):
    """Remove horizontal slant using image moments and bilinear resampling.

    Pure per-image preprocessing; no labels or dataset-fitted parameters.
    Blank images and degenerate vertical moments are left unchanged.
    """
    a = np.asarray(images, dtype=np.float64)
    if a.ndim != 3 or a.shape[1:] != (28, 28):
        raise ValueError('expected N x 28 x 28 images')
    result = np.empty_like(a)
    grid_y, grid_x = np.mgrid[:28, :28]
    for start in range(0, len(a), 512):
        v = a[start:start+512]; mass = v.sum((1,2))
        safe = np.maximum(mass, 1.)
        cx = (v*grid_x).sum((1,2))/safe; cy = (v*grid_y).sum((1,2))/safe
        dx = grid_x[None]-cx[:,None,None]; dy = grid_y[None]-cy[:,None,None]
        yy = (v*dy*dy).sum((1,2)); xy = (v*dx*dy).sum((1,2))
        slope = np.divide(xy, yy, out=np.zeros_like(xy), where=yy>1e-6)
        source_x = grid_x[None]+slope[:,None,None]*(grid_y[None]-cy[:,None,None])
        left = np.floor(source_x).astype(int); frac = source_x-left
        row = np.arange(len(v))[:,None,None]
        lo = v[row, grid_y, np.clip(left,0,27)] * ((left>=0)&(left<28))
        hi = v[row, grid_y, np.clip(left+1,0,27)] * ((left+1>=0)&(left+1<28))
        result[start:start+len(v)] = lo*(1-frac)+hi*frac
    return result


def translate_images(images, rng, radius):
    """Training-only random integer translations with zero-filled boundaries."""
    images = np.asarray(images)
    if not 0 <= radius <= 3:
        raise ValueError('translation radius must be in 0..3')
    if radius == 0:
        return images.copy()
    result = np.zeros_like(images)
    for i, (dy, dx) in enumerate(rng.integers(-radius, radius+1, (len(images),2))):
        sy = slice(max(0,-dy), min(28,28-dy)); sx = slice(max(0,-dx), min(28,28-dx))
        ty = slice(max(0,dy), min(28,28+dy)); tx = slice(max(0,dx), min(28,28+dx))
        result[i,ty,tx] = images[i,sy,sx]
    return result


def affine_images(images, rng, degrees):
    """Training-only rotation and axis scaling about the image center.

    Rotation is uniform in +/- degrees; each axis scale is independently
    uniform in 1 +/- degrees/100. Bilinear sampling uses a zero border.
    """
    a=np.asarray(images,dtype=np.float64)
    if a.ndim!=3 or a.shape[1:]!=(28,28) or not np.isfinite(degrees) or not 0<=degrees<=15:
        raise ValueError('expected N x 28 x 28 images and affine degrees in 0..15')
    if degrees==0:
        return a.copy()
    angle=np.deg2rad(rng.uniform(-degrees,degrees,len(a)))[:,None,None]
    scale=rng.uniform(1-degrees/100,1+degrees/100,(len(a),2))
    yy,xx=np.mgrid[:28,:28];x=xx-13.5;y=yy-13.5
    sx=(np.cos(angle)*x+np.sin(angle)*y)/scale[:,0,None,None]+13.5
    sy=(-np.sin(angle)*x+np.cos(angle)*y)/scale[:,1,None,None]+13.5
    x0=np.floor(sx).astype(int);y0=np.floor(sy).astype(int)
    fx=sx-x0;fy=sy-y0;batch=np.arange(len(a))[:,None,None]
    result=np.zeros_like(a)
    for dx,dy,weight in ((0,0,(1-fx)*(1-fy)),(1,0,fx*(1-fy)),(0,1,(1-fx)*fy),(1,1,fx*fy)):
        xi=x0+dx;yi=y0+dy;valid=(xi>=0)&(xi<28)&(yi>=0)&(yi<28)
        result+=a[batch,np.clip(yi,0,27),np.clip(xi,0,27)]*weight*valid
    return result


def elastic_images(images, rng, amplitude, sigma=4.):
    """Training-only smooth displacement, in pixels, with zero-filled sampling.

    Each independent x/y field has zero spatial mean and the requested RMS.
    Gaussian smoothing uses reflected boundaries; image sampling uses zeros.
    No image labels or dataset-fitted parameters enter this transform.
    """
    from scipy.ndimage import gaussian_filter, map_coordinates
    a=np.asarray(images,dtype=np.float64)
    if a.ndim!=3 or a.shape[1:]!=(28,28):
        raise ValueError('expected N x 28 x 28 images')
    if not np.isfinite(amplitude) or not 0<=amplitude<=2 or not np.isfinite(sigma) or not 1<=sigma<=8:
        raise ValueError('elastic RMS must be in 0..2 pixels and sigma in 1..8')
    if amplitude==0:
        return a.copy()
    noise=rng.standard_normal((len(a),2,28,28))
    field=gaussian_filter(noise,sigma=(0,0,sigma,sigma),mode='reflect')
    field-=field.mean((2,3),keepdims=True)
    rms=np.sqrt((field*field).mean((2,3),keepdims=True))
    field=np.divide(field,rms,out=np.zeros_like(field),where=rms>1e-12)*amplitude
    yy,xx=np.mgrid[:28,:28]
    return np.stack([map_coordinates(row,[yy+f[0],xx+f[1]],order=1,mode='grid-constant',cval=0.,prefilter=False)
        for row,f in zip(a,field)])


def gradient_cells(images):
    """Fixed unsigned-gradient cell descriptors; no learned preprocessing.

    Sixteen 7x7 cells, eight interpolated orientation bins normalized per cell,
    mean intensity, relative edge strength, and explicit cell row/column.
    """
    result = np.zeros((len(images),16,14), dtype=np.uint8)
    for start in range(0,len(images),512):
        a = np.asarray(images[start:start+512],dtype=np.float64)
        gy, gx = np.gradient(a,axis=(1,2))
        mag = np.hypot(gx,gy)
        angle = (np.arctan2(gy,gx) % np.pi) * (8/np.pi)
        lo = np.floor(angle).astype(int) % 8; fraction = angle-np.floor(angle)
        energy = np.zeros((len(a),16)); hist = np.zeros((len(a),16,8))
        for row in range(4):
            for col in range(4):
                k = row*4+col; sy=slice(row*7,(row+1)*7); sx=slice(col*7,(col+1)*7)
                m=mag[:,sy,sx]; l=lo[:,sy,sx]; f=fraction[:,sy,sx]
                energy[:,k]=m.sum((1,2))
                for b in range(8):
                    hist[:,k,b]=(m*((l==b)*(1-f)+((l+1)%8==b)*f)).sum((1,2))
                result[start:start+len(a),k,8]=np.rint(a[:,sy,sx].mean((1,2))*15/255).astype(np.uint8)
                result[start:start+len(a),k,12]=3*(row+1)
                result[start:start+len(a),k,13]=3*(col+1)
        norm=hist.sum(2,keepdims=True)
        normalized=np.divide(hist,norm,out=np.zeros_like(hist),where=norm>0)
        result[start:start+len(a),:,:8]=np.rint(15*np.sqrt(normalized)).astype(np.uint8)
        peak=energy.max(1,keepdims=True)
        result[start:start+len(a),:,9]=np.rint(15*np.divide(energy,peak,out=np.zeros_like(energy),where=peak>0)).astype(np.uint8)
    return result


def encode(images, encoding='blocks', repeats=1, *, deskew=False, query_ticks=8):
    """Fixed, label-free spike counts. Every image position contributes to a scan.

    blocks: 49 row-major 4x4 patches from the original 28x28 image.
    rows: 14 rows of 2x2 average-pooled pixels, four pulse-density ticks per row.
    rows-counts: one pooled grayscale row per tick, with counts in 0..15.
    The network receives only the current row/patch, never the whole frame.
    """
    images = np.asarray(images)
    if images.ndim != 3 or images.shape[1:] != (28, 28) or repeats < 1 or not 1 <= query_ticks <= 64:
        raise ValueError('expected N x 28 x 28 images and positive repeats')
    if deskew:
        images = deskew_images(images)
    if encoding == 'gradient-counts':
        x = gradient_cells(images)
    elif encoding == 'blocks':
        x = (images.reshape(-1, 7, 4, 7, 4).transpose(0, 1, 3, 2, 4).reshape(-1, 49, 16) >= 80).astype(np.uint8)
    elif encoding in ('rows-counts', 'rows-columns-counts', 'rows-full-counts', 'rows-interleaved-counts', 'rows-position-counts'):
        if encoding == 'rows-interleaved-counts':
            # Even columns then odd columns per row: retain all 784 pixels.
            levels = images.astype(np.float64).reshape(-1,28,14,2).transpose(0,1,3,2).reshape(-1,56,14)
        elif encoding == 'rows-full-counts':
            levels = images.reshape(-1,28,14,2).mean(3)
        else:
            levels = images.reshape(-1,14,2,14,2).mean((2,4))
            if encoding == 'rows-columns-counts':
                levels = np.concatenate([levels, levels.transpose(0,2,1)], axis=1)
        x = np.rint(levels*15/255).astype(np.uint8)
    elif encoding in ('rows', 'rows-query'):
        levels = np.rint(images.reshape(-1, 14, 2, 14, 2).mean((2, 4)) * 4 / 255).astype(int)
        t = np.arange(4)[None, None, :, None]
        v = levels[:, :, None, :]
        x = ((t + 1) * v // 4 - t * v // 4).reshape(-1, 56, 14).astype(np.uint8)
    else:
        raise ValueError('unknown encoding')
    x = np.tile(x, (1, repeats, 1))
    if x.shape[1] + query_ticks > 128:
        raise ValueError('encoding exceeds admitted 128-tick horizon')
    x = np.pad(x, ((0, 0), (0, query_ticks), (0, 0)))
    if encoding == 'rows-query' or encoding.endswith('-counts'):
        x = np.pad(x, ((0,0),(0,0),(0,2)))
        x[:,:,14] = 1  # Constant tonic input, independent of image and label.
        x[:,-query_ticks:,15] = 1  # Common readout query after the final image row.
        if encoding == 'rows-position-counts':
            x[:,:-query_ticks,15] = np.tile(np.arange(1,15,dtype=np.uint8),repeats)
            x[:,-query_ticks:,15] = 15
    return x


def evaluate(trainer, x, y):
    correct = 0
    for i in range(0, len(x), 64):
        result = trainer.evaluate(x[i:i+64], y[i:i+64])
        correct += np.count_nonzero(np.argmax(result['logits'], axis=1) == y[i:i+64])
    return float(correct / len(y))


def warm_weights(directories, plan, runner, multiplier=1.):
    """Average compatible native latent weights, with fresh optimizer state."""
    from brian2_rust.native_xylo_conversion import validate_frozen
    banks=[];parents=[]
    for directory in directories:
        directory=Path(directory)
        prior=json.loads((directory/'frozen.json').read_text())
        report=json.loads((directory/'training.json').read_text())
        validate_frozen(prior)
        assert sha(directory/'frozen.json')==report['frozen_file_sha256']
        assert sha(directory/'checkpoint.json')==report['checkpoint_sha256']
        for key in ('sizes','class_codes','weight_scale'):
            if prior['plan'][key]!=plan[key]:
                raise ValueError('incompatible warm-start '+key)
        restored=NativeXyloTrainer(prior['plan'],runner=runner)
        restored.restore(directory/'checkpoint.json')
        assert freeze_native_xylo(restored)['sha256']==prior['sha256']
        banks.append([np.asarray(w).copy() for w in restored.state['weights']])
        parents.append(dict(run=str(directory),checkpoint_sha256=sha(directory/'checkpoint.json'),
            frozen_file_sha256=sha(directory/'frozen.json'),validation_accuracy=report['selected']['validation_accuracy']))
    if not banks or not np.isfinite(multiplier) or multiplier<=0:
        raise ValueError('positive multiplier and at least one warm-start source required')
    initial=[np.mean([b[k] for b in banks],axis=0)*multiplier for k in range(3)]
    if not all(np.all((w*plan['weight_scale']>=-128)&(w*plan['weight_scale']<=127)) for w in initial):
        raise ValueError('scaled warm start exceeds signed int8 range')
    receipt=dict(kind='averaged-warm-start' if len(banks)>1 else 'warm-start',weight_multiplier=multiplier,
        parents=parents,optimizer_moments_preserved=False,aggregation='equal mean of latent weights before native retraining')
    return initial,receipt


def recurrent_group_mask(hidden, groups):
    """Fixed block-diagonal connectivity; at most 63 recurrent targets per cell."""
    if not 1<=hidden<=128 or not 1<=groups<=hidden or hidden%groups or hidden//groups>63:
        raise ValueError('equal recurrent groups must each contain at most 63 of 1..128 neurons')
    ids=np.arange(hidden)//(hidden//groups)
    return ids[:,None]==ids[None,:]


def project_recurrence(trainer, mask):
    """Projected Adam: inactive weights and their moments stay exactly zero."""
    for key in ('weights','first_moment','second_moment'):
        values=np.asarray(trainer.state[key][2]).reshape(mask.shape).copy()
        values[~mask]=0
        trainer.state[key][2]=values.ravel().tolist()


def widen_weights(frozen, plan, groups, jitter, rng):
    """Duplicate a frozen hidden bank into independent recurrent groups.

    Readout weights are divided between groups. Small independent input and
    recurrent perturbations break symmetry before native joint retraining.
    This initialization does not claim prediction equivalence to its parent.
    """
    from brian2_rust.native_xylo_conversion import validate_frozen
    ni,nh,no=validate_frozen(frozen);target=plan['sizes']
    if groups<2 or target!=[ni,nh*groups,no] or plan['class_codes']!=frozen['plan']['class_codes']:
        raise ValueError('widening requires replicated hidden groups with the same inputs and codebook')
    if plan['threshold']!=frozen['plan']['threshold'] or plan['weight_scale']!=frozen['plan']['weight_scale']:
        raise ValueError('widening requires unchanged thresholds and weight scale')
    if not np.isfinite(jitter) or not 0<=jitter<=.1:
        raise ValueError('widening jitter must be in 0..0.1 latent units')
    mask=recurrent_group_mask(target[1],groups)
    wi,wo,wr=[np.asarray(w,dtype=float)/plan['weight_scale'] for w in frozen['target_weights']]
    wi=wi.reshape(ni,nh);wo=wo.reshape(nh,no);wr=wr.reshape(nh,nh)
    expanded_in=np.concatenate([wi+rng.normal(0,jitter,wi.shape) for _ in range(groups)],axis=1)
    expanded_out=np.tile(wo/groups,(groups,1))
    expanded_rec=np.zeros(mask.shape)
    for g in range(groups):expanded_rec[g*nh:(g+1)*nh,g*nh:(g+1)*nh]=wr+rng.normal(0,jitter,wr.shape)
    result=[w.ravel() for w in (expanded_in,expanded_out,expanded_rec)]
    if not all(np.all((w*plan['weight_scale']>=-128)&(w*plan['weight_scale']<=127)) for w in result):
        raise ValueError('widened initialization exceeds signed int8 range')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('dataset', 'runner', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    start_group = p.add_mutually_exclusive_group()
    start_group.add_argument('--resume', type=Path)
    start_group.add_argument('--widen-from', type=Path, help='initialize independent recurrent groups from a frozen native model')
    start_group.add_argument('--warm-start', type=Path, help='reuse latent weights, reset optimizer; permit a new encoder or readout window')
    p.add_argument('--warm-average', type=Path, nargs='+', default=[], help='additional native checkpoints to average with warm-start; reset Adam before retraining')
    p.add_argument('--warm-weight-scale', type=float, default=1., help='multiply warm-start latent weights; reject overflow instead of clipping')
    p.add_argument('--deskew', action='store_true')
    p.add_argument('--query-ticks', type=int, default=8)
    p.add_argument('--augmentation-shift', type=int, default=0)
    p.add_argument('--augmentation-affine', type=float, default=0., help='training-only rotation degrees; also scales each axis by up to degrees/100')
    p.add_argument('--augmentation-elastic', type=float, default=0., help='training-only smooth displacement RMS in pixels')
    p.add_argument('--elastic-sigma', type=float, default=4.)
    p.add_argument('--evaluation-round', type=int, default=3)
    p.add_argument('--prior-test-accuracy', type=float, default=.9728)
    p.add_argument('--logit-scale', type=float, default=12.)
    p.add_argument('--surrogate-slope', type=float, default=3.)
    p.add_argument('--updates', type=int, default=3000)
    p.add_argument('--encoding', choices=('blocks', 'rows', 'rows-query', 'rows-counts', 'rows-columns-counts', 'rows-full-counts', 'rows-interleaved-counts', 'rows-position-counts', 'gradient-counts'), default='rows')
    p.add_argument('--codebook', choices=('pairs5','hadamard8'), default='pairs5')
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--hidden', type=int, default=63)
    p.add_argument('--recurrent-groups', type=int, default=1)
    p.add_argument('--widen-jitter', type=float, default=.03)
    p.add_argument('--threshold', type=int, default=16)
    p.add_argument('--output-threshold', type=int, default=None)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--rec-init', type=float, default=.5)
    p.add_argument('--readout-ticks', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--validate-every', type=int, default=300)
    p.add_argument('--seed', type=int, default=SEED)
    a = p.parse_args()
    if min(a.updates,a.validate_every,a.batch_size,a.readout_ticks,a.repeats)<1:
        p.error('updates, validation interval, batch size, readout ticks and repeats must be positive')
    if a.warm_average and not a.warm_start:
        p.error('warm-average requires warm-start')
    if not np.isfinite(a.warm_weight_scale) or a.warm_weight_scale<=0 or (a.warm_weight_scale!=1 and not a.warm_start):
        p.error('positive warm-weight-scale requires warm-start')
    if not np.isfinite(a.augmentation_elastic) or not 0<=a.augmentation_elastic<=2 or not np.isfinite(a.elastic_sigma) or not 1<=a.elastic_sigma<=8:
        p.error('elastic RMS must be in 0..2 pixels and sigma in 1..8')
    if a.evaluation_round<1 or not 0<=a.prior_test_accuracy<=1:
        p.error('invalid evaluation provenance')
    if not np.isfinite(a.augmentation_affine) or not 0<=a.augmentation_affine<=15:
        p.error('augmentation affine degrees must be in 0..15')
    if not 0<=a.augmentation_shift<=3:
        p.error('augmentation shift must be in 0..3 pixels')
    try:mask=recurrent_group_mask(a.hidden,a.recurrent_groups)
    except ValueError as error:p.error(str(error))
    a.output.mkdir(parents=True, exist_ok=False)
    source_bytes = Path(__file__).read_bytes()
    (a.output/'training-script.py').write_bytes(source_bytes)
    assert sha(a.dataset) == DATASET_SHA256
    with np.load(a.dataset, allow_pickle=False) as data:
        # Intentionally never index either official test member during training.
        images, labels = data['x_train'], data['y_train']
    x = np.concatenate([encode(images[i:min(i+512,54000)], a.encoding, a.repeats,
        deskew=a.deskew, query_ticks=a.query_ticks) for i in range(0,54000,512)]); y = labels[:54000]
    vx = np.concatenate([encode(images[i:i+512], a.encoding, a.repeats,
        deskew=a.deskew, query_ticks=a.query_ticks) for i in range(54000,60000,512)]); vy = labels[54000:]
    rng = np.random.default_rng(a.seed); ni = x.shape[2]; nh = a.hidden
    codes = class_codes(a.codebook); no = codes.shape[1]
    plan = xylo_recurrent_training_plan([ni, nh, no], readout_start=x.shape[1]-a.readout_ticks,
        thresholds=(a.threshold, a.threshold if a.output_threshold is None else a.output_threshold), weight_scale=32., learning_rate=a.lr,
        logit_scale=a.logit_scale, seed=a.seed, class_codes=codes.tolist(),
        input_max=15 if a.encoding.endswith('-counts') else 1)
    plan['surrogate']['slope'] = a.surrogate_slope
    initial = [rng.normal(.03, .2, (ni, nh)).ravel(), rng.normal(.02, .1, (nh, no)).ravel(),
               (np.eye(nh)*a.rec_init + rng.normal(0, .015, (nh, nh))).ravel()]
    initial[2]=(initial[2].reshape(nh,nh)*mask).ravel()
    resume_receipt = None
    trainer = NativeXyloTrainer(plan, runner=a.runner, weights=initial)
    if a.widen_from:
        parent=json.loads((a.widen_from/'frozen.json').read_text())
        parent_report=json.loads((a.widen_from/'training.json').read_text())
        assert sha(a.widen_from/'frozen.json')==parent_report['frozen_file_sha256']
        initial=widen_weights(parent,plan,a.recurrent_groups,a.widen_jitter,rng)
        trainer=NativeXyloTrainer(plan,runner=a.runner,weights=initial)
        resume_receipt=dict(kind='widen-frozen',source=str(a.widen_from),frozen_file_sha256=sha(a.widen_from/'frozen.json'),
            groups=a.recurrent_groups,jitter=a.widen_jitter,optimizer_moments_preserved=False,source_weights='quantized native weights')
    if a.warm_start:
        initial,resume_receipt=warm_weights([a.warm_start,*a.warm_average],plan,a.runner,a.warm_weight_scale)
        trainer=NativeXyloTrainer(plan,runner=a.runner,weights=initial)
    if a.resume:
        prior = json.loads((a.resume/'frozen.json').read_text())
        prior_report = json.loads((a.resume/'training.json').read_text())
        assert sha(a.resume/'frozen.json') == prior_report['frozen_file_sha256']
        assert sha(a.resume/'checkpoint.json') == prior_report['checkpoint_sha256']
        assert prior_report['encoding'] == a.encoding and prior_report['repeats'] == a.repeats
        assert prior_report.get('deskew', False) == a.deskew
        assert prior_report.get('query_ticks', 8) == a.query_ticks
        assert prior_report.get('recurrent_groups',1)==a.recurrent_groups
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
    if a.recurrent_groups>1:
        project_recurrence(trainer,mask)
        initial=[np.asarray(w).copy() for w in trainer.state['weights']]
    start = time.monotonic(); history = []; best = -1.
    for step in range(1, a.updates+1):
        ix = rng.integers(0, len(x), a.batch_size)
        if a.augmentation_shift or a.augmentation_affine or a.augmentation_elastic:
            raw = deskew_images(images[ix]) if a.deskew else images[ix]
            raw = affine_images(raw, rng, a.augmentation_affine)
            raw = elastic_images(raw, rng, a.augmentation_elastic, a.elastic_sigma)
            batch = encode(translate_images(raw, rng, a.augmentation_shift), a.encoding, a.repeats, query_ticks=a.query_ticks)
        else:
            batch = x[ix]
        result = trainer.step(batch, y[ix])
        if a.recurrent_groups>1:project_recurrence(trainer,mask)
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
            deskew=a.deskew, query_ticks=a.query_ticks, augmentation_shift=a.augmentation_shift,
            augmentation_affine=a.augmentation_affine, augmentation_elastic=a.augmentation_elastic, elastic_sigma=a.elastic_sigma,
            evaluation_round=a.evaluation_round, prior_official_test_accuracy_known=a.prior_test_accuracy,
            steps=x.shape[1], readout_start=plan['readout_start'], sizes=plan['sizes'],
            recurrent_groups=a.recurrent_groups,active_weight_count=ni*nh+nh*no+int(mask.sum()),
            masked_recurrent_weights=int(mask.size-mask.sum()),max_recurrent_fanout=int(mask.sum(1).max()),
            classes=list(range(10)), class_codes=codes.tolist(), class_decoder_execution='host-fixed-integer-codebook',
            training_samples=len(y), validation_samples=len(vy), train_original_index_range=[0, 54000],
            validation_original_index_range=[54000, 60000], test_arrays_read=False,
            selected=selected, history=history, resume=resume_receipt, updates_completed=step, requested_updates=a.updates,
            selection='highest native validation accuracy, earliest tie; no test or SDK selection',
            training='native Rust integer forward, full-sequence surrogate BPTT/STE, Adam; all three banks trained',
            external_training_framework=False, seed=a.seed, arguments={k:([str(p) for p in v] if k=='warm_average' else str(v) if isinstance(v, Path) else v) for k,v in vars(a).items()},
            seconds=time.monotonic()-start, script_sha256=hashlib.sha256(source_bytes).hexdigest(), runner_sha256=sha(a.runner),
            checkpoint_sha256=sha(a.output/'checkpoint.json'), frozen_file_sha256=sha(a.output/'frozen.json'))
        (a.output/'training.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
