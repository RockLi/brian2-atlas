"""Deterministic static models and input probes; no training or test labels."""
import itertools
import numpy as np

SEED=20261013


def probes(n,upper,rng):
    rows=[np.zeros(n,int),np.full(n,upper),np.zeros(n,int)]
    for i in sorted({0,n-1}):
        row=np.zeros(n,int);row[i]=upper;rows.append(row)
    rows.extend([np.arange(n)%2*upper,(1-np.arange(n)%2)*upper])
    if upper==1 and n<=4:rows.extend(itertools.product([0,1],repeat=n))
    rows.extend(rng.integers(0,upper+1,(6,n)))
    rows.extend([np.full(n,upper),np.zeros(n,int)])
    return np.asarray(rows,dtype=int).tolist()


def cases():
    rng=np.random.default_rng(SEED);result={}
    for i in range(18):
        n=[2,3,4,8,16,32][i%6];h=1+i%5;theta=2**(1+i%6)
        w=rng.integers(0,theta+1,(n,h));w[0,0]=theta
        # Give a second hidden state a nonzero remainder when available.
        if h>1:w[0,1]=theta-1
        result[f'count_{i:02d}']=dict(profile='binary-count-v1',weights=w.tolist(),thresholds=[theta]*h,
            bias=[0]*h,readout=np.eye(h,dtype=int).tolist(),output_bias=[0]*h,inputs=probes(n,1,rng))
    for name,n,w in [('count_cap_127',127,64),('count_zero_kernel',2,0)]:
        result[name]=dict(profile='binary-count-v1',weights=np.full((n,2),w).tolist(),thresholds=[64]*2,
            bias=[0]*2,readout=[[1,0],[0,1]],output_bias=[0]*2,inputs=probes(n,1,rng))
    for i in range(24):
        n=1+i%8;h=1+i%7;o=1+i%4;shift=1+i%16
        w=rng.integers(-127,128,(n,h));w[0,0]=127
        if h>1:w[0,1]=-127
        bias=(rng.integers(-127,128,h)*2**(i%7)).tolist()
        readout=rng.integers(-127,128,(h,o)).tolist()
        out_bias=(rng.integers(-127,128,o)*2**(i%6)).tolist()
        result[f'dense_{i:02d}']=dict(profile='dense-int8-v1',weights=w.tolist(),thresholds=[2**shift]*h,
            bias=bias,readout=readout,output_bias=out_bias,inputs=probes(n,127,rng))
    # Isolate clipping and odd signed coefficients at rounding boundaries.
    result['dense_clip_round']=dict(profile='dense-int8-v1',weights=[[127,-127,1]],thresholds=[2]*3,
        bias=[0,0,-1],readout=[[127],[-127],[3]],output_bias=[7],
        inputs=[[0],[1],[2],[3],[126],[127],[0]])
    return result


def build(c):
    import brian2 as b
    b.start_scope();b.set_device('runtime');b.prefs.codegen.target='numpy'
    b.prefs.codegen.loop_invariant_optimisations=False
    x=np.asarray(c['inputs']);w=np.asarray(c['weights']);r=np.asarray(c['readout'])
    ins=[f'x{i:03d}' for i in range(w.shape[0])]
    hidden=[f'h{i:03d}' for i in range(w.shape[1])];out=[f'y{i:03d}' for i in range(r.shape[1])]
    def affine(names,coeff,bias):
        terms=[f'({int(v)})*{name}' for name,v in zip(names,coeff) if v]
        if bias:terms.append(str(int(bias)))
        def tree(t):
            if not t:return '0'
            if len(t)==1:return t[0]
            mid=len(t)//2;return '('+tree(t[:mid])+'+'+tree(t[mid:])+')'
        return tree(terms)
    lines=[]
    for i,name in enumerate(hidden):
        theta=c['thresholds'][i]
        bias=c['bias'][i]+(theta//2 if c['profile']=='dense-int8-v1' else 0)
        lines.append(f'{name}=clip(floor(({affine(ins,w[:,i],bias)})/{theta}),0,127)')
    lines += [f'{name}={affine(hidden,r[:,i],c["output_bias"][i])}' for i,name in enumerate(out)]
    g=b.NeuronGroup(len(x),'\n'.join([f'{n}:1 (constant)' for n in ins]+[f'{n}:1' for n in hidden+out]),dt=b.ms)
    for i,name in enumerate(ins):setattr(g,name,x[:,i])
    g.run_regularly('\n'.join(lines),when='start')
    return b.Network(g)
