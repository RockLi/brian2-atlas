"""Seeded temporal networks, dimensions, delays and duration boundaries."""
import numpy as np


def cases():
    rng=np.random.default_rng(20261013)
    result={}
    def network(nin,n,steps,index):
        def edges(pre,post):
            rows=[]
            for i in range(pre):
                for j in range(post):
                    if rng.random()<min(.3,8/post):
                        rows.append([i,j,int(rng.choice([-15,-7,-1,1,7,15])),int(rng.integers(0,8))])
            return rows or [[0,0,1,0]]
        return dict(input_size=nin,steps=steps,
            events=[(t,i) for t in range(steps) for i in range(nin) if rng.random()<.15],
            tau_ms=rng.choice([1,2,4,8,16,32],n).tolist(),
            drive=(rng.integers(-8,17,n)/4).tolist(),threshold=rng.choice([.5,1,2,4,8],n).tolist(),
            reset_v=(rng.integers(-4,5,n)/4).tolist(),
            reset='v -= threshold' if index%2 else 'v = v_reset',
            inputs=edges(nin,n),recurrent=edges(n,n))
    for i in range(32):
        result[f'random_{i:02d}']=network(int(rng.integers(1,9)),int(rng.integers(1,33)),97+i,i)
    for name,nin,n,steps in [('one_tick',1,1,1),('one_neuron',1,1,65),
                            ('maximum_dimensions',256,256,33),('maximum_duration',2,3,10000)]:
        result[name]=network(nin,n,steps,0)
    return result
