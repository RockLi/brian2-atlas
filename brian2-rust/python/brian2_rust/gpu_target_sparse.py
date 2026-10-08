"""Bounded sparse current-event queues for ordered mutable pre pathways."""
from .metal_delays import SPARSE_BUFFER_KINDS,sparse_delay_arrays

ROLE='target-owned-sparse-synapse-pathway'
BITSET_ROLE='target-owned-bitset-synapse-pathway'
KINDS=SPARSE_BUFFER_KINDS+('active_ranks','active_counts')
BITSET_KINDS=SPARSE_BUFFER_KINDS+('bitmap_words','bitmap_offsets')


def buffer_names(q,r,*,bitset=False):
    return tuple(f'synapse/{q}/pathway/{r}/target_sparse_{kind}' for kind in (BITSET_KINDS if bitset else KINDS))


def arrays(model,plan,storage,q,r,budget,*,bitset=False):
    import numpy as np
    syn=model['definition']['synapses'][q];inst=model['instance']['synapses'][q]
    edges=len(inst['source']);targets=syn['target_count']
    if max(edges,targets)*4>budget:
        raise MemoryError('GPU target sparse queue exceeds configured memory limit')
    base=plan.buffers.index(f'synapse/{q}/pathway/{r}/delays')
    groups=sparse_delay_arrays(syn,inst,storage[base],storage[base+1],budget)
    # Current events contain at most one delivery per edge/tick. Pending entries
    # remain in their separate ordered queue and never consume this capacity.
    if bitset:
        # CSR offsets remain edge ranks; this independent prefix addresses packed
        # words. Empty targets own zero words, and every partial word is private.
        degrees=np.bincount(np.asarray(inst['target'],np.int64),minlength=targets)
        offsets=np.r_[np.uint64(0),np.cumsum(degrees//32+(degrees%32!=0),dtype=np.uint64)]
        words=int(offsets[-1])
        if words>np.iinfo(np.uint32).max or max(words,targets+1)*4>budget:
            raise MemoryError('GPU target bitmap exceeds configured memory limit')
        return groups+[np.zeros(words,np.uint32),offsets.astype(np.uint32)]
    return groups+[np.zeros(edges,np.uint32),np.zeros(targets,np.uint32)]
