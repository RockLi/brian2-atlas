"""Bounded v2 projection builders; all forward/VJP/optimizer work stays native."""


def _positive(*values):
    if any(type(v) is not int or v <= 0 for v in values):
        raise ValueError('projection dimensions must be positive integers')


def _admit(edges, max_edges):
    _positive(max_edges)
    if edges > max_edges:
        raise ValueError('projection edge budget exceeded before topology allocation')


def dense_training_projection(source_layer, target_layer, source_count, target_count,
                              *, max_edges=1_000_000):
    """Target-major edge order, source-major distinct parameter IDs.

    Equal source/target layers create same-tick spike feedback, which changes
    membrane state for the next threshold tick. No additional delay is added.
    """
    _positive(source_count, target_count)
    _admit(source_count*target_count, max_edges)
    return dict(source_layer=source_layer, target_layer=target_layer,
                parameter_count=source_count*target_count,
                sources=[i for j in range(target_count) for i in range(source_count)],
                targets=[j for j in range(target_count) for i in range(source_count)],
                parameter_ids=[i*target_count+j for j in range(target_count) for i in range(source_count)])


def conv2d_training_projection(source_layer, target_layer, input_shape, out_channels,
                               kernel_size, *, stride=1, padding=0, max_edges=1_000_000):
    """Return (projection, CHW output shape) for shared OIHW cross-correlation.

    Integer stride/padding, zero padding, no bias/dilation/pooling. Every spatial
    use of a kernel coefficient references the same parameter/optimizer slot.
    The conservative edge budget is checked before constructing index lists.
    """
    if len(input_shape) != 3:
        raise ValueError('input_shape must be (channels,height,width)')
    channels,height,width=input_shape
    kh,kw=(kernel_size,kernel_size) if type(kernel_size) is int else kernel_size
    _positive(channels,height,width,out_channels,kh,kw,stride)
    if type(padding) is not int or padding < 0:
        raise ValueError('padding must be a nonnegative integer')
    oh=(height+2*padding-kh)//stride+1
    ow=(width+2*padding-kw)//stride+1
    _positive(oh,ow)
    _admit(out_channels*oh*ow*channels*kh*kw,max_edges)
    sources=[];targets=[];ids=[]
    for oc in range(out_channels):
        for oy in range(oh):
            for ox in range(ow):
                for ic in range(channels):
                    for ky in range(kh):
                        for kx in range(kw):
                            iy=oy*stride+ky-padding;ix=ox*stride+kx-padding
                            if 0<=iy<height and 0<=ix<width:
                                sources.append((ic*height+iy)*width+ix)
                                targets.append((oc*oh+oy)*ow+ox)
                                ids.append(((oc*channels+ic)*kh+ky)*kw+kx)
    if not sources:
        raise ValueError('convolution has no input edges')
    return dict(source_layer=source_layer,target_layer=target_layer,
                parameter_count=out_channels*channels*kh*kw,
                sources=sources,targets=targets,parameter_ids=ids),(out_channels,oh,ow)


def validate_projections(sizes, projections, *, allow_empty=False):
    if not 1<=len(projections)<=256:
        raise ValueError('invalid projection count')
    counts=[];topology_bytes=0
    fields={'source_layer','target_layer','parameter_count','sources','targets','parameter_ids'}
    for p in projections:
        if set(p)!=fields:
            raise ValueError('invalid projection fields')
        source=p['source_layer'];target=p['target_layer'];count=p['parameter_count']
        if (type(source) is not int or not 0<=source<len(sizes) or
                type(target) is not int or not 0<target<len(sizes)):
            raise ValueError('invalid projection layer endpoint')
        _positive(count)
        edges=len(p['sources'])
        if (not edges and not allow_empty) or len(p['targets'])!=edges or len(p['parameter_ids'])!=edges:
            raise ValueError('invalid projection edge/parameter shape')
        for values,limit in ((p['sources'],sizes[source]),(p['targets'],sizes[target]),(p['parameter_ids'],count)):
            if any(type(v) is not int or not 0<=v<limit for v in values):
                raise ValueError('projection index outside declared domain')
        counts.append(count);topology_bytes+=edges*24+128
    return counts,topology_bytes
