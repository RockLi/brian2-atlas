"""Bitwise packing conformance at float32 rounding and ABI boundaries."""
import struct
import time
import json

import numpy as np
import pytest

from brian2_rust.gpu_types import pack
from brian2_rust.plan import PlanValidationError
from gpu_packing_benchmark import scalar_pack


@pytest.mark.parametrize('dtype',['f32','f64'])
@pytest.mark.parametrize('size',[32,64])
def test_wire_rounding_boundaries_and_random_bits(tmp_path,dtype,size):
    rng=np.random.default_rng(13579)
    bits=rng.integers(0,2**32,size=24000,dtype=np.uint32)
    boundaries=np.array([sign|(e<<23)|m for sign in (0,2**31)
        for e in range(255) for m in (0,1,0x3fffff,0x7ffffe,0x7fffff)],np.uint32)
    bits=np.concatenate((bits,boundaries))
    values=bits.view(np.float32);values=values[np.isfinite(values)]
    if size==64:
        lower=values.astype(np.float64)
        with np.errstate(over='ignore'):
            upper=np.nextafter(values,np.float32(np.inf)).astype(np.float64)
        middle=(lower+upper)*.5;middle=middle[np.isfinite(middle)]
        values=np.concatenate((lower,middle,np.nextafter(middle,-np.inf),np.nextafter(middle,np.inf)))
    wire=[struct.pack('>f' if size==32 else '>d',v).hex() for v in values]
    expected=scalar_pack(wire,dtype);actual=pack(wire,dtype)
    np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
    assert actual.flags.owndata and actual.dtype==np.float32
    np.savez_compressed(tmp_path/'packing-results.npz',input=values,actual=actual.view(np.uint32),reference=expected.view(np.uint32))


@pytest.mark.parametrize('values',[[],['80000000'],['3f800000','8000000000000000'],
    [1.,'3ff0000000000000',-0.],('00000000','80000000')])
def test_empty_mixed_width_and_mixed_values(values):
    np.testing.assert_array_equal(pack(values,'f64').view(np.uint32),scalar_pack(values,'f64').view(np.uint32))


@pytest.mark.parametrize('values',[['7f800000'],['ff800000'],['7fc00001'],['7f800001'],
    ['7ff0000000000000'],['7fefffffffffffff'],['47effffff0000000'],[np.inf],[np.nan]])
def test_nonfinite_or_unrepresentable_values_reject(values):
    for fn in (pack,scalar_pack):
        with pytest.raises(PlanValidationError,match='finite float32'):fn(values,'f32')


@pytest.mark.parametrize('values',[['3f80000z'],['3f80    ','0000    '],['3f800000000000'],[''],['123','456']])
def test_malformed_element_boundaries_do_not_merge(values):
    for fn in (pack,scalar_pack):
        with pytest.raises((ValueError,struct.error)):fn(values,'f32')


@pytest.mark.parametrize('dtype',['<f4','>f4','<f8','>f8','i8','u8','bool'])
def test_numeric_initializer_owns_independent_contiguous_storage(dtype):
    source=np.arange(12).astype(dtype)[::2];source.flags.writeable=False
    a=pack(source,'f64');b=pack(source,'f64')
    np.testing.assert_array_equal(a.view(np.uint32),scalar_pack(source,'f64').view(np.uint32))
    assert a.flags.c_contiguous and a.flags.owndata
    assert not np.shares_memory(source,a) and not np.shares_memory(a,b)
    a[:]=123
    np.testing.assert_array_equal(b,source.astype(np.float32))


@pytest.mark.parametrize('dtype',['bool','i32','u32','i64','u64'])
def test_integer_word_planes_unchanged(dtype):
    width=16 if dtype.endswith('64') else 8
    values=['0'*width,'f'*width,'8'+'0'*(width-1),'7f800001'*(width//8)]
    np.testing.assert_array_equal(pack(values,dtype).view(np.uint32),scalar_pack(values,dtype).view(np.uint32))


def test_large_wire_pair_measurement(tmp_path):
    values=np.linspace(-32,32,131072,dtype=np.float64)
    wire=[struct.pack('>d',v).hex() for v in values]
    rows=[];rng=np.random.default_rng(2718)
    for pair in range(-1,7):
        for policy in rng.permutation(['scalar','bulk']):
            fn=scalar_pack if policy=='scalar' else pack
            start=time.perf_counter();actual=fn(wire,'f64');elapsed=time.perf_counter()-start
            np.testing.assert_array_equal(actual.view(np.uint32),values.astype(np.float32).view(np.uint32))
            rows.append(dict(pair=pair,policy=str(policy),seconds=elapsed))
    (tmp_path/'packing-timing.json').write_text(json.dumps(rows,indent=2)+'\n')
