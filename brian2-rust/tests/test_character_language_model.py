"""Dataset boundaries, causal encoding and perplexity for the language pilot."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

path=Path(__file__).resolve().parents[1]/'examples/neuromorphic/language_model/common.py'
spec=importlib.util.spec_from_file_location('atlas_character_lm',path)
lm=importlib.util.module_from_spec(spec)
spec.loader.exec_module(lm)


def test_fixed_vocabulary_and_normalization():
    assert len(lm.VOCAB)==len(set(lm.VOCAB))==32
    assert ''.join(lm.VOCAB[i] for i in lm.normalize('AbC:9\n?'))=='abc  \n?'


def test_contexts_are_causal_and_stay_inside_each_split():
    tokens=np.arange(200000,dtype=np.int64)%32
    for split,(start,stop) in lm.SPLITS.items():
        x,y,positions=lm.windows(tokens,split)
        assert len(y)==stop-start-lm.CONTEXT
        np.testing.assert_array_equal(x[0],tokens[start:start+lm.CONTEXT])
        np.testing.assert_array_equal(x[-1],tokens[stop-lm.CONTEXT-1:stop-1])
        np.testing.assert_array_equal(y,tokens[positions])
        changed=tokens.copy();changed[positions[0]]=(changed[positions[0]]+1)%32
        cx,cy,_=lm.windows(changed,split)
        np.testing.assert_array_equal(cx[0],x[0])
        assert cy[0]!=y[0]


def test_encoding_preserves_order_and_has_explicit_constant_input():
    x=lm.raster(np.array([[0,1,2,3],[3,2,1,0]]))
    assert x.shape==(2,16,129)
    assert np.all(x.sum(axis=2)==5)
    assert np.all(x[:,:,-1]==1)
    assert not np.array_equal(x[0],x[1])
    np.testing.assert_array_equal(np.flatnonzero(x[0,0]),[0,33,66,99,128])
    np.testing.assert_array_equal(x[:,0],x[:,-1])


@pytest.mark.parametrize('value',[np.zeros((1,3),dtype=int),np.full((1,4),32),np.full((1,4),-1),np.zeros((1,4),dtype=float)])
def test_encoding_rejects_invalid_context(value):
    with pytest.raises(ValueError):lm.raster(value)


def test_perplexity_is_exponential_mean_nll_not_mean_inverse_probability():
    probs=np.full((2,32),1/32)
    probs[0]=.5/31;probs[0,0]=.5
    probs[1]=.875/31;probs[1,1]=.125
    result=lm.metrics(np.log(probs),np.array([0,1]))
    assert result['perplexity']==pytest.approx(4.)
    assert result['bits_per_character']==pytest.approx(2.)
    np.testing.assert_allclose(lm.log_probs(np.log(probs)+1000),np.log(probs),rtol=0,atol=1e-12)
    assert lm.metrics(np.zeros((2,32)),np.array([0,1]))['perplexity']==pytest.approx(32.)


def test_baselines_do_not_fit_held_out_tokens():
    tokens=np.arange(200000,dtype=np.int64)%32
    x,y,_=lm.windows(tokens,'validation')
    before=lm.baselines(tokens,x,y)
    changed=tokens.copy();changed[180000:]=0
    assert before==lm.baselines(changed,x,y)


def test_rejects_wrong_corpus_identity(tmp_path):
    bad=tmp_path/'input.txt';bad.write_text('untrusted corpus')
    with pytest.raises(ValueError,match='SHA-256'):lm.corpus(bad)
