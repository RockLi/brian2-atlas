"""Explicitly corrected GeNN comparator with checked uniform delay groups."""
from gpu_stdp_compare import genn_run, workload_options
from genn_postsynaptic_barrier import patch_and_build


def run(neurons,degree,steps,output,*,prepared=None,**workload):
    options=workload_options(**workload)
    from pygenn import GeNNModel
    original=GeNNModel.build;correction={}
    def build(self,*args,**kwargs):
        result=original(self,*args,**kwargs)
        correction.update(patch_and_build(output/'project',groups=options['delay_span']))
        return result
    GeNNModel.build=build
    try:
        result,timing=genn_run(neurons,degree,steps,output,prepared=prepared,trace_mode='device',**options)
    finally:GeNNModel.build=original
    if not correction:raise RuntimeError('Corrected comparator did not rebuild its patched source')
    timing['generated_code_correction']=correction
    timing['label']='GeNN with explicit generated postsynaptic barrier correction; not stock GeNN'
    return result,timing
