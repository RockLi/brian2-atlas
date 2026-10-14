"""Compare native Atlas with stock Rockpool/JAX DynapSim circuit dynamics."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
import numpy as np
import brian2 as b
from brian2_atlas import export_network
from brian2_atlas.neuromorphic import export_ir
from brian2_rust.dynapse_model import build,STATES
from brian2_rust.results import load_results

# Declared before SDK comparisons: float32 SDK vs reference-f64 Atlas.
RTOL=2e-4
ATOL={'iahp':1e-17,'imem':1e-17,'isyn':1e-17,'vmem':2e-6}


def cases():
    rng=np.random.default_rng(20261013);result={}
    for name,raster,params in [
        ('silent',np.zeros((128,1)),{}),('constant',np.ones((128,1)),{}),
        ('one_tick',np.ones((1,1)),{}),('zero_weight',np.ones((128,1)),{'weight':0}),
        ('adaptation',np.ones((512,1)),{'Iw_ahp':2.**-24}),
        ('bias_only',np.zeros((128,1)),{'Idc':2.**-28}),
        ('refractory',np.ones((256,1)),{'weight':4,'Iref':2.**-31}),
        ('eight_neurons',rng.integers(0,2,(512,8)),{'weight':[.5,1,2,4,.5,1,2,4]})]:
        result[name]=dict(raster=np.asarray(raster,int).tolist(),parameters=params)
    for i in range(8):
        n=1+i%4
        result[f'random_{i:02d}']=dict(raster=rng.integers(0,2,(97+i,n)).tolist(),parameters={
            'weight':rng.choice([.5,1,2,4],n).tolist(),
            'Ispkthr':rng.choice([2.**-26,2.**-25,2.**-24],n).tolist(),
            'Iw_ahp':rng.choice([2.**-40,2.**-34,2.**-32,2.**-24],n).tolist(),
            'Iref':rng.choice([2.**-31,2.**-30,2.**-29],n).tolist()})
    return result


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('output','runner','sdk-python'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output=a.output.absolute();a.output.mkdir(parents=True,exist_ok=False)
    bundles=[]
    for name,c in cases().items():
        dest=a.output/name;dest.mkdir();source=dest/'source.json';steps=len(c['raster'])
        model=export_network(build(**c),steps*b.ms,source)
        bundle=export_ir(source,dest/'converted',target='dynapse',profile='dynapse-dpi-v1',runner=a.runner)
        proc=subprocess.run([str(a.runner.absolute()),str(source),str(dest/'atlas')],capture_output=True,text=True)
        if proc.returncode:raise RuntimeError(proc.stderr)
        result=load_results(model,dest/'atlas')['populations'][0]
        values={k:result['trace'][k] for k in STATES}
        values['spikes']=np.column_stack((result['spike_ticks'],result['indices']))
        np.savez_compressed(dest/'atlas.npz',**values)
        bundles.append(dest/'converted/bundle.json');print(name,'Atlas complete',flush=True)
    runtime=Path(__file__).resolve().parents[3]/'python/brian2_rust/neuromorphic_runtime.py'
    env={**os.environ,'PYTHONPATH':'','PYTHONNOUSERSITE':'1','JAX_PLATFORMS':'cpu'}
    proc=subprocess.run([str(a.sdk_python.absolute()),str(Path(__file__).with_name('sdk_worker.py')),
        '--runtime',str(runtime),'--bundles',*map(str,bundles)],env=env,capture_output=True,text=True,timeout=300)
    (a.output/'sdk.log').write_text(proc.stdout+proc.stderr)
    if proc.returncode:raise RuntimeError(proc.stderr[-4000:])
    reports=[]
    for path in bundles:
        dest=path.parent.parent;atlas=np.load(dest/'atlas.npz');sdk=np.load(path.parent/'sdk.npz')
        errors={}
        for key in ATOL:
            np.testing.assert_allclose(sdk[key],atlas[key],rtol=RTOL,atol=ATOL[key],err_msg=dest.name+': '+key)
            errors[key]=float(np.max(np.abs(sdk[key]-atlas[key])))
        np.testing.assert_array_equal(sdk['q'],atlas['q'],err_msg=dest.name)
        np.testing.assert_array_equal(sdk['spikes'],atlas['spikes'],err_msg=dest.name)
        np.testing.assert_allclose(sdk['timer_ref_final'],atlas['timer_ref'][-1],rtol=RTOL,atol=1e-9)
        execution=json.loads((path.parent/'execution.json').read_text())
        assert execution['result_sha256']==sha(path.parent/'sdk.npz') and execution['bundle_sha256']==sha(path)
        reports.append(dict(case=dest.name,steps=atlas['q'].shape[0],neurons=atlas['q'].shape[1],
            trace_values=sum(sdk[k].size for k in (*ATOL,'q')),spike_events=len(sdk['spikes']),
            spike_count=int(sdk['q'].sum()),max_absolute_error=errors,spike_differences=0,
            source_model_sha256=json.loads(path.read_text())['source_model_sha256'],
            atlas_result_sha256=sha(dest/'atlas.npz'),sdk=execution))
    assert np.load(a.output/'silent/converted/sdk.npz')['q'].sum()==0
    assert np.load(a.output/'constant/converted/sdk.npz')['q'].sum()>0
    assert np.load(a.output/'adaptation/converted/sdk.npz')['iahp'].max()>2.**-40
    report=dict(schema='atlas.dynapse.audit.v1',passed=True,cases=reports,rtol=RTOL,atol=ATOL,
        timer_ref_final_atol=1e-9,runner_sha256=sha(a.runner),physical_hardware=False,hardware_mapping_tested=False,
        same_tick_comparison=True,source_numeric_profile='reference-f64',sdk_precision='float32',
        compared='five traces and final refractory timer; exact count raster and native event presence',seed=20261013)
    (a.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(cases=len(reports),trace_values=sum(r['trace_values'] for r in reports),
        spikes=sum(r['spike_count'] for r in reports),spike_events=sum(r['spike_events'] for r in reports))))

if __name__=='__main__':main()
